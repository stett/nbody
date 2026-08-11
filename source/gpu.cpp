#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include "gpu.h"
#include "shaders/accelerate.h"
#include "shaders/integrate.h"

using nbody::GpuDevice;


GpuDevice::GpuDevice()
    : instance(make_instance())
    , physical_device(make_physical_device())
    , device(make_device())
    , command_pool(make_command_pool())
    , command_buffer(make_command_buffer())
    , semaphore(device.createSemaphore({ }))
    , descriptor_pool(make_descriptor_pool())
    , descriptor_set_layout(make_descriptor_set_layout())
    , descriptor_set(make_descriptor_set())
    , pipeline_layout(make_pipeline_layout())
    , shader_integrate(make_shader(spv_integrate))
    , shader_accelerate(make_shader(spv_accelerate))
    , pipeline_integrate(make_pipeline(shader_integrate))
    , pipeline_accelerate(make_pipeline(shader_accelerate))
    , buffer_bodies(make_buffer<Body>(0))
    , buffer_nodes(make_buffer<bh::Node>(0))
{ }

std::string GpuDevice::probe() noexcept
{
    try
    {
        vk::raii::Context context;
        vk::ApplicationInfo app_info("nbody", 1, "nbody", 1, VK_API_VERSION_1_4);
        std::vector<const char*> extensions = { "VK_KHR_portability_enumeration" };
        vk::raii::Instance instance(
            context,
            vk::InstanceCreateInfo(
                vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
                &app_info,
                {},
                extensions));

        vk::raii::PhysicalDevices devices(instance);
        if (devices.empty())
            return "no Vulkan physical devices";

        for (const vk::raii::PhysicalDevice& device : devices)
            for (const vk::QueueFamilyProperties& family : device.getQueueFamilyProperties())
                if (family.queueFlags & vk::QueueFlagBits::eCompute)
                    return {};

        return "no Vulkan queue family supports compute";
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    catch (...)
    {
        return "unknown Vulkan error";
    }
}

vk::raii::Instance GpuDevice::make_instance()
{
    // initialize the vk::ApplicationInfo structure
    vk::ApplicationInfo app_info("nbody", 1, "nbody", 1, VK_API_VERSION_1_4);

    // Specify required instance extensions, including the portability enumeration extension.
    std::vector<const char *> extensions = { "VK_KHR_portability_enumeration" };

    // initialize the instance create info
    vk::InstanceCreateInfo instance_create_info(
        vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
        &app_info,
        {}, // enabled layers
        extensions);

    return { context, instance_create_info };
}

vk::raii::PhysicalDevice GpuDevice::make_physical_device()
{
    vk::raii::PhysicalDevices devices(instance);
    if (devices.empty())
        throw std::runtime_error("no Vulkan physical devices");

    // Pick the first device that can actually run compute rather than simply the first
    // enumerated one. probe() reports availability if *any* device supports compute, so
    // taking front() blindly could fail construction on a machine probe called usable.
    for (vk::raii::PhysicalDevice& device : devices)
        for (const vk::QueueFamilyProperties& family : device.getQueueFamilyProperties())
            if (family.queueFlags & vk::QueueFlagBits::eCompute)
                return device;

    throw std::runtime_error("no Vulkan queue family supports compute");
}

vk::raii::Device GpuDevice::make_device()
{
    // find the index of the first queue family that supports compute
    auto queue_family_properties = physical_device.getQueueFamilyProperties();
    auto compute_queue_family_properties =
        std::find_if(
            queue_family_properties.begin(),
            queue_family_properties.end(),
            [](vk::QueueFamilyProperties const &qfp)
            { return qfp.queueFlags & vk::QueueFlagBits::eCompute; });
    // Throw rather than assert: an assert vanishes in release, and this is a
    // recoverable, reportable condition -- the variant simply becomes unavailable.
    if (compute_queue_family_properties == queue_family_properties.end())
        throw std::runtime_error("no Vulkan queue family supports compute");

    queue_family_index = static_cast<uint32_t>(std::distance(
        queue_family_properties.begin(),
        compute_queue_family_properties));

    // create a Device
    float queue_priority = 0.0f;
    vk::DeviceQueueCreateInfo device_queue_create_info({}, queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info({}, device_queue_create_info);

    return { physical_device, device_create_info };
}

vk::raii::CommandPool GpuDevice::make_command_pool()
{
    return { device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_index } };
}

vk::raii::CommandBuffer GpuDevice::make_command_buffer()
{
    vk::CommandBufferAllocateInfo command_buffer_allocator_info( command_pool, vk::CommandBufferLevel::ePrimary, 1 );
    return std::move(vk::raii::CommandBuffers(device, command_buffer_allocator_info).front());
}

vk::raii::DescriptorPool GpuDevice::make_descriptor_pool()
{
    // The pool must cover every descriptor in every set allocated from it. We allocate
    // one set from the layout below, which has two storage buffer bindings, so we need
    // two storage buffer descriptors. Under-sizing this fails with ErrorOutOfPoolMemory
    // on drivers that enforce it (e.g. MoltenVK).
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2)
    };
    return { device, { { vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet }, 1, pool_sizes } };
}

vk::raii::DescriptorSetLayout GpuDevice::make_descriptor_set_layout()
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings =
    {
        { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
    };

    return { device, { { }, bindings } };
}

vk::raii::DescriptorSet GpuDevice::make_descriptor_set()
{
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(descriptor_pool, *descriptor_set_layout);
    return std::move(vk::raii::DescriptorSets(device, descriptorSetAllocateInfo).front());
}

vk::raii::PipelineLayout GpuDevice::make_pipeline_layout()
{
    vk::PushConstantRange push_constant_range(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants));
    return { device, { { }, { *descriptor_set_layout }, push_constant_range } };
}

vk::raii::ShaderModule GpuDevice::make_shader(const unsigned char* spv, size_t size)
{
    return { device, { { }, size, reinterpret_cast<const uint32_t*>(spv) } };
}

vk::raii::Pipeline GpuDevice::make_pipeline(vk::raii::ShaderModule& shader)
{
    // create the pipeline
    vk::PipelineShaderStageCreateInfo shader_stage_create_info({ }, vk::ShaderStageFlagBits::eCompute, *shader, "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info({ }, shader_stage_create_info, *pipeline_layout, { }, -1);
    return { device, nullptr, compute_pipeline_create_info };
}

void GpuDevice::write(const std::vector<Body>& bodies, const std::vector<bh::Node>& nodes)
{
    // update buffers
    buffer_bodies.write(bodies.data(), sizeof(Body) * bodies.size());
    buffer_nodes.write(nodes.data(), sizeof(bh::Node) * nodes.size());

    // update descriptor sets
    vk::DescriptorBufferInfo descriptor_buffer_info_bodies = { buffer_bodies.buffer, 0, buffer_bodies.size };
    vk::DescriptorBufferInfo descriptor_buffer_info_nodes = { buffer_nodes.buffer, 0, buffer_nodes.size };
    std::array<vk::WriteDescriptorSet, 2> descriptor_set_writes
    {
        vk::WriteDescriptorSet{
            *descriptor_set,
            0, // destination binding
            0, // starting array element
            1, // descriptor count
            vk::DescriptorType::eStorageBuffer,
            nullptr,
            &descriptor_buffer_info_bodies
        },

        vk::WriteDescriptorSet{
            *descriptor_set,
            1, // destination binding
            0, // starting array element
            1, // descriptor count
            vk::DescriptorType::eStorageBuffer,
            nullptr,
            &descriptor_buffer_info_nodes
        }
    };
    device.updateDescriptorSets(descriptor_set_writes, { });

    // update push constant values
    push_constants.num_bodies = (int)bodies.size();
    push_constants.num_nodes = (int)nodes.size();
}

void GpuDevice::read(std::vector<Body>& bodies)
{
    // Copy back only what both sides can hold. buffer_bodies.size is the allocation,
    // which only ever grows, so reading that many bytes overruns `bodies` whenever the
    // body count has shrunk since the buffer was sized.
    const size_t want = bodies.size() * sizeof(Body);
    buffer_bodies.read(bodies.data(), std::min<size_t>(want, buffer_bodies.used));
}

void GpuDevice::integrate(const float dt, const float size, const bool wrap)
{
    // update relevant push constants
    push_constants.dt = dt;
    push_constants.size = size;
    push_constants.wrap = wrap ? 1 : 0;

    // set up command to run the integrate pipeline
    command_buffer.begin({ });
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_integrate);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, { descriptor_set }, { });
    command_buffer.pushConstants<PushConstants>(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    const uint32_t group_count = (push_constants.num_bodies + 255) / 256;
    command_buffer.dispatch(group_count, 1, 1);
    command_buffer.end();

    // submit the command and wait for the result
    vk::raii::Fence fence(device, vk::FenceCreateInfo{ });
    vk::raii::Queue queue = device.getQueue(queue_family_index, 0);
    const vk::SubmitInfo submit_info(
        0, // wait semaphore count
        nullptr, // wait semaphores
        nullptr, // wait destination stage mask flags
        1, // command buffer count
        &*command_buffer);
    queue.submit(submit_info, *fence);
    const vk::Result result = device.waitForFences({ *fence }, VK_TRUE, UINT64_MAX);
    assert(result == vk::Result::eSuccess);
}

void GpuDevice::accelerate(const float theta, const float gravity, const Mode mode)
{
    // update relevant push constants
    push_constants.theta = theta;
    push_constants.G = gravity;   // or set_gravity() would silently not reach the device
    push_constants.mode = mode;

    // set up command to run the accelerate pipeline
    command_buffer.begin({ });
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_accelerate);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, { descriptor_set }, { });
    command_buffer.pushConstants<PushConstants>(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    const uint32_t group_count = (push_constants.num_bodies + 255) / 256;
    command_buffer.dispatch(group_count, 1, 1);
    command_buffer.end();

    // submit the command and wait for the result
    vk::raii::Fence fence(device, vk::FenceCreateInfo{ });
    vk::raii::Queue queue = device.getQueue(queue_family_index, 0);
    const vk::SubmitInfo submit_info(
        0, // wait semaphore count
        nullptr, // wait semaphores
        nullptr, // wait destination stage mask flags
        1, // command buffer count
        &*command_buffer);
    queue.submit(submit_info, *fence);
    const vk::Result result = device.waitForFences({ *fence }, VK_TRUE, UINT64_MAX);
    assert(result == vk::Result::eSuccess);
}

nbody::Buffer::Buffer(
    vk::raii::PhysicalDevice& physical_device,
    vk::raii::Device& device,
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties)
    : size(size)
    , usage(usage)
    , properties(properties)
    , physical_device(physical_device)
    , device(device)
    , buffer(nullptr)
    , memory(nullptr)
{
    allocate(size);
}

void nbody::Buffer::allocate(const size_t _size)
{
    size = _size;
    if (size == 0) { return; }
    buffer = { device, { { }, size, usage } };
    memory = GpuDevice::alloc_device_memory(device, physical_device.getMemoryProperties(), buffer.getMemoryRequirements(), properties);
    buffer.bindMemory(memory, 0);
}

void nbody::Buffer::write(const void* data, const size_t data_size)
{
    // resize, only if growing
    if (data_size > size)
    {
        allocate(data_size);
    }

    // track how much of the (possibly larger) allocation is actually live
    used = data_size;
    if (used == 0) { return; }

    // copy data to gpu mapped memory
    void* target = memory.mapMemory(0, used);
    memcpy(target, data, used);
    memory.unmapMemory();
}

void nbody::Buffer::read(void* data, const size_t data_size) const
{
    assert(data_size <= size);
    if (data_size == 0) { return; }
    void* source = memory.mapMemory(0, data_size);
    std::memcpy(data, source, data_size);
    memory.unmapMemory();
}

namespace
{
    // Sentinel for "no memory type satisfies this mask". Distinct from any real index.
    constexpr uint32_t no_memory_type = uint32_t( ~0 );

    // Like find_memory_type, but reports failure instead of asserting -- alloc_device_memory
    // deliberately asks for a mask it may not get and falls back.
    uint32_t try_find_memory_type(
        vk::PhysicalDeviceMemoryProperties const& memoryProperties,
        uint32_t typeBits,
        vk::MemoryPropertyFlags requirementsMask)
    {
        for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ )
        {
            if ( ( typeBits & 1 ) && ( ( memoryProperties.memoryTypes[i].propertyFlags & requirementsMask ) == requirementsMask ) )
                return i;
            typeBits >>= 1;
        }
        return no_memory_type;
    }

    // Decode only the flags that matter for placement. Avoids vk::to_string, which moved
    // headers between SDK versions.
    std::string describe_memory_flags(const vk::MemoryPropertyFlags flags)
    {
        std::string out;
        const auto add = [&out](const char* name) { if (!out.empty()) out += '|'; out += name; };
        if (flags & vk::MemoryPropertyFlagBits::eDeviceLocal)  add("DEVICE_LOCAL");
        if (flags & vk::MemoryPropertyFlagBits::eHostVisible)  add("HOST_VISIBLE");
        if (flags & vk::MemoryPropertyFlagBits::eHostCoherent) add("HOST_COHERENT");
        if (flags & vk::MemoryPropertyFlagBits::eHostCached)   add("HOST_CACHED");
        return out.empty() ? "none" : out;
    }

    // Dump the memory type table the first time we allocate. Where the storage buffers land
    // decides whether the barnes-hut traversal reads from VRAM or across PCIe, and that is
    // not visible from any external profiler on this app (the vulkan instance is
    // compute-only and never presents, so the frame-based tools have nothing to hook).
    void log_memory_types_once(vk::PhysicalDeviceMemoryProperties const& memoryProperties)
    {
        static bool logged = false;
        if (logged) { return; }
        logged = true;

        std::cerr << "nbody: vulkan memory types:" << std::endl;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            const vk::MemoryType& type = memoryProperties.memoryTypes[i];
            std::cerr
                << "nbody:   [" << i << "] heap " << type.heapIndex
                << " (" << (memoryProperties.memoryHeaps[type.heapIndex].size >> 20) << " MiB) "
                << describe_memory_flags(type.propertyFlags) << std::endl;
        }
    }
}

uint32_t GpuDevice::find_memory_type(
    vk::PhysicalDeviceMemoryProperties const& memoryProperties,
    uint32_t typeBits,
    vk::MemoryPropertyFlags requirementsMask)
{
    const uint32_t typeIndex = try_find_memory_type( memoryProperties, typeBits, requirementsMask );
    assert( typeIndex != no_memory_type );
    return typeIndex;
}

vk::raii::DeviceMemory GpuDevice::alloc_device_memory(
    vk::raii::Device const& device,
    vk::PhysicalDeviceMemoryProperties const& memoryProperties,
    vk::MemoryRequirements const& memoryRequirements,
    vk::MemoryPropertyFlags memoryPropertyFlags)
{
    // Prefer memory that is device-local *and* mappable (the BAR window; the whole of VRAM
    // when resizable BAR is on). The buffers are read by every shader invocation and the
    // barnes-hut traversal in particular is a divergent, latency-bound pointer chase --
    // serving those reads from system RAM over PCIe costs far more than the brute-force
    // path, whose warp-uniform sequential sweep both coalesces and caches.
    //
    // find_memory_type takes the first type satisfying the mask, and on discrete NVIDIA the
    // plain HOST_VISIBLE|HOST_COHERENT system-memory type is enumerated before the
    // device-local one, so asking for the base mask alone reliably lands in system RAM.
    //
    // Fall back when there is no such type, or when the request does not fit the BAR heap:
    // pre-resizable-BAR parts expose only 256MB there, and overflowing it fails allocation.
    // Callers map this memory, so the fallback must stay host-visible.
    log_memory_types_once( memoryProperties );

    // For a mappable request, treat eDeviceLocal as a preference rather than a requirement:
    // ask for it, but keep a host-visible-only mask to retreat to. Requiring it in the
    // fallback too would defeat the point, since the BAR type is the only one that has both
    // and it is exactly the one that just failed.
    const bool mappable = bool( memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible );
    const vk::MemoryPropertyFlags fallback_flags = mappable
        ? ( memoryPropertyFlags & ~vk::MemoryPropertyFlags( vk::MemoryPropertyFlagBits::eDeviceLocal ) )
        : memoryPropertyFlags;

    const uint32_t fallback = find_memory_type( memoryProperties, memoryRequirements.memoryTypeBits, fallback_flags );
    uint32_t preferred = no_memory_type;
    if ( mappable )
    {
        preferred = try_find_memory_type(
            memoryProperties,
            memoryRequirements.memoryTypeBits,
            fallback_flags | vk::MemoryPropertyFlagBits::eDeviceLocal );

        // The heap size is an upper bound, not the free space -- the other buffer is already
        // resident, and a grow holds both the old and new allocation at once. Screening on it
        // avoids the obviously-hopeless request; the try/catch below covers the rest.
        if ( preferred != no_memory_type &&
             memoryRequirements.size > memoryProperties.memoryHeaps[memoryProperties.memoryTypes[preferred].heapIndex].size )
            preferred = no_memory_type;
    }

    if ( preferred != no_memory_type && preferred != fallback )
    {
        try
        {
            return { device, vk::MemoryAllocateInfo( memoryRequirements.size, preferred ) };
        }
        catch ( const vk::SystemError& )
        {
            // A small BAR window fills up long before VRAM does. Losing device-local
            // placement costs performance; failing the allocation would cost the frame.
            std::cerr
                << "nbody: device-local BAR allocation of " << ( memoryRequirements.size >> 20 )
                << " MiB failed, falling back to host memory (shader reads will cross PCIe)"
                << std::endl;
        }
    }

    return { device, vk::MemoryAllocateInfo( memoryRequirements.size, fallback ) };
}
