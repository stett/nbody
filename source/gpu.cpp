#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include "gpu.h"
#include "nbody/profile.h"
#include "shaders/accelerate.h"
#include "shaders/integrate.h"

using nbody::GpuDevice;

namespace
{
    // Opt-in vulkan diagnostics, off unless NBODY_VK_VERBOSE is set. Nothing here is needed
    // in a normal run, and only the presence of a value is checked, so any value will do.
    bool vulkan_verbose()
    {
        static const bool verbose = std::getenv("NBODY_VK_VERBOSE") != nullptr;
        return verbose;
    }

    // Report which tools have inserted themselves into this device, and whether the one
    // extension we want from them came with it.
    //
    // Worth having because the failure is otherwise mute: a capture tool that attached with
    // the wrong layer looks exactly like one that did not attach at all, and neither says so.
    // vkGetPhysicalDeviceToolProperties tells them apart -- an interception layer is
    // obliged to name itself here, so an empty list means nothing hooked the process, while
    // a populated list without VK_EXT_frame_boundary means something did but not the layer
    // that provides it.
    void log_attached_tools_once(const vk::raii::PhysicalDevice& physical_device, const bool frame_boundary)
    {
        if (!vulkan_verbose()) { return; }

        static bool logged = false;
        if (logged) { return; }
        logged = true;

        try
        {
            const std::vector<vk::PhysicalDeviceToolProperties> tools = physical_device.getToolProperties();
            if (tools.empty())
            {
                std::cerr << "nbody: no vulkan tools attached to this device" << std::endl;
            }
            else
            {
                std::cerr << "nbody: vulkan tools attached:" << std::endl;
                for (const vk::PhysicalDeviceToolProperties& tool : tools)
                    std::cerr
                        << "nbody:   " << tool.name.data()
                        << " " << tool.version.data()
                        << " -- " << tool.description.data() << std::endl;
            }
        }
        catch (const vk::SystemError& e)
        {
            std::cerr << "nbody: could not query vulkan tool properties: " << e.what() << std::endl;
        }

        std::cerr
            << "nbody: VK_EXT_frame_boundary "
            << (frame_boundary ? "present" : "absent -- only the capture layer provides it")
            << std::endl;
    }
}


GpuDevice::GpuDevice()
    : instance(make_instance())
    , physical_device(make_physical_device())
    , device(make_device())
    , queue(device.getQueue(queue_family_index, 0))
    , fence(device, vk::FenceCreateInfo{ })
    , command_pool(make_command_pool())
    , command_buffer(make_command_buffer())
    , descriptor_pool(make_descriptor_pool())
    , descriptor_set_layout(make_descriptor_set_layout())
    , descriptor_set(make_descriptor_set())
    , pipeline_layout(make_pipeline_layout())
    , shader_integrate(make_shader(spv_integrate))
    , shader_accelerate(make_shader(spv_accelerate))
    , pipeline_integrate(make_pipeline(shader_integrate))
    , pipeline_accelerate(make_pipeline(shader_accelerate))
    , buffer_bodies(make_device_buffer<Body>(0))
    , buffer_nodes(make_device_buffer<bh::Node>(0))
    , staging_bodies(make_staging_buffer<Body>(0))
    , staging_nodes(make_staging_buffer<bh::Node>(0))
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

    // Capture tools delimit their work by frame boundary, which normally means
    // vkQueuePresentKHR. This instance is compute-only -- make_instance() requests no
    // surface extension and there is no swapchain anywhere -- so to a tool it appears to
    // submit work forever without ever completing a frame. Nsight Graphics reports "no
    // supported graphics API detected" on attach, and a capture request queued against it
    // waits for a present that never arrives.
    //
    // VK_EXT_frame_boundary is the way out: it lets a non-presenting application mark the
    // end of a frame itself, by chaining a VkFrameBoundaryEXT onto a queue submission (see
    // integrate()). It is purely advisory, so if the driver does not offer it the only cost
    // is that the app stays undebuggable by those tools.
    std::vector<const char*> extensions;
    frame_boundary_enabled = false;
    for (const vk::ExtensionProperties& extension : physical_device.enumerateDeviceExtensionProperties())
    {
        if (std::strcmp(extension.extensionName.data(), VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME) == 0)
        {
            extensions.push_back(VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME);
            frame_boundary_enabled = true;
            break;
        }
    }

    // Absence is the normal case and says nothing is wrong, so it goes unremarked: no
    // driver implements this, it arrives with the capture tool's own layer. Announce only
    // the presence, which confirms the tool's layer took hold and frames will be marked.
    static bool logged_frame_boundary = false;
    if (frame_boundary_enabled && !logged_frame_boundary)
    {
        logged_frame_boundary = true;
        std::cerr
            << "nbody: VK_EXT_frame_boundary available -- marking frame ends for capture"
            << std::endl;
    }

    log_attached_tools_once(physical_device, frame_boundary_enabled);

    // Enabling the extension is not enough; the feature has to be switched on too, or the
    // VkFrameBoundaryEXT chained at submit time is ignored.
    vk::PhysicalDeviceFrameBoundaryFeaturesEXT frame_boundary_features(VK_TRUE);

    // create a Device
    float queue_priority = 0.0f;
    vk::DeviceQueueCreateInfo device_queue_create_info({}, queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info({}, device_queue_create_info, {}, extensions);
    if (frame_boundary_enabled)
        device_create_info.pNext = &frame_boundary_features;

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
    NBODY_PROFILE_ZONE();
    // Land the data in host memory and size the device buffers to match. The copy across
    // is recorded into the next command buffer rather than done here, so it runs on the
    // transfer hardware alongside everything else instead of on this thread.
    const size_t bodies_bytes = sizeof(Body) * bodies.size();
    const size_t nodes_bytes = sizeof(bh::Node) * nodes.size();

    staging_bodies.write(bodies.data(), bodies_bytes);
    staging_nodes.write(nodes.data(), nodes_bytes);
    buffer_bodies.reserve(bodies_bytes);
    buffer_nodes.reserve(nodes_bytes);
    upload_pending = true;

    // update descriptor sets -- the shaders bind the device buffers, never the staging pair
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
    NBODY_PROFILE_ZONE();
    // Read from staging, which record_readback() filled during the submission the caller
    // has already waited on. Copy back only what both sides can hold: the allocation only
    // ever grows, so reading all of it overruns `bodies` whenever the count has shrunk.
    const size_t want = bodies.size() * sizeof(Body);
    staging_bodies.read(bodies.data(), std::min<size_t>(want, staging_bodies.used));
}

// Copy whatever write() staged into the device buffers, and order the shaders after it.
void GpuDevice::record_upload()
{
    if (!upload_pending) { return; }
    upload_pending = false;

    if (staging_bodies.used > 0)
        command_buffer.copyBuffer(
            staging_bodies.buffer, buffer_bodies.buffer,
            vk::BufferCopy(0, 0, staging_bodies.used));

    if (staging_nodes.used > 0)
        command_buffer.copyBuffer(
            staging_nodes.buffer, buffer_nodes.buffer,
            vk::BufferCopy(0, 0, staging_nodes.used));

    const vk::MemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead);

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        { }, barrier, { }, { });
}

// Bring the bodies back to where the host can read them once the shaders are done.
void GpuDevice::record_readback()
{
    if (buffer_bodies.used == 0) { return; }

    const vk::MemoryBarrier before(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eTransferRead);

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        { }, before, { }, { });

    command_buffer.copyBuffer(
        buffer_bodies.buffer, staging_bodies.buffer,
        vk::BufferCopy(0, 0, buffer_bodies.used));

    // Make the transfer visible to the host reads that follow the fence.
    const vk::MemoryBarrier after(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eHostRead);

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eHost,
        { }, after, { }, { });
}

void GpuDevice::record_dispatch(vk::raii::Pipeline& pipeline)
{
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, { descriptor_set }, { });
    command_buffer.pushConstants<PushConstants>(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    const uint32_t group_count = (push_constants.num_bodies + 255) / 256;
    command_buffer.dispatch(group_count, 1, 1);
}

// Order one compute dispatch after another within a command buffer.
//
// Consecutive dispatches are otherwise free to overlap: submission order constrains when
// work *starts*, never when it finishes, and no memory dependency is implied. Since
// accelerate writes bodies[].acc and integrate reads it, the two need an explicit
// dependency or integrate can read accelerations that were never written. The barrier
// supplies both halves -- execution (all of the first dispatch completes before any of the
// second begins) and memory (the writes are made available and visible).
void GpuDevice::record_dispatch_barrier()
{
    const vk::MemoryBarrier barrier(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead);

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        { },
        barrier,
        { },
        { });
}

// Submit the recorded command buffer and block until the device has finished it.
//
// `frame_end` closes a frame for capture tools. Only the last submission of a simulation
// step should set it, or a tool would see each dispatch as a frame of its own.
void GpuDevice::submit_and_wait(const bool frame_end)
{
    NBODY_PROFILE_ZONE();
    device.resetFences({ *fence });

    vk::SubmitInfo submit_info(
        0, // wait semaphore count
        nullptr, // wait semaphores
        nullptr, // wait destination stage mask flags
        1, // command buffer count
        &*command_buffer);

    // The buffer is named as the frame's output so a tool can report what the frame
    // produced; imageCount stays 0, since a compute-only frame renders to nothing.
    vk::FrameBoundaryEXT frame_boundary;
    frame_boundary.flags = vk::FrameBoundaryFlagBitsEXT::eFrameEnd;
    frame_boundary.frameID = frame_id;
    frame_boundary.bufferCount = 1;
    frame_boundary.pBuffers = &*buffer_bodies.buffer;
    if (frame_end && frame_boundary_enabled)
    {
        submit_info.pNext = &frame_boundary;
        ++frame_id;
    }

    queue.submit(submit_info, *fence);

    {
        // Split from the submission: the only span that says how long the shaders took.
        NBODY_PROFILE_ZONE_NAMED("wait for device");
        const vk::Result result = device.waitForFences({ *fence }, VK_TRUE, UINT64_MAX);
        assert(result == vk::Result::eSuccess);
    }
}

void GpuDevice::set_accelerate_constants(const float theta, const float gravity, const Mode mode)
{
    push_constants.theta = theta;
    push_constants.G = gravity;   // or set_gravity() would silently not reach the device
    push_constants.mode = mode;
}

void GpuDevice::set_integrate_constants(const float dt, const float size, const bool wrap)
{
    push_constants.dt = dt;
    push_constants.size = size;
    push_constants.wrap = wrap ? 1 : 0;
}

void GpuDevice::integrate(const float dt, const float size, const bool wrap)
{
    set_integrate_constants(dt, size, wrap);

    command_buffer.begin({ });
    record_upload();
    record_dispatch(pipeline_integrate);
    record_readback();
    command_buffer.end();

    submit_and_wait(true);
}

void GpuDevice::accelerate(const float theta, const float gravity, const Mode mode)
{
    set_accelerate_constants(theta, gravity, mode);

    command_buffer.begin({ });
    record_upload();
    record_dispatch(pipeline_accelerate);
    record_readback();
    command_buffer.end();

    submit_and_wait(false);
}

// A whole simulation step in one submission.
//
// Running accelerate() and integrate() back to back costs two round trips: the host blocks
// on the first fence before it has even recorded the second dispatch, so the device
// finishes accelerating and then idles while the host wakes up and submits again. Recording
// both against one barrier keeps the ordering guarantee the fence was providing while
// leaving the device with work already queued behind the first dispatch.
//
// The separate entry points remain for callers that genuinely need one half on its own.
void GpuDevice::step(
    const float dt,
    const float theta,
    const float gravity,
    const Mode mode,
    const float size,
    const bool wrap)
{
    NBODY_PROFILE_ZONE();

    command_buffer.begin({ });
    record_upload();

    set_accelerate_constants(theta, gravity, mode);
    record_dispatch(pipeline_accelerate);

    record_dispatch_barrier();

    // Push constants are recorded into the command buffer, so re-pushing here applies to
    // the second dispatch only and leaves the first one's values alone.
    set_integrate_constants(dt, size, wrap);
    record_dispatch(pipeline_integrate);

    record_readback();
    command_buffer.end();

    submit_and_wait(true);
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

    // The previous mapping belonged to the memory object replaced below, so drop it before
    // anything can observe it pointing into a freed allocation.
    mapped = nullptr;

    if (size == 0) { return; }
    buffer = { device, { { }, size, usage } };
    memory = GpuDevice::alloc_device_memory(device, physical_device.getMemoryProperties(), buffer.getMemoryRequirements(), properties);
    buffer.bindMemory(memory, 0);

    // Device-local storage is not host-visible and cannot be mapped at all; it is reached
    // through a staging buffer instead.
    if (!(properties & vk::MemoryPropertyFlagBits::eHostVisible))
        return;

    // Map once and keep it for as long as the allocation lives. vkMapMemory is not a
    // pointer handout: the driver reserves address space and populates page tables, work
    // proportional to the size of the allocation. Mapping per access paid that on every
    // buffer on every frame. Vulkan explicitly permits a mapping to persist, and nothing
    // here benefits from letting it lapse.
    //
    // No flush or invalidate accompanies the copies below because every host-visible mask
    // requested here is also host-coherent.
    mapped = memory.mapMemory(0, size);
}

void nbody::Buffer::reserve(const size_t bytes)
{
    // resize, only if growing
    if (bytes > size)
    {
        allocate(bytes);
    }

    // track how much of the (possibly larger) allocation is actually live
    used = bytes;
}

void nbody::Buffer::write(const void* data, const size_t data_size)
{
    NBODY_PROFILE_ZONE();
    reserve(data_size);
    if (used == 0) { return; }
    assert(mapped != nullptr);
    memcpy(mapped, data, used);
}

void nbody::Buffer::read(void* data, const size_t data_size) const
{
    NBODY_PROFILE_ZONE();
    assert(data_size <= size);
    if (data_size == 0) { return; }
    assert(mapped != nullptr);
    std::memcpy(data, mapped, data_size);
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
    log_memory_types_once( memoryProperties );

    // Callers state exactly what they need and get it. There is no preference to express
    // any more: shader storage asks for DEVICE_LOCAL and lands in the full VRAM heap, while
    // staging asks for HOST_CACHED, which no device-local type offers and which therefore
    // resolves to system memory on its own.
    const uint32_t memoryTypeIndex = find_memory_type( memoryProperties, memoryRequirements.memoryTypeBits, memoryPropertyFlags );

    if ( vulkan_verbose() )
    {
        const vk::MemoryType& type = memoryProperties.memoryTypes[memoryTypeIndex];
        std::cerr
            << "nbody: allocating " << ( memoryRequirements.size >> 10 ) << " KiB"
            << " from memory type " << memoryTypeIndex
            << " (" << describe_memory_flags( type.propertyFlags ) << ")"
            << " on heap " << type.heapIndex << std::endl;
    }

    return { device, vk::MemoryAllocateInfo( memoryRequirements.size, memoryTypeIndex ) };
}
