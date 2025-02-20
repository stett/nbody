#if NBODY_GPU
#include <iostream>
#include <string>
#include "gpu.h"
#include "shaderc/shaderc.hpp"

using nbody::GPU;

static const std::string glsl_integrate = R"glsl(
    #version 450
    layout(local_size_x = 256) in;

    struct Body
    {
        float mass;
        float radius;
        vec3 pos;
        vec3 vel;
        vec3 acc;
    };

    layout(std430, binding = 0) buffer Bodies {
        Body bodies[];
    };

    layout(push_constant) uniform PushConstants {
        float dt;
        int num_bodies;
    } pc;

    const float G = 6.67430e-11;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(pc.num_bodies))
            return;

        vec3 pos = bodies[i].pos;
        vec3 vel = bodies[i].vel;
        vec3 acc = bodies[i].acc;

        // integrate using semi-implicit euler
        vel += acc * pc.dt;
        pos += vel * pc.dt;

        // update body info
        //bodies[i].pos = pos;
        //bodies[i].vel = vel;
        bodies[i].pos = vec3(0);
        bodies[i].vel = vec3(0);
    }
)glsl";

GPU::GPU()
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
    , shader_integrate(make_shader(glsl_integrate))
    , pipeline_integrate(make_pipeline(shader_integrate))
    , buffer_bodies(make_buffer_bodies(0))
{ }

vk::raii::Instance GPU::make_instance()
{
    // initialize the vk::ApplicationInfo structure
    vk::ApplicationInfo app_info("nbody", 1, "nbody", 1, VK_API_VERSION_1_4);

    // Specify required instance extensions, including the portability enumeration extension.
    std::vector<const char *> extensions = {"VK_KHR_portability_enumeration"};

    // initialize the instance create info
    vk::InstanceCreateInfo instance_create_info(
        vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, // creation flags
        &app_info,
        {}, // enabled layers
        extensions);

    return { context, instance_create_info };
}

vk::raii::PhysicalDevice GPU::make_physical_device()
{
    // get the first physical device
    return vk::raii::PhysicalDevices(instance).front();
}

vk::raii::Device GPU::make_device()
{
    // find the index of the first queue family that supports compute
    auto queue_family_properties = physical_device.getQueueFamilyProperties();
    auto compute_queue_family_properties =
        std::find_if(
            queue_family_properties.begin(),
            queue_family_properties.end(),
            [](vk::QueueFamilyProperties const &qfp)
            { return qfp.queueFlags & vk::QueueFlagBits::eCompute; });
    assert(compute_queue_family_properties != queue_family_properties.end());
    queue_family_index = static_cast<uint32_t>(std::distance(
        queue_family_properties.begin(),
        compute_queue_family_properties));

    // create a Device
    float queue_priority = 0.0f;
    vk::DeviceQueueCreateInfo device_queue_create_info({}, queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info({}, device_queue_create_info);

    return { physical_device, device_create_info };
}

vk::raii::CommandPool GPU::make_command_pool()
{
    return { device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_index } };
}

vk::raii::CommandBuffer GPU::make_command_buffer()
{
    vk::CommandBufferAllocateInfo command_buffer_allocator_info( command_pool, vk::CommandBufferLevel::ePrimary, 1 );
    return std::move(vk::raii::CommandBuffers(device, command_buffer_allocator_info).front());
}

vk::raii::DescriptorPool GPU::make_descriptor_pool()
{
    // NOTE: I don't know how to calculate how many of these I actually need...
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
    };
    return { device, { { vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet }, 1, pool_sizes } };
}

vk::raii::DescriptorSetLayout GPU::make_descriptor_set_layout()
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings =
    {
        { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
    };

    return { device, { { }, bindings } };
}

vk::raii::DescriptorSet GPU::make_descriptor_set()
{
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(descriptor_pool, *descriptor_set_layout);
    return std::move(vk::raii::DescriptorSets(device, descriptorSetAllocateInfo).front());
}

vk::raii::PipelineLayout GPU::make_pipeline_layout()
{
    vk::PushConstantRange push_constant_range(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants));
    return { device, { { }, { *descriptor_set_layout }, push_constant_range } };
}

vk::raii::ShaderModule GPU::make_shader(const std::string& glsl)
{
    const std::vector<uint32_t> spv = glsl_to_spv(glsl);
    return { device, { vk::ShaderModuleCreateFlags(), spv } };
}

vk::raii::Pipeline GPU::make_pipeline(vk::raii::ShaderModule& shader)
{
    // create the pipeline
    vk::PipelineShaderStageCreateInfo shader_stage_create_info({ }, vk::ShaderStageFlagBits::eCompute, *shader, "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info({ }, shader_stage_create_info, *pipeline_layout, { }, -1);
    return { device, nullptr, compute_pipeline_create_info };
}

nbody::Buffer GPU::make_buffer_bodies(uint32_t new_num_bodies)
{
    return {
        physical_device, device, sizeof(Body) * new_num_bodies,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst };
}

void GPU::integrate(std::vector<Body>& bodies, const float dt)
{
    // update body buffer
    buffer_bodies.write(bodies.data(), sizeof(Body) * bodies.size());
    vk::DescriptorBufferInfo descriptor_buffer_info(buffer_bodies.buffer, 0, buffer_bodies.size);
    vk::WriteDescriptorSet write_descriptor_set(
            descriptor_set,
            0, // destination binding
            0, // starting array element
            1, // descriptor count
            vk::DescriptorType::eStorageBuffer,
            nullptr,
            &descriptor_buffer_info);
    device.updateDescriptorSets(write_descriptor_set, { });

    // update push constant values
    push_constants.dt = dt;
    push_constants.num_bodies = (int)bodies.size();

    // set up command to run the integrate pipeline
    command_buffer.begin({ });
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_integrate);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, { descriptor_set }, { });
    command_buffer.pushConstants<PushConstants>(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    const uint32_t group_count = (bodies.size() + 255) / 256;
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

    // map the memory back to the bodies array
    buffer_bodies.read(bodies.data(), buffer_bodies.size);
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
    memory = GPU::alloc_device_memory(device, physical_device.getMemoryProperties(), buffer.getMemoryRequirements(), properties);
    buffer.bindMemory(memory, 0);
}

void nbody::Buffer::write(const void* data, const size_t data_size)
{
    // resize, only if growing
    if (data_size > size)
    {
        allocate(data_size);
    }

    // copy data to gpu mapped memory
    void* target = memory.mapMemory(0, data_size);
    memcpy(target, data, data_size);
    memory.unmapMemory();
}

void nbody::Buffer::read(void* data, const size_t data_size) const
{
    assert(data_size <= size);
    void* source = memory.mapMemory(0, data_size);
    std::memcpy(data, source, data_size);
    memory.unmapMemory();
}

std::vector<uint32_t> GPU::glsl_to_spv(const std::string& glsl, const std::string& identifier)
{
    // create a shaderc compiler instance & options object
    shaderc::Compiler compiler;
    shaderc::CompileOptions options; // todo: set optimization level?
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
    options.SetGenerateDebugInfo();
    options.SetOptimizationLevel(shaderc_optimization_level_zero);

    // compile the glsl
    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
        glsl,
        shaderc_shader_kind::shaderc_glsl_compute_shader,
        identifier.c_str(),
        options
    );

    // Check for compilation errors.
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << "Shader compilation error: " << result.GetErrorMessage() << std::endl;
        assert(false);
        return { };
    }

    // Get the SPIR-V code as a vector of uint32_t.
    return { result.cbegin(), result.cend() };
}

uint32_t GPU::find_memory_type(
    vk::PhysicalDeviceMemoryProperties const& memoryProperties,
    uint32_t typeBits,
    vk::MemoryPropertyFlags requirementsMask)
{
    uint32_t typeIndex = uint32_t( ~0 );
    for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ )
    {
        if ( ( typeBits & 1 ) && ( ( memoryProperties.memoryTypes[i].propertyFlags & requirementsMask ) == requirementsMask ) )
        {
            typeIndex = i;
            break;
        }
        typeBits >>= 1;
    }
    assert( typeIndex != uint32_t( ~0 ) );
    return typeIndex;
}

vk::raii::DeviceMemory GPU::alloc_device_memory(
    vk::raii::Device const& device,
    vk::PhysicalDeviceMemoryProperties const& memoryProperties,
    vk::MemoryRequirements const& memoryRequirements,
    vk::MemoryPropertyFlags memoryPropertyFlags)
{
    uint32_t memoryTypeIndex = find_memory_type( memoryProperties, memoryRequirements.memoryTypeBits, memoryPropertyFlags );
    vk::MemoryAllocateInfo memoryAllocateInfo( memoryRequirements.size, memoryTypeIndex );
    return { device, memoryAllocateInfo };
}

#endif
