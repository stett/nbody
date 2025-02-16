#if NBODY_GPU
#include "gpu.h"

using nbody::GPU;

GPU::GPU()
    : instance(make_instance())
    , device(make_device())
    , command_pool(make_command_pool())
    , command_buffer(make_command_buffer())
    , semaphore(device.createSemaphore({ }))
    , descriptor_pool(make_descriptor_pool())
    , descriptor_set_layout(make_descriptor_set_layout())
    , descriptor_set(make_descriptor_set())
    , pipeline_layout(make_pipeline_layout())
    , pipeline_integrate(make_pipeline(""))
    , pipeline_accelerate(make_pipeline(""))
{
    setup_storage_buffer();
    setup_command_buffer();
}

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

vk::raii::Device GPU::make_device()
{
    // get the first physical device
    vk::raii::PhysicalDevice physical_device = vk::raii::PhysicalDevices(instance).front();

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
    return { device, { { }, queue_family_index } };
}

vk::raii::CommandBuffer GPU::make_command_buffer()
{
    vk::CommandBufferAllocateInfo command_buffer_allocator_info( command_pool, vk::CommandBufferLevel::ePrimary, 1 );
    return std::move(vk::raii::CommandBuffers(device, command_buffer_allocator_info).front());
}

vk::raii::DescriptorPool GPU::make_descriptor_pool()
{
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
    };
    vk::DescriptorPoolCreateInfo create_info({ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet }, 2, pool_sizes);
    return { device, create_info };
}

vk::raii::DescriptorSetLayout GPU::make_descriptor_set_layout()
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings =
    {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
    };

    return { device, { { }, bindings } };
}

vk::raii::DescriptorSet GPU::make_descriptor_set()
{
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptor_pool,       // The descriptor pool handle
        1,                      // Number of descriptor sets to allocate
        &*descriptor_set_layout // Pointer to the descriptor set layout
    );

    // Allocate the descriptor set(s) using the RAII device method:
    std::vector<vk::raii::DescriptorSet> descriptor_sets = device.allocateDescriptorSets(allocInfo);
    assert(descriptor_sets.size() == 1);
    return std::move(descriptor_sets.front());
}

vk::raii::PipelineLayout GPU::make_pipeline_layout()
{
    vk::PipelineLayoutCreateInfo pipeline_layout_create_info({ }, { *descriptor_set_layout }, { });
    return { device, pipeline_layout_create_info };
}

vk::raii::Pipeline GPU::make_pipeline(const std::string& code)
{
    // create the shader module
    vk::ShaderModuleCreateInfo shader_module_create_info({ }, code.size(), reinterpret_cast<const uint32_t*>(code.data()));
    vk::ShaderModule shader_module = device.createShaderModule(shader_module_create_info);

    // We want to use as much shared memory for the compute shader invocations as available, so we calculate it based on the device limits and pass it to the shader via specialization constants
    //uint32_t sharedDataSize = std::min((uint32_t)1024, (uint32_t)(vulkanDevice->properties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));
    //vk::SpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    //vk::SpecializationInfo specialization_info = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(int32_t), &sharedDataSize);
    //computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

    // create the pipeline
    vk::PipelineShaderStageCreateInfo shader_stage_create_info({ }, vk::ShaderStageFlagBits::eCompute, shader_module, "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info({ }, shader_stage_create_info, *pipeline_layout, { }, -1);
    return { device, nullptr, compute_pipeline_create_info };
}

void GPU::setup_storage_buffer()
{
    /*
    // ...
    // We mark a few particles as attractors that move along a given path, these will pull in the other particles
    std::vector<glm::vec3> attractors = {
            glm::vec3(5.0f, 0.0f, 0.0f),
            glm::vec3(-5.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 5.0f),
            glm::vec3(0.0f, 0.0f, -5.0f),
            glm::vec3(0.0f, 4.0f, 0.0f),
            glm::vec3(0.0f, -8.0f, 0.0f),
    };

    numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;

    // Initial particle positions
    std::vector<Particle> particleBuffer(numParticles);

    std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned)time(nullptr));
    std::normal_distribution<float> rndDist(0.0f, 1.0f);

    for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++)
    {
        for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++)
        {
            Particle& particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];

            // First particle in group as heavy center of gravity
            if (j == 0)
            {
                particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f);
                particle.vel = glm::vec4(glm::vec4(0.0f));
            }
            else
            {
                // Position
                glm::vec3 position(attractors[i] + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)) * 0.75f);
                float len = glm::length(glm::normalize(position - attractors[i]));
                position.y *= 2.0f - (len * len);

                // Velocity
                glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
                glm::vec3 velocity = glm::cross((position - attractors[i]), angular) + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine) * 0.025f);

                float mass = (rndDist(rndEngine) * 0.5f + 0.5f) * 75.0f;
                particle.pos = glm::vec4(position, mass);
                particle.vel = glm::vec4(velocity, 0.0f);
            }

            // Color gradient offset
            particle.vel.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
        }
    }

    compute.uniformData.particleCount = numParticles;

    VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

    // Staging
    // SSBO won't be changed on the host after upload so copy to device local memory

    vks::Buffer stagingBuffer;

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, storageBufferSize, particleBuffer.data());
    // The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &storageBuffer, storageBufferSize);

    // Copy from staging buffer to storage buffer
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = storageBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffer.buffer, 1, &copyRegion);
    // Execute a transfer barrier to the compute queue, if necessary
    if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
    {
        VkBufferMemoryBarrier buffer_barrier =
                {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                        0,
                        graphics.queueFamilyIndex,
                        compute.queueFamilyIndex,
                        storageBuffer.buffer,
                        0,
                        storageBuffer.size
                };

        vkCmdPipelineBarrier(
                copyCmd,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
    }
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();
    */
}

void GPU::setup_command_buffer()
{
    command_buffer.begin(vk::CommandBufferBeginInfo());

    /*

    // Acquire barrier
    if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
    {
        VkBufferMemoryBarrier buffer_barrier =
                {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        0,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        graphics.queueFamilyIndex,
                        compute.queueFamilyIndex,
                        storageBuffer.buffer,
                        0,
                        storageBuffer.size
                };

        vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
    }

    // First pass: Calculate particle movement
    // -------------------------------------------------------------------------------------------------------
    vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineCalculate);
    vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
    vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

    // Add memory barrier to ensure that the computer shader has finished writing to the buffer
    VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
    bufferBarrier.buffer = storageBuffer.buffer;
    bufferBarrier.size = storageBuffer.descriptor.range;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    // Transfer ownership if compute and graphics queue family indices differ
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            1, &bufferBarrier,
            0, nullptr);

    // Second pass: Integrate particles
    // -------------------------------------------------------------------------------------------------------
    vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineIntegrate);
    vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

    // Release barrier
    if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
    {
        VkBufferMemoryBarrier buffer_barrier =
                {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        0,
                        compute.queueFamilyIndex,
                        graphics.queueFamilyIndex,
                        storageBuffer.buffer,
                        0,
                        storageBuffer.size
                };

        vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
    }
    */

    command_buffer.end();
}

#endif
