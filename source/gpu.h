#pragma once
#if NBODY_GPU
#include <iostream>
#include "vulkan/vulkan_raii.hpp"

namespace nbody
{
    class GPU
    {
    public:

        GPU();

    private:

        // RAII vk objects
        vk::raii::Context context;
        vk::raii::Instance instance;
        vk::raii::Device device;
        vk::raii::CommandPool command_pool;
        vk::raii::CommandBuffer command_buffer;
        vk::raii::Semaphore semaphore;
        vk::raii::DescriptorPool descriptor_pool;
        vk::raii::DescriptorSetLayout descriptor_set_layout;
        vk::raii::DescriptorSet descriptor_set;
        vk::raii::PipelineLayout pipeline_layout;
        vk::raii::Pipeline pipeline_integrate;
        vk::raii::Pipeline pipeline_accelerate;

        // cached vk data
        uint32_t queue_family_index;

        // member initializer functions
        vk::raii::Instance make_instance();
        vk::raii::Device make_device();
        vk::raii::CommandPool make_command_pool();
        vk::raii::CommandBuffer make_command_buffer();
        vk::raii::DescriptorPool make_descriptor_pool();
        vk::raii::DescriptorSetLayout make_descriptor_set_layout();
        vk::raii::DescriptorSet make_descriptor_set();
        vk::raii::PipelineLayout make_pipeline_layout();
        vk::raii::Pipeline make_pipeline(const std::string& code);

        // setup functions
        void setup_storage_buffer();
        void setup_command_buffer();
    };
}
#endif