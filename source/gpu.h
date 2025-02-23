#pragma once
#if NBODY_GPU
#include <vector>
#include "vulkan/vulkan_raii.hpp"
#include "nbody/body.h"
#include "nbody/bhtree.h"
#include "nbody/constants.h"

namespace nbody
{
    enum class Mode : int { N2 = 0, NLogN = 1 };

    struct PushConstants
    {
        float dt = 0;
        float theta = 0;
        float G = nbody::G;
        int num_bodies = 0;
        int num_nodes = 0;
        Mode mode = Mode::NLogN;
    };

    struct Buffer
    {
        vk::DeviceSize size;
        vk::BufferUsageFlags usage;
        vk::MemoryPropertyFlags properties;
        vk::raii::PhysicalDevice& physical_device;
        vk::raii::Device& device;
        vk::raii::Buffer buffer;
        vk::raii::DeviceMemory memory;

        Buffer(
            vk::raii::PhysicalDevice& physical_device,
            vk::raii::Device& device,
            vk::DeviceSize size,
            vk::BufferUsageFlags usage,
            vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        // allocate gpu memory
        void allocate(size_t size);

        // copy data to the buffer, potentially resizing
        void write(const void* data, size_t data_size);

        // copy data from the buffer
        void read(void* data, size_t data_size) const;
    };

    class GPU
    {
    public:

        GPU();

        void write(const std::vector<Body>& bodies, const std::vector<bh::Node>& nodes);
        void read(std::vector<Body>& bodies);
        void integrate(float dt);
        void accelerate(float theta, Mode mode);

    private:

        // RAII vk objects
        vk::raii::Context context;
        vk::raii::Instance instance;
        vk::raii::PhysicalDevice physical_device;
        vk::raii::Device device;
        vk::raii::CommandPool command_pool;
        vk::raii::CommandBuffer command_buffer;
        vk::raii::Semaphore semaphore;
        vk::raii::DescriptorPool descriptor_pool;
        vk::raii::DescriptorSetLayout descriptor_set_layout;
        vk::raii::DescriptorSet descriptor_set;
        vk::raii::PipelineLayout pipeline_layout;
        vk::raii::ShaderModule shader_integrate;
        vk::raii::ShaderModule shader_accelerate;
        vk::raii::Pipeline pipeline_integrate;
        vk::raii::Pipeline pipeline_accelerate;
        nbody::Buffer buffer_bodies;
        nbody::Buffer buffer_nodes;

        // cached vk data
        uint32_t queue_family_index;

        // constant values for shaders
        PushConstants push_constants;

        // member initializer functions
        vk::raii::Instance make_instance();
        vk::raii::PhysicalDevice make_physical_device();
        vk::raii::Device make_device();
        vk::raii::CommandPool make_command_pool();
        vk::raii::CommandBuffer make_command_buffer();
        vk::raii::DescriptorPool make_descriptor_pool();
        vk::raii::DescriptorSetLayout make_descriptor_set_layout();
        vk::raii::DescriptorSet make_descriptor_set();
        vk::raii::PipelineLayout make_pipeline_layout();
        vk::raii::ShaderModule make_shader(const std::string& glsl);
        vk::raii::Pipeline make_pipeline(vk::raii::ShaderModule& shader);

        template <typename Type>
        nbody::Buffer make_buffer(
            uint32_t num,
            vk::BufferUsageFlags flags =
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst)
        { return { physical_device, device, sizeof(Type) * num, flags }; }

    public:

        static std::vector<uint32_t> glsl_to_spv(const std::string& glsl, const std::string& identifier = "unidentified");

        static uint32_t find_memory_type(vk::PhysicalDeviceMemoryProperties const& memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask);

        static vk::raii::DeviceMemory alloc_device_memory(
                vk::raii::Device const& device,
                vk::PhysicalDeviceMemoryProperties const& memoryProperties,
                vk::MemoryRequirements const& memoryRequirements,
                vk::MemoryPropertyFlags memoryPropertyFlags);
    };
}
#endif