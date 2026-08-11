#pragma once
#include <vector>
#include "vulkan/vulkan_raii.hpp"
#include "nbody/body.h"
#include "nbody/bhtree.h"
#include "nbody/constants.h"

namespace nbody
{
    enum class Mode : int { N2 = 0, NLogN = 1 };

    // Must match the push_constant block in shaders/include/common.glsl field for
    // field — the whole struct is memcpy'd across via pushConstants<PushConstants>.
    // Append new fields at the end so existing offsets stay put.
    struct PushConstants
    {
        float dt = 0;
        float theta = 0;
        float G = nbody::G;
        int num_bodies = 0;
        int num_nodes = 0;
        Mode mode = Mode::NLogN;
        float size = 0;
        int wrap = 1;
    };

    struct Buffer
    {
        // `size` is the allocated capacity, which only ever grows. `used` is how many
        // bytes of it currently hold live data — reads must clamp to `used`, otherwise
        // shrinking the body count copies stale capacity past the end of the caller's
        // vector.
        vk::DeviceSize size;
        vk::DeviceSize used = 0;
        vk::BufferUsageFlags usage;
        vk::MemoryPropertyFlags properties;
        vk::raii::PhysicalDevice& physical_device;
        vk::raii::Device& device;
        vk::raii::Buffer buffer;
        vk::raii::DeviceMemory memory;

        // Persistent mapping of `memory`, established by allocate() and valid whenever
        // `size` is non-zero. Null only for an empty allocation.
        void* mapped = nullptr;

        Buffer(
            vk::raii::PhysicalDevice& physical_device,
            vk::raii::Device& device,
            vk::DeviceSize size,
            vk::BufferUsageFlags usage,
            vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        // allocate gpu memory
        void allocate(size_t size);

        // copy data to the buffer, potentially resizing
        void write(const void* data, size_t data_size);

        // copy data from the buffer
        void read(void* data, size_t data_size) const;
    };

    // A vulkan compute device plus the pipelines the simulation runs on it. Owned by
    // Context and shared between the GPU variants, so switching between them does not
    // pay for shader compilation again.
    class GpuDevice
    {
    public:

        // Throws if the device cannot be brought up.
        GpuDevice();

        // Is there a compute-capable Vulkan device on this machine? Returns an empty
        // string if so, otherwise a human-readable reason. Never throws.
        //
        // Deliberately cheap: creates an instance and inspects queue families, but does
        // not create a logical device, allocate buffers or build pipelines. Full
        // construction failure is reported separately, when the variant is first selected.
        static std::string probe() noexcept;

        void write(const std::vector<Body>& bodies, const std::vector<bh::Node>& nodes);
        void read(std::vector<Body>& bodies);
        void integrate(float dt, float size, bool wrap);
        void accelerate(float theta, float gravity, Mode mode);

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

        // Whether the device came up with VK_EXT_frame_boundary. Assigned by make_device()
        // and so, like queue_family_index, deliberately left without a default member
        // initializer: those run after the member init list and would clobber it.
        bool frame_boundary_enabled;

        // Labels each frame-end submit so a capture tool can tell the steps apart. Not
        // touched by make_device(), so a default initializer is safe here.
        uint64_t frame_id = 0;

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
        vk::raii::ShaderModule make_shader(const unsigned char* spv, size_t size);

        template <size_t size>
        vk::raii::ShaderModule make_shader(const unsigned char (&spv)[size])
        { return std::move(make_shader(spv, size)); }

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

        static uint32_t find_memory_type(vk::PhysicalDeviceMemoryProperties const& memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask);

        static vk::raii::DeviceMemory alloc_device_memory(
                vk::raii::Device const& device,
                vk::PhysicalDeviceMemoryProperties const& memoryProperties,
                vk::MemoryRequirements const& memoryRequirements,
                vk::MemoryPropertyFlags memoryPropertyFlags);
    };
}