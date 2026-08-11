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
        // `size` is non-zero. Null for an empty allocation, and for device-local memory
        // that was never host-visible in the first place -- write() and read() are only
        // meaningful on a staging buffer.
        void* mapped = nullptr;

        Buffer(
            vk::raii::PhysicalDevice& physical_device,
            vk::raii::Device& device,
            vk::DeviceSize size,
            vk::BufferUsageFlags usage,
            vk::MemoryPropertyFlags properties);

        // allocate gpu memory
        void allocate(size_t size);

        // Grow to hold `bytes` and record that many as live, without touching the contents.
        // For a device-local buffer, whose storage is only ever filled by a transfer, this
        // is the whole of what sizing means.
        void reserve(size_t bytes);

        // copy data to the buffer, potentially resizing. requires host-visible memory
        void write(const void* data, size_t data_size);

        // copy data from the buffer. requires host-visible memory
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

        // Accelerate and integrate in a single submission, ordered by a pipeline barrier.
        // Equivalent to accelerate() followed by integrate(), but without the host round
        // trip between them; prefer it whenever both halves are wanted.
        void step(float dt, float theta, float gravity, Mode mode, float size, bool wrap);

    private:

        // RAII vk objects
        vk::raii::Context context;
        vk::raii::Instance instance;
        vk::raii::PhysicalDevice physical_device;
        vk::raii::Device device;

        // The queue and the fence are the same objects every submission, so they are held
        // rather than built per dispatch. Initialized after `device` and so able to use the
        // queue_family_index that make_device() resolves.
        vk::raii::Queue queue;
        vk::raii::Fence fence;

        vk::raii::CommandPool command_pool;
        vk::raii::CommandBuffer command_buffer;
        vk::raii::DescriptorPool descriptor_pool;
        vk::raii::DescriptorSetLayout descriptor_set_layout;
        vk::raii::DescriptorSet descriptor_set;
        vk::raii::PipelineLayout pipeline_layout;
        vk::raii::ShaderModule shader_integrate;
        vk::raii::ShaderModule shader_accelerate;
        vk::raii::Pipeline pipeline_integrate;
        vk::raii::Pipeline pipeline_accelerate;
        // The buffers the shaders read are device-local and not mappable. The host reaches
        // them only through the staging pair, which the command buffer copies to and from.
        nbody::Buffer buffer_bodies;
        nbody::Buffer buffer_nodes;
        nbody::Buffer staging_bodies;
        nbody::Buffer staging_nodes;

        // Set by write(), consumed by the next command buffer: staging holds bodies the
        // device buffers have not been given yet.
        bool upload_pending = false;

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

        // command buffer recording and submission
        void record_dispatch(vk::raii::Pipeline& pipeline);
        void record_dispatch_barrier();
        void record_upload();
        void record_readback();
        void submit_and_wait(bool frame_end);
        void set_accelerate_constants(float theta, float gravity, Mode mode);
        void set_integrate_constants(float dt, float size, bool wrap);

        // Storage the shaders bind. Device-local and not host-visible, so it comes from the
        // full VRAM heap rather than the small mappable window, and shader reads never
        // cross the bus.
        template <typename Type>
        nbody::Buffer make_device_buffer(uint32_t num)
        {
            return {
                physical_device, device, sizeof(Type) * num,
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal };
        }

        // The host's side of a transfer. Asking for HOST_CACHED is what keeps this in
        // system memory and, more to the point, makes it readable at a sensible speed:
        // reading back through an uncached mapping is punishingly slow.
        template <typename Type>
        nbody::Buffer make_staging_buffer(uint32_t num)
        {
            return {
                physical_device, device, sizeof(Type) * num,
                vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent |
                vk::MemoryPropertyFlagBits::eHostCached };
        }

    public:

        static uint32_t find_memory_type(vk::PhysicalDeviceMemoryProperties const& memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask);

        static vk::raii::DeviceMemory alloc_device_memory(
                vk::raii::Device const& device,
                vk::PhysicalDeviceMemoryProperties const& memoryProperties,
                vk::MemoryRequirements const& memoryRequirements,
                vk::MemoryPropertyFlags memoryPropertyFlags);
    };
}