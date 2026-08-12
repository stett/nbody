#pragma once
#include <cstddef>
#include <cstdint>
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

    // The bodies as parallel arrays, grouped by how often each field crosses the bus. Must
    // match the like-named structs in shaders/include/body_split.glsl field for field. A
    // vec3 costs 16 bytes in std430 regardless, so the paired field rides along for free.
    struct BodyPosMass
    {
        Vector pos;
        float mass = 0;
    };

    struct BodyVelRadius
    {
        Vector vel;
        float radius = 0;
    };

    struct BodyAcc
    {
        Vector acc;
        float __pad = 0;
    };

    // Which body arrays a transfer brings back. A bitmask because the point of the split is
    // that the three move independently.
    enum class Readback : uint32_t
    {
        None = 0,
        Positions = 1 << 0,
        Velocities = 1 << 1,
        Accelerations = 1 << 2,
        All = Positions | Velocities | Accelerations,
    };

    constexpr Readback operator|(const Readback a, const Readback b)
    { return static_cast<Readback>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); }

    constexpr Readback operator&(const Readback a, const Readback b)
    { return static_cast<Readback>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b)); }

    constexpr Readback operator~(const Readback a)
    { return static_cast<Readback>(~static_cast<uint32_t>(a) & static_cast<uint32_t>(Readback::All)); }

    constexpr bool any(const Readback a) { return a != Readback::None; }

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

        // Half-open byte range written since the last copy was recorded, coalesced into one
        // span. Uploading this rather than the whole array is what makes a one-body edit
        // cost one body's worth of traffic.
        vk::DeviceSize dirty_begin = 0;
        vk::DeviceSize dirty_end = 0;

        // Persistent mapping of `memory`, established by allocate() and valid whenever
        // `size` is non-zero. Null for an empty allocation, and for device-local memory
        // that was never host-visible in the first place.
        void* mapped = nullptr;

        Buffer(
            vk::raii::PhysicalDevice& physical_device,
            vk::raii::Device& device,
            vk::DeviceSize size,
            vk::BufferUsageFlags usage,
            vk::MemoryPropertyFlags properties);

        // allocate gpu memory
        void allocate(size_t size);

        // Grow to hold `bytes` and record that many as live. Returns whether the storage
        // moved, which discards the contents and so obliges the caller to rewrite all of it.
        bool reserve(size_t bytes);

        // Copy in at `offset` and mark the range for upload. Requires host-visible memory
        // sized by a prior reserve().
        void write(const void* data, size_t offset, size_t bytes);

        // Address into the mapping, for a caller building its data in place. Does no dirty
        // bookkeeping, so worker threads may call it on disjoint ranges; mark with dirty().
        std::byte* at(size_t offset);
        const std::byte* at(size_t offset) const;

        // Fold [offset, offset + bytes) into the pending upload range.
        void dirty(size_t offset, size_t bytes);

        void clear_dirty() { dirty_begin = dirty_end = 0; }
        [[nodiscard]] bool is_dirty() const { return dirty_end > dirty_begin; }
    };

    // A vulkan compute device plus the pipelines the simulation runs on it. Owned by
    // Context and shared between the GPU variants, so switching between them does not
    // pay for shader compilation again.
    //
    // Carries two body layouts so they can be measured against each other: `interleaved`,
    // one array of Body moved whole every step, and `split`, three arrays grouped by
    // transfer lifetime. They share the device, queue, command buffer and node buffer.
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

        // ---- interleaved layout: the baseline. Everything moves every step. -------------

        void write_interleaved(const std::vector<Body>& bodies, const std::vector<bh::Node>& nodes);
        void read_interleaved(std::vector<Body>& bodies);
        void integrate_interleaved(float dt, float size, bool wrap);
        void accelerate_interleaved(float theta, float gravity, Mode mode);
        void step_interleaved(float dt, float theta, float gravity, Mode mode, float size, bool wrap);

        // ---- split layout ----------------------------------------------------------------

        // Writable pointers into staging, for a caller de-interleaving in place. Valid until
        // the next reserve_bodies(); worker threads may fill disjoint sub-ranges.
        struct BodyMapping
        {
            BodyPosMass* pos_mass;
            BodyVelRadius* vel_radius;
            BodyAcc* acc;
        };

        // Size the body staging arrays. A grow discards the contents, so this must be
        // followed by a write of the whole range.
        void reserve_bodies(size_t num_bodies);

        // Map [offset, offset + count) of the body arrays and mark it for upload.
        BodyMapping map_bodies(size_t offset, size_t count);

        // Stage the barnes-hut nodes. Sizes itself: the tree is rewritten whole every frame.
        void write_nodes(const std::vector<bh::Node>& nodes);

        // Which staging arrays match the device. Reading one staged() does not name gives
        // the previous step's data.
        [[nodiscard]] Readback staged() const { return staging_valid; }
        [[nodiscard]] size_t staged_body_count() const;
        [[nodiscard]] const BodyPosMass* staged_pos_mass() const;
        [[nodiscard]] const BodyVelRadius* staged_vel_radius() const;
        [[nodiscard]] const BodyAcc* staged_acc() const;

        // Bring back anything in `want` that staging lacks, in a submission of its own. A
        // no-op when the last dispatch already read it back.
        void download(Readback want);

        void integrate(float dt, float size, bool wrap, Readback readback);
        void accelerate(float theta, float gravity, Mode mode, Readback readback);

        // Accelerate and integrate in a single submission, ordered by a pipeline barrier.
        // Equivalent to accelerate() followed by integrate(), but without the host round
        // trip between them; prefer it whenever both halves are wanted.
        void step(float dt, float theta, float gravity, Mode mode, float size, bool wrap, Readback readback);

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

        // Shared by both layouts: the tree is the same structure either way.
        nbody::Buffer buffer_nodes;
        nbody::Buffer staging_nodes;

        // ---- interleaved layout ----------------------------------------------------------
        vk::raii::DescriptorSetLayout descriptor_set_layout_interleaved;
        vk::raii::DescriptorSet descriptor_set_interleaved;
        vk::raii::PipelineLayout pipeline_layout_interleaved;
        vk::raii::ShaderModule shader_integrate_interleaved;
        vk::raii::ShaderModule shader_accelerate_interleaved;
        vk::raii::Pipeline pipeline_integrate_interleaved;
        vk::raii::Pipeline pipeline_accelerate_interleaved;
        nbody::Buffer buffer_bodies;
        nbody::Buffer staging_bodies;

        // Staging holds bodies the device buffer has not been given yet.
        bool upload_pending_interleaved = false;

        // ---- split layout ----------------------------------------------------------------
        vk::raii::DescriptorSetLayout descriptor_set_layout_split;
        vk::raii::DescriptorSet descriptor_set_split;
        vk::raii::PipelineLayout pipeline_layout_split;
        vk::raii::ShaderModule shader_integrate_split;
        vk::raii::ShaderModule shader_accelerate_split;
        vk::raii::Pipeline pipeline_integrate_split;
        vk::raii::Pipeline pipeline_accelerate_split;

        // The buffers the shaders read are device-local and not mappable. The host reaches
        // them only through the staging pair, which the command buffer copies to and from.
        nbody::Buffer buffer_pos_mass;
        nbody::Buffer buffer_vel_radius;
        nbody::Buffer buffer_acc;
        nbody::Buffer staging_pos_mass;
        nbody::Buffer staging_vel_radius;
        nbody::Buffer staging_acc;

        // Which staging arrays agree with the device. Set by an upload or a readback,
        // cleared per-array by a dispatch that writes it.
        Readback staging_valid = Readback::None;

        // Whether a bound device buffer has moved since the descriptor set was written.
        // Separate from the dirty ranges: re-binding can be needed with no new data.
        bool descriptors_stale_split = true;

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
        vk::raii::DescriptorSetLayout make_descriptor_set_layout(uint32_t num_bindings);
        vk::raii::DescriptorSet make_descriptor_set(vk::raii::DescriptorSetLayout& layout);
        vk::raii::PipelineLayout make_pipeline_layout(vk::raii::DescriptorSetLayout& layout);
        vk::raii::ShaderModule make_shader(const unsigned char* spv, size_t size);

        template <size_t size>
        vk::raii::ShaderModule make_shader(const unsigned char (&spv)[size])
        { return std::move(make_shader(spv, size)); }

        vk::raii::Pipeline make_pipeline(vk::raii::ShaderModule& shader, vk::raii::PipelineLayout& layout);

        // Size the split device buffers to their staging counterparts and rebind if anything
        // moved. Before recording, so a dispatch that uploads nothing still binds storage.
        void prepare_split();

        // command buffer recording and submission
        void record_dispatch(vk::raii::Pipeline& pipeline, vk::raii::PipelineLayout& layout, vk::raii::DescriptorSet& set);
        void record_dispatch_barrier();
        void record_upload_interleaved();
        void record_readback_interleaved();
        void record_upload_split();
        void record_readback_split(Readback what);
        void submit_and_wait(bool frame_end, const vk::raii::Buffer& frame_buffer);
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