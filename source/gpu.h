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

    // The device's view of the bodies, as three parallel arrays rather than one array of
    // Body. Each must match the like-named struct in shaders/include/common.glsl field for
    // field.
    //
    // The grouping is by transfer lifetime, not by field. A vec3 has 16-byte base alignment
    // even in std430, so an array of bare positions costs the same 16 bytes per body that
    // {pos, mass} does -- the fourth word is already paid for, and the only question is
    // whether it carries something the same loop wants:
    //
    //   pos_mass     read every frame by the n^2 inner loop and by the host tree build,
    //                which both want exactly these two fields and now stream one array
    //                instead of two. Downloaded every frame; mass rides along as static
    //                freight, uploaded once.
    //   vel_radius   device-resident. Radius is read once per invocation rather than once
    //                per pair, so it does not belong in the array the n^2 loop streams,
    //                and neither field is worth downloading unless a caller asks.
    //   acc          scratch between the accelerate and integrate dispatches. Transferred
    //                only when the host has set it or explicitly wants to read it.
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

    // Which of the body arrays a transfer brings back from the device. A bitmask because
    // the whole point of the split is that the three are fetched independently: the
    // barnes-hut tree build needs Positions every frame, and nothing else needs anything
    // until a caller actually reads the bodies.
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

        // The half-open byte range of `mapped` the host has written since the last copy was
        // recorded, coalesced into one span. Empty when begin == end. record_upload() sends
        // this rather than the whole array, which is what makes a one-body edit cost one
        // body's worth of bus traffic.
        vk::DeviceSize dirty_begin = 0;
        vk::DeviceSize dirty_end = 0;

        // Persistent mapping of `memory`, established by allocate() and valid whenever
        // `size` is non-zero. Null for an empty allocation, and for device-local memory
        // that was never host-visible in the first place. Callers that de-interleave on the
        // way through address it through at() rather than paying for a second copy.
        void* mapped = nullptr;

        Buffer(
            vk::raii::PhysicalDevice& physical_device,
            vk::raii::Device& device,
            vk::DeviceSize size,
            vk::BufferUsageFlags usage,
            vk::MemoryPropertyFlags properties);

        // allocate gpu memory
        void allocate(size_t size);

        // Grow to hold `bytes` and record that many as live, without preserving the
        // contents. Returns whether the storage moved -- a caller that wanted to write only
        // part of the buffer has to write all of it after a move, because the rest of the
        // new allocation holds nothing.
        bool reserve(size_t bytes);

        // Copy into the buffer at `offset` and mark the range for upload. Requires
        // host-visible memory sized by a prior reserve().
        void write(const void* data, size_t offset, size_t bytes);

        // Address into the mapping, for a caller that would rather build its data in place
        // than build it elsewhere and copy. Does no dirty bookkeeping, so it is safe to call
        // from worker threads on disjoint ranges; mark the range with dirty() first.
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

        // Writable pointers into the staging body arrays, for a caller that would rather
        // de-interleave in place than build three vectors and copy them. Valid until the
        // next reserve_bodies(); disjoint sub-ranges may be filled from worker threads,
        // since map_bodies() has already done all the shared bookkeeping.
        struct BodyMapping
        {
            BodyPosMass* pos_mass;
            BodyVelRadius* vel_radius;
            BodyAcc* acc;
        };

        // Size the body staging arrays. Contents are not preserved across a grow, so this
        // must be followed by a write of the whole range.
        void reserve_bodies(size_t num_bodies);

        // Map [offset, offset + count) of the body arrays and mark it for upload.
        BodyMapping map_bodies(size_t offset, size_t count);

        // Stage the barnes-hut nodes. Sizes as needed, so unlike the bodies there is no
        // separate reserve step -- the tree is rebuilt whole every frame anyway.
        void write_nodes(const std::vector<bh::Node>& nodes);

        // Which body arrays currently held in staging match the device, and the pointers to
        // them. Reading an array that staged() does not name gives the previous step's data.
        [[nodiscard]] Readback staged() const { return staging_valid; }
        [[nodiscard]] size_t staged_body_count() const;
        [[nodiscard]] const BodyPosMass* staged_pos_mass() const;
        [[nodiscard]] const BodyVelRadius* staged_vel_radius() const;
        [[nodiscard]] const BodyAcc* staged_acc() const;

        // Bring back anything in `want` that staging does not already hold, in a submission
        // of its own. A no-op when the last dispatch already read it back, which is how the
        // per-frame path avoids the extra round trip.
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
        vk::raii::DescriptorSetLayout descriptor_set_layout;
        vk::raii::DescriptorSet descriptor_set;
        vk::raii::PipelineLayout pipeline_layout;
        vk::raii::ShaderModule shader_integrate;
        vk::raii::ShaderModule shader_accelerate;
        vk::raii::Pipeline pipeline_integrate;
        vk::raii::Pipeline pipeline_accelerate;
        // The buffers the shaders read are device-local and not mappable. The host reaches
        // them only through the staging pair, which the command buffer copies to and from.
        nbody::Buffer buffer_pos_mass;
        nbody::Buffer buffer_vel_radius;
        nbody::Buffer buffer_acc;
        nbody::Buffer buffer_nodes;
        nbody::Buffer staging_pos_mass;
        nbody::Buffer staging_vel_radius;
        nbody::Buffer staging_acc;
        nbody::Buffer staging_nodes;

        // Which staging body arrays currently agree with the device. Set by an upload (the
        // host just made the two match) or a readback, and cleared per-array by a dispatch
        // that writes it. download() consults this to skip work already done.
        Readback staging_valid = Readback::None;

        // Whether the bound device buffers have moved since the descriptor set was last
        // written. Separate from the dirty ranges: a buffer can need re-binding without any
        // new data, and needs re-binding before a dispatch that uploads nothing.
        bool descriptors_stale = true;

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

        // Bring the device buffers up to the size of their staging counterparts and rebind
        // the descriptor set if anything moved. Runs before recording, not during it, so
        // that a dispatch which uploads nothing still binds valid storage.
        void prepare();

        // command buffer recording and submission
        void record_dispatch(vk::raii::Pipeline& pipeline);
        void record_dispatch_barrier();
        void record_upload();
        void record_readback(Readback what);
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