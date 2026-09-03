#pragma once
#include <algorithm>
#include <memory>
#include "context.h"
#include "solver.h"
#include "gpu.h"
#include "detail/parallel.h"
#include "nbody/profile.h"

namespace nbody
{
    // Runs the simulation on a vulkan compute device, building its own Barnes-Hut octree
    // on the device every frame from morton codes via a radix tree, rather than uploading a
    // CPU-built bh::Node tree the way GpuSolverSplit's NLogN mode does.
    //
    // Body memory layout is otherwise identical to GpuSolverSplit -- three arrays,
    // { pos, mass }, { vel, rad }, { acc } -- and shares that layout's buffers on the
    // device. There is no N2/NLogN choice here: this variant is unconditionally
    // tree-based, so unlike GpuSolver/GpuSolverSplit it takes no Mode.
    class GpuSolverBarnesHutMortonSoA final : public Solver
    {
    public:

        GpuSolverBarnesHutMortonSoA(std::shared_ptr<Context> context, StateRef state)
            : Solver(std::move(context), std::move(state))
            , _gpu(_context->require_gpu())
        {}

        [[nodiscard]] StateRef state() const override
        {
            materialize();
            return _state;
        }

        void adopt(StateRef state) override
        {
            _state = std::move(state);

            // The incoming State is authoritative and the device holds nothing of it
            // yet, so it must be uploaded before anything reads device-side.
            _device_dirty = false;
            _host_dirty = true;
        }

        void ingest() const override
        {
            // A caller changed the bodies, so whatever is on the device is now stale.
            // Record it rather than uploading here: ingest() is const and may run on a
            // read path, and the upload is only actually needed before the next
            // dispatch.
            _host_dirty = true;
        }

        // No update() override: the Solver base's default (accelerate() then integrate(dt))
        // is exactly right here, since unlike GpuSolverSplit there is no single combined
        // device submission for this variant to fold the two into.
        void accelerate() override
        {
            NBODY_PROFILE_ZONE();
            // An empty body array cannot be bound: Buffer::allocate() leaves a null
            // vk::Buffer at size 0, and prepare_split()/prepare_morton_soa() would bind
            // it with range 0 -- VUID-VkDescriptorBufferInfo-range-00341, and undefined
            // behaviour without the nullDescriptor feature.
            if (_state->bodies.empty())
                return;

            if (_host_dirty)
                upload_bodies();

            // Not _gpu->accelerate(): that dispatches the existing split-layout shader
            // against the old bh::Node buffer, which this variant does not use.
            _gpu->accelerate_morton_soa(_state->theta, _state->gravity, Readback::None);
            _device_dirty = true;
        }

        void integrate(const float dt) override
        {
            NBODY_PROFILE_ZONE();
            // See accelerate(): an empty body array cannot be bound as a descriptor.
            if (_state->bodies.empty())
                return;

            // Upload first if a caller has mutated the bodies since the last dispatch.
            if (_host_dirty)
                upload_bodies();

            // Same integrate_split.comp pipeline GpuSolverSplit uses -- it only touches
            // pos/vel, which do not change shape for this variant.
            _gpu->integrate(dt, _state->size, _state->wrap, Readback::None);
            _device_dirty = true;
        }

        // The octree lives entirely in device-local scratch buffers (buffer_octree_nodes/
        // bounds/masses in GpuDevice) with no host-visible staging copy, unlike
        // GpuSolverSplit's bh::Tree. There is nothing on the CPU side to read here without
        // designing an explicit readback path (a staging buffer + copy + fence wait,
        // mirroring buffer_pos_mass/staging_pos_mass) purely to serve the demo's debug
        // wireframe. Left as a TODO rather than invented here: debug visualization is not
        // required for this variant to be selectable and inert, and a readback added for
        // its own sake would be untested, unused code.
        [[nodiscard]] size_t debug_node_count() const override { return 0; }

    private:

        // De-interleave Body straight into the mapped staging allocations. Identical to
        // GpuSolverSplit::upload_bodies() -- body layout is shared.
        void upload_bodies()
        {
            NBODY_PROFILE_ZONE();

            const size_t num_bodies = _state->bodies.size();
            _gpu->reserve_bodies(num_bodies);

            const GpuDevice::BodyMapping mapping = _gpu->map_bodies(0, num_bodies);

            detail::parallel_blocks(*_context->pool, num_bodies, [this, mapping](const size_t begin, const size_t end)
            {
                for (size_t i = begin; i < end; ++i)
                {
                    const Body& body = _state->bodies[i];
                    mapping.pos_mass[i] = { body.pos, body.mass };
                    mapping.vel_radius[i] = { body.vel, body.radius };
                    mapping.acc[i] = { body.acc, 0 };
                }
            });

            _host_dirty = false;
        }

        // Reassemble Body from the parallel arrays. Identical to
        // GpuSolverSplit::materialize() -- body layout is shared.
        void materialize() const
        {
            NBODY_PROFILE_ZONE();

            if (!_device_dirty)
                return;

            _gpu->download(Readback::All);

            const size_t num_bodies = std::min(_state->bodies.size(), _gpu->staged_body_count());
            const BodyPosMass* const pos_mass = _gpu->staged_pos_mass();
            const BodyVelRadius* const vel_radius = _gpu->staged_vel_radius();
            const BodyAcc* const acc = _gpu->staged_acc();

            detail::parallel_blocks(*_context->pool, num_bodies, [this, pos_mass, vel_radius, acc](const size_t begin, const size_t end)
            {
                for (size_t i = begin; i < end; ++i)
                {
                    Body& body = _state->bodies[i];
                    body.pos = pos_mass[i].pos;
                    body.mass = pos_mass[i].mass;
                    body.vel = vel_radius[i].vel;
                    body.radius = vel_radius[i].radius;
                    body.acc = acc[i].acc;
                }
            });

            _device_dirty = false;
        }

        std::shared_ptr<GpuDevice> _gpu;

        // The device holds results the canonical State has not seen yet.
        // mutable: materialize() is called from the const state().
        mutable bool _device_dirty = false;

        // The canonical State holds bodies the device has not seen yet.
        // mutable: ingest() is const so that reads can drive it.
        mutable bool _host_dirty = true;
    };
}
