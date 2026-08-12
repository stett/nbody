#pragma once
#include <algorithm>
#include <memory>
#include "context.h"
#include "solver.h"
#include "gpu.h"
#include "detail/parallel.h"
#include "detail/tree.h"
#include "nbody/profile.h"

namespace nbody
{
    // Runs the simulation on a vulkan compute device, in either barnes-hut or
    // brute-force mode. The difference from GpuSolver is that rather than interleaving
    // body data in an AoS, this solver splits it out into three buffers - an SoA.
    //
    // These contain { pos, mass }, { vel, rad }, { acc }.
    class GpuSolverSplit final : public Solver
    {
    public:

        GpuSolverSplit(std::shared_ptr<Context> context, StateRef state, const Mode mode)
            : Solver(std::move(context), std::move(state))
            , _gpu(_context->require_gpu())
            , _mode(mode)
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
            _tree.clear({ .size = _state->size });
        }

        void ingest() const override
        {
            // A caller changed the bodies, so whatever is on the device is now stale.
            // Record it rather than uploading here: ingest() is const and may run on a
            // read path, and the upload is only actually needed before the next
            // dispatch.
            _host_dirty = true;
        }

        // Both halves in one submission, ordered by a barrier, rather than the base
        // implementation's two submits with a host wait between them.
        void update(const float dt) override
        {
            NBODY_PROFILE_ZONE();
            // See accelerate(): an empty body array cannot be bound as a descriptor.
            if (_state->bodies.empty())
                return;

            // Before the tree is built off positions the device has not been told about.
            if (_host_dirty)
                upload_bodies();

            build_or_clear_tree();
            upload_nodes();

            _gpu->step(dt, _state->theta, _state->gravity, _mode, _state->size, _state->wrap, wanted_readback());
            _device_dirty = true;
        }

        void accelerate() override
        {
            NBODY_PROFILE_ZONE();
            // An empty body array cannot be bound: Buffer::allocate() leaves a null
            // vk::Buffer at size 0, and prepare_split() would bind it with range 0 --
            // VUID-VkDescriptorBufferInfo-range-00341, and undefined behaviour without
            // the nullDescriptor feature.
            if (_state->bodies.empty())
                return;

            if (_host_dirty)
                upload_bodies();

            build_or_clear_tree();
            upload_nodes();

            // Nothing is fetched: materialize() collects the accelerations if asked.
            _gpu->accelerate(_state->theta, _state->gravity, _mode, Readback::None);
            _device_dirty = true;
        }

        void integrate(const float dt) override
        {
            NBODY_PROFILE_ZONE();
            // See accelerate(): an empty body array cannot be bound as a descriptor.
            if (_state->bodies.empty())
                return;

            // Upload first if a caller has mutated the bodies since the last dispatch.
            // Without this, integrate() without a preceding accelerate() would step the
            // device's pre-mutation copy and then materialize() would download the
            // result straight over the caller's write, destroying it.
            if (_host_dirty)
                upload_bodies();

            _gpu->integrate(dt, _state->size, _state->wrap, Readback::None);
            _device_dirty = true;
        }

        // N^2 mode's root-only tree is a binding placeholder, not a real acceleration
        // structure, so don't offer it to the renderer.
        [[nodiscard]] const bh::Tree* tree() const override
        {
            return _mode == Mode::NLogN ? &_tree : nullptr;
        }

    private:

        // Positions in barnes-hut mode, because the next frame's tree is built from them.
        // Everything, if the last step was followed by a materialize(): a caller that reads
        // every frame will read again, and folding it in here saves a whole round trip.
        [[nodiscard]] Readback wanted_readback() const
        {
            Readback want = _mode == Mode::NLogN ? Readback::Positions : Readback::None;
            if (_materialize_expected)
                want = want | Readback::All;
            _materialize_expected = false;
            return want;
        }

        // Bring _tree in line with the current bodies, ready to be bound for a dispatch.
        void build_or_clear_tree()
        {
            NBODY_PROFILE_ZONE();
            if (_mode != Mode::NLogN)
            {
                // The N^2 shader never reads the node buffer, but prepare() binds it
                // regardless and a zero-sized allocation leaves a null vk::Buffer, so
                // keep a root-only tree to bind against.
                _tree.clear({ .size = _state->size });
                return;
            }

            // Straight out of the staging positions: the same values as State::bodies, but
            // without re-interleaving a million bodies to reach two fields.
            _gpu->download(Readback::Positions);
            detail::build_tree(_tree, _gpu->staged_pos_mass(), _gpu->staged_body_count(), _state->size);
        }

        // De-interleave Body straight into the mapped staging allocations. The split has to
        // touch every field either way, so an intermediate copy would be pure overhead.
        void upload_bodies()
        {
            NBODY_PROFILE_ZONE();

            const size_t num_bodies = _state->bodies.size();
            _gpu->reserve_bodies(num_bodies);

            // Mapped once here, so the workers below only write their own disjoint slice.
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

        // Rebuilt from scratch every frame, so there is no sending less than all of it.
        void upload_nodes()
        {
            NBODY_PROFILE_ZONE();
            _gpu->write_nodes(_tree.nodes());
        }

        // Reassemble Body from the parallel arrays. The only thing that asks for velocities
        // and accelerations: a caller that steps without reading never moves them.
        void materialize() const
        {
            NBODY_PROFILE_ZONE();

            // Set even when there is nothing to do: what matters is that this caller reads.
            _materialize_expected = true;

            if (!_device_dirty)
                return;

            _gpu->download(Readback::All);

            // Take only what both sides hold: the staging allocation only ever grows, so
            // reading all of it overruns `bodies` whenever the count has shrunk.
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
        Mode _mode;
        bh::Tree _tree;

        // The device holds results the canonical State has not seen yet.
        // mutable: materialize() is called from the const state().
        mutable bool _device_dirty = false;

        // The canonical State holds bodies the device has not seen yet.
        // mutable: ingest() is const so that reads can drive it.
        mutable bool _host_dirty = true;

        // Whether a materialize() followed the last step, used to predict the next one.
        mutable bool _materialize_expected = false;
    };
}
