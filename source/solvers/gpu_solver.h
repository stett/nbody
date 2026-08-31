#pragma once
#include <memory>
#include "context.h"
#include "solver.h"
#include "gpu.h"
#include "detail/tree.h"
#include "nbody/profile.h"

namespace nbody
{
    // Runs the simulation on a vulkan compute device, in either barnes-hut or
    // brute-force mode, over one interleaved array of Body. The baseline GpuSolverSplit
    // (solvers/gpu_solver_split.h) is measured against; everything here moves every step.
    class GpuSolver final : public Solver
    {
    public:

        GpuSolver(std::shared_ptr<Context> context, StateRef state, const Mode mode)
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

            build_or_clear_tree();
            upload();
            _gpu->step_interleaved(dt, _state->theta, _state->gravity, _mode, _state->size, _state->wrap);
            _device_dirty = true;

            // Publish every step, as integrate() does.
            materialize();
        }

        void accelerate() override
        {
            NBODY_PROFILE_ZONE();
            // An empty body array cannot be bound: Buffer::allocate() leaves a null
            // vk::Buffer at size 0, and write_interleaved() would bind it with range 0 --
            // VUID-VkDescriptorBufferInfo-range-00341, and undefined behaviour without
            // the nullDescriptor feature.
            if (_state->bodies.empty())
                return;

            build_or_clear_tree();
            upload();
            _gpu->accelerate_interleaved(_state->theta, _state->gravity, _mode);
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
                upload();

            _gpu->integrate_interleaved(dt, _state->size, _state->wrap);
            _device_dirty = true;

            // Publish every step. The demo reads the bodies each frame anyway, so
            // staying lazy would buy nothing today; this is solver-local policy and can
            // change without touching the interface.
            materialize();
        }

        // N^2 mode's root-only tree is a binding placeholder, not a real acceleration
        // structure, so don't offer it to the renderer.
        // The N^2 mode's root-only tree is a binding placeholder, not a real acceleration
        // structure, so don't offer it to the renderer.
        [[nodiscard]] size_t debug_node_count() const override
        {
            return _mode == Mode::NLogN ? _tree.nodes().size() : 0;
        }

        size_t write_debug_nodes(const std::span<DebugNode> out) const override
        {
            return _mode == Mode::NLogN ? detail::write_debug_nodes(_tree, out) : 0;
        }

    private:

        // Bring _tree in line with the current bodies, ready to be bound for a dispatch.
        void build_or_clear_tree()
        {
            NBODY_PROFILE_ZONE();
            if (_mode == Mode::NLogN)
            {
                detail::build_tree(_tree, _state->bodies, _state->size);
            }
            else
            {
                // The N^2 shader never reads the node buffer, but write() binds it
                // regardless and a zero-sized allocation leaves a null vk::Buffer, so
                // keep a root-only tree to bind against.
                _tree.clear({ .size = _state->size });
            }
        }

        // Push the canonical State into this solver's representation. Today the device
        // layout already matches Body so this is a straight upload; a solver with a
        // different internal layout would transform here.
        void upload()
        {
            NBODY_PROFILE_ZONE();
            _gpu->write_interleaved(_state->bodies, _tree.nodes());
            _host_dirty = false;
        }

        // Convert the device's representation back into the canonical State. Today the
        // device layout already matches Body so this is a straight download; a solver
        // with a different internal layout would transform here.
        void materialize() const
        {
            NBODY_PROFILE_ZONE();
            if (!_device_dirty)
                return;
            _gpu->read_interleaved(_state->bodies);
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
    };
}
