#pragma once
#include <memory>
#include "context.h"
#include "solver.h"
#include "gpu.h"
#include "detail/tree.h"

namespace nbody
{
    // Runs the simulation on a vulkan compute device, in either barnes-hut or
    // brute-force mode.
    //
    // Unlike the CPU solvers this one has a representation of its own -- the device
    // buffers -- so it is the first real user of the standard-format conversion:
    // materialize() brings State::bodies back in line with the device, and ingest()
    // pushes a caller's mutation the other way.
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

        void accelerate() override
        {
            // Nothing to dispatch, and dispatching anyway is invalid rather than merely
            // wasteful: Buffer::allocate() early-returns at size 0 and leaves a null
            // vk::Buffer, which write() would then bind into a descriptor with range 0.
            // That is VK_NULL_HANDLE plus VUID-VkDescriptorBufferInfo-range-00341, and
            // undefined behaviour on any driver without the nullDescriptor feature.
            // Reachable from a caller that steps before spawning anything.
            if (_state->bodies.empty())
                return;

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

            upload();
            _gpu->accelerate(_state->theta, _state->gravity, _mode);
            _device_dirty = true;
        }

        void integrate(const float dt) override
        {
            // See accelerate(): an empty body array cannot be bound as a descriptor.
            if (_state->bodies.empty())
                return;

            // Upload first if a caller has mutated the bodies since the last dispatch.
            // Without this, integrate() without a preceding accelerate() would step the
            // device's pre-mutation copy and then materialize() would download the
            // result straight over the caller's write, destroying it.
            if (_host_dirty)
                upload();

            _gpu->integrate(dt, _state->size, _state->wrap);
            _device_dirty = true;

            // Publish every step. The demo reads the bodies each frame anyway, so
            // staying lazy would buy nothing today; this is solver-local policy and can
            // change without touching the interface.
            materialize();
        }

        // N^2 mode's root-only tree is a binding placeholder, not a real acceleration
        // structure, so don't offer it to the renderer.
        [[nodiscard]] const bh::Tree* tree() const override
        {
            return _mode == Mode::NLogN ? &_tree : nullptr;
        }

    private:

        // Push the canonical State into this solver's representation. Today the device
        // layout already matches Body so this is a straight upload; a solver with a
        // different internal layout would transform here.
        void upload()
        {
            _gpu->write(_state->bodies, _tree.nodes());
            _host_dirty = false;
        }

        // Convert the device's representation back into the canonical State. Today the
        // device layout already matches Body so this is a straight download; a solver
        // with a different internal layout would transform here.
        void materialize() const
        {
            if (!_device_dirty)
                return;
            _gpu->read(_state->bodies);
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
