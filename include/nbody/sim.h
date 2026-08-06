#pragma once
#include <vector>
#include <functional>
#include <memory>
#include <span>
#include "BS_thread_pool.hpp"
#include "body.h"
#include "bhtree.h"
#include "constants.h"

namespace nbody
{
    class GPU;  // forward declaration — keeps Vulkan headers out of sim.h

    class Sim
    {
    public:

        Sim();
        ~Sim();

        // Try to bring up the compute backend. Falls back to the CPU path on any
        // failure, so GPU support is a runtime capability rather than a build option.
        void init_gpu();
        [[nodiscard]] bool using_gpu() const { return _use_gpu; }

        // --- state -------------------------------------------------------------

        // Read-only access. This is the default spelling on purpose: a later variant
        // may hold its bodies somewhere other than this vector, and reads are cheap
        // while handing out mutable access is not.
        [[nodiscard]] const std::vector<Body>& bodies() const { return _bodies; }

        // Mutable access. Callers that only read must NOT use this — it is the hook a
        // later variant uses to notice the state changed and re-ingest it, so taking it
        // for a read costs a needless conversion. Spelled distinctly rather than as a
        // non-const overload so that `sim.bodies()` on a non-const Sim cannot silently
        // select the expensive one.
        [[nodiscard]] std::vector<Body>& mutable_bodies() { return _bodies; }

        // world extent; space wraps over [-size/2, +size/2] when wrap() is set
        [[nodiscard]] float size() const { return _size; }
        void set_size(const float v) { _size = v; }

        // barnes-hut opening angle
        [[nodiscard]] float theta() const { return _theta; }
        void set_theta(const float v) { _theta = v; }

        // gravitational constant
        [[nodiscard]] float gravity() const { return _gravity; }
        void set_gravity(const float v) { _gravity = v; }

        // whether space wraps into a 3-torus. Honored identically by every backend.
        [[nodiscard]] bool wrap() const { return _wrap; }
        void set_wrap(const bool v) { _wrap = v; }

        // --- stepping ----------------------------------------------------------

        // full update of simulation
        void update(float dt);

        // update of acceleration
        void accelerate();

        // integration of acceleration and velocity
        void integrate(float dt);

        // apply a function to every body in parallel
        void visit(const std::function<void(Body& body)>& func);

        // --- visualization ------------------------------------------------------
        // The barnes-hut tree, or nullptr for a backend that builds none. Callers must
        // tolerate null: not every simulation variant is tree-based.
        [[nodiscard]] const bh::Tree* tree() const { return &_acc_tree; }

        // Convenience for the common "how many nodes / draw them" case. Empty when
        // there is no tree.
        //
        // WARNING: the returned span is invalidated by the next accelerate()/update(),
        // which rebuilds and reserves the node array. Use it immediately; never store it
        // across a step.
        [[nodiscard]] std::span<const bh::Node> nodes() const
        {
            const bh::Tree* t = tree();
            return t ? std::span<const bh::Node>(t->nodes()) : std::span<const bh::Node>{};
        }

    private:

        float _size = 10000.f;
        float _theta = .5f;
        float _gravity = G;
        bool _wrap = true;

        std::vector<Body> _bodies;
        bh::Tree _acc_tree;
        BS::thread_pool _pool;

        bool _use_gpu = false;
        std::unique_ptr<GPU> _gpu;
    };
}
