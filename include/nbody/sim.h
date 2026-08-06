#pragma once
#include <vector>
#include <functional>
#include <memory>
#include "BS_thread_pool.hpp"
#include "body.h"
#include "bhtree.h"
#include "constants.h"

namespace nbody
{
    class GPU;  // forward declaration — keeps Vulkan headers out of sim.h

    struct Sim
    {
        // world extent; space wraps over [-size/2, +size/2] when `wrap` is set
        float size = 10000.f;

        // barnes-hut opening angle
        float theta = .5f;

        // gravitational constant
        float gravity = G;

        // whether space wraps into a 3-torus. Honored identically by the CPU and GPU
        // paths — the GPU shader used to skip wrapping entirely, which let the two
        // disagree about where bodies ended up.
        bool wrap = true;

        std::vector<Body> bodies;
        bh::Tree acc_tree;
        BS::thread_pool pool;
        bool use_gpu = false;
        std::unique_ptr<GPU> gpu;
        Sim();
        ~Sim();
        // Try to bring up the compute backend. Falls back to the CPU path on any
        // failure, so GPU support is a runtime capability rather than a build option.
        void init_gpu();

        // full update of simulation
        void update(float dt);

        // update of acceleration
        void accelerate();

        // integration of acceleration and velocity
        void integrate(float dt);

        // apply a function to every body in parallel
        void visit(const std::function<void(Body& body)>& func);
    };
}
