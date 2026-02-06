#pragma once
#include <vector>
#include <functional>
#include "BS_thread_pool.hpp"
#include "body.h"
#include "bhtree.h"
#include "gpu.h"

namespace nbody
{
    struct Sim
    {
        float size = 10000.f;
        std::vector<Body> bodies;
        bh::Tree acc_tree;
        BS::thread_pool pool;
#if NBODY_GPU
        bool use_gpu = true;
        GPU gpu;
#endif

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
