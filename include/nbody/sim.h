#pragma once
#include <vector>
#include "body.h"
#include "bhtree.h"

namespace nbody
{
    struct Sim
    {
        float size = 1000.f;
        std::vector<Body> bodies;
        bh::Tree acc_tree;

        void update(float dt);
        void integrate(float dt);
        void accelerate();
    };
}
