#pragma once
#include <vector>
#include "nbody/body.h"
#include "nbody/bhtree.h"

namespace nbody::detail
{
    // Rebuild the barnes-hut acceleration tree from scratch. Shared by the CPU and GPU
    // barnes-hut solvers so the two cannot drift apart in how the tree is constructed.
    inline void build_tree(bh::Tree& tree, const std::vector<Body>& bodies, const float size)
    {
        tree.clear({ .size = size });
        tree.reserve(bodies.size() << 2);
        for (const Body& body : bodies)
            tree.insert(body.pos, body.mass);
    }
}
