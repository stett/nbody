#pragma once
#include <vector>
#include "nbody/body.h"
#include "nbody/bhtree.h"
#include "nbody/profile.h"

namespace nbody::detail
{
    // Rebuild the barnes-hut acceleration tree from scratch. Shared by the CPU and GPU
    // barnes-hut solvers so the two cannot drift apart in how the tree is constructed.
    inline void build_tree(bh::Tree& tree, const std::vector<Body>& bodies, const float size)
    {
        // Serial, and shared by both barnes-hut solvers: the part of a GPU frame the
        // device cannot help with.
        NBODY_PROFILE_ZONE();
        tree.clear({ .size = size });
        tree.reserve(bodies.size() << 2);
        {
            NBODY_PROFILE_ZONE_NAMED("Insert bodies");
            for (const Body& body : bodies)
                tree.insert(body.pos, body.mass);
        }

        NBODY_PROFILE_PLOT("bh nodes", static_cast<int64_t>(tree.nodes().size()));
    }
}
