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
        // Serial, and shared verbatim by the CPU and GPU barnes-hut solvers, which makes it
        // the part of a GPU frame that the device cannot help with. Worth its own zone for
        // exactly that reason.
        NBODY_PROFILE_ZONE();
        tree.clear({ .size = size });
        tree.reserve(bodies.size() << 2);
        for (const Body& body : bodies)
            tree.insert(body.pos, body.mass);

        // Plotted rather than zoned: the interesting thing about the node count is how it
        // tracks cost over time, not how long counting it took.
        NBODY_PROFILE_PLOT("bh nodes", static_cast<int64_t>(tree.nodes().size()));
    }
}
