#pragma once
#include <vector>
#include "nbody/body.h"
#include "nbody/bhtree.h"
#include "nbody/profile.h"

namespace nbody::detail
{
    // Rebuild the barnes-hut acceleration tree from scratch. Shared by the CPU and GPU
    // barnes-hut solvers so the two cannot drift apart in how the tree is constructed.
    //
    // Templated on the element so the GPU solver can build straight out of its staging
    // positions: Body and BodyPosMass both expose .pos and .mass.
    template <typename Item>
    void build_tree(bh::Tree& tree, const Item* items, const size_t count, const float size)
    {
        // Serial, and shared by both barnes-hut solvers: the part of a GPU frame the
        // device cannot help with.
        NBODY_PROFILE_ZONE();
        tree.clear({ .size = size });
        tree.reserve(count << 2);
        {
            NBODY_PROFILE_ZONE_NAMED("Insert bodies");
            for (size_t i = 0; i < count; ++i)
                tree.insert(items[i].pos, items[i].mass);
        }

        NBODY_PROFILE_PLOT("bh nodes", static_cast<int64_t>(tree.nodes().size()));
    }

    inline void build_tree(bh::Tree& tree, const std::vector<Body>& bodies, const float size)
    {
        build_tree(tree, bodies.data(), bodies.size(), size);
    }
}
