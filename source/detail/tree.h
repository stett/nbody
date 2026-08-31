#pragma once
#include <algorithm>
#include <span>
#include <vector>
#include "nbody/body.h"
#include "nbody/bhtree.h"
#include "nbody/debug.h"
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

    // Fill `out` with a drawable view of `tree`, returning how many nodes were written.
    //
    // Shared by the three solvers that hold a bh::Tree, for the same reason build_tree is:
    // so they cannot drift apart on it. Nothing is cached -- the caller owns the storage
    // and decides when to pay, which matters because the weight below is a full traversal
    // per node.
    inline size_t write_debug_nodes(const bh::Tree& tree, const std::span<DebugNode> out)
    {
        NBODY_PROFILE_ZONE_NAMED("debug nodes");

        const std::vector<bh::Node>& src = tree.nodes();
        const size_t count = std::min(out.size(), src.size());
        for (size_t i = 0; i < count; ++i)
        {
            const Vector& center = src[i].bounds.center;

            // The gravitational potential at the node's own centre, at apply()'s default
            // opening angle. Deliberately not State::theta: this is a picture, and pinning
            // it keeps the shading still while the slider moves the physics.
            float weight = 0;
            tree.apply(center, [&weight, &center](const bh::Node& other)
            {
                const Vector delta = other.com - center;
                weight += other.mass / delta.size_sq();
            });

            // bounds.size is already a full edge length, which is why DebugNode uses that
            // convention: every bh::Tree solver copies it rather than converting.
            out[i] = { .center = center, .size = src[i].bounds.size, .weight = weight };
        }
        return count;
    }
}
