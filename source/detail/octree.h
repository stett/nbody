#pragma once
#include <cstdint>
#include <span>
#include <optional>
#include "detail/morton.h"
#include "detail/radix.h"
#include "nbody/vector.h"
#include "nbody/bounds.h"


namespace nbody::detail
{
    using std::span;

    struct OctreeNode
    {
        int32_t get_level() const { return level; }
        int32_t get_prefix() const { return prefix; }
        int32_t get_parent() const { return parent; }

        int32_t level = 0;
        int32_t prefix = 0;
        int32_t parent = 0;
    };


    /*
    struct OctreeNode
    {
        // bounds for this node
        Bounds bounds;

        // index to the parent node
        int32_t parent;

        // index to next node at the same level, or the parent's "next"
        int32_t next;

        // index to the first payload element if there are leaves in this node
        int32_t element;
    };
    */

    namespace scalar
    {
        // do a prefix sum to compute the offset into the octree node array
        // corresponding to each radix node.
        //
        // return the total number of octree nodes.
        int32_t octree_node_offsets(const span<const int32_t> node_cpl_deltas, const span<int32_t> node_offsets);

        // Build an octree from a set of points, populating a span of nodes in a flat array.
        //
        // The entire span of points being read from must be provided, but the span of nodes being
        // written to can be a subset of the total, so that the octree can be built in parallel. In
        // this case, the node_offset parameter must be set to the index of the first node in theh
        // span being written to.
        void octree(const span<const Vector> points, const span<OctreeNode> nodes);
    }
}
