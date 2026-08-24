#pragma once
#include <cstdint>
#include <span>
#include "detail/morton.h"
#include "detail/radix.h"
#include "nbody/vector.h"
#include "nbody/bounds.h"


namespace nbody::detail
{
    using std::span;

    struct OctreeNode
    {
        // bounds for this node
        Bounds bounds;

        // index to next node at the same level
        int32_t next; 

        // index to first child
        int32_t children; 

        // index to the parent node
        int32_t parent;
    };

    namespace scalar
    {
        // Build an octree from a set of points, populating a span of nodes in a flat array.
        //
        // The entire span of points being read from must be provided, but the span of nodes being
        // written to can be a subset of the total, so that the octree can be built in parallel. In
        // this case, the node_offset parameter must be set to the index of the first node in theh
        // span being written to.
        void octree(const span<const Vector> points, const span<OctreeNode> nodes);
    }
}
