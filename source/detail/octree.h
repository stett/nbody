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

    /*
    template <typename BitsT>
    class OctreeNode
    {
    public:

        OctreeNode() : _bits(0) { }

        OctreeNode(BitsT prefix, size_t level) : _bits(prefix >> (level + 1)) { }

        BitsT prefix() const
        {
            _bits << (level() + 1);
        }
        size_t level() const
        {
            return std::countl_zero(_bits);
        }

    private:

        BitsT _bits;
    };
    */

    /*
    struct QuadtreeNode
    {
        bool is_leaf = false;
        union
        {
            int32_t children[4];
            int32_t payload;
        };
    };
    */

    struct OctreeNode
    {
        int32_t parent = 0;
        int32_t next = 0;
        int32_t child = 0;
    };

    /*
    template <typename BitsT>
    BitsT clear_leading_bits(size_t n) { }
    */

    template <typename MortonT>
    int32_t quadrant(const MortonT& parent, const MortonT& child)
    {
        // the "modulus" number of bits following the cpl of a parent
        // and a child will be the quadrant
        using BitsT = typename MortonT::Bits;
        static constexpr size_t modulus = MortonT::modulus;
        const int32_t prefix_len = detail::scalar::cpl(parent, child);
        const BitsT quadrant_bits = ((child << prefix_len) >> ((sizeof(BitsT) * 8) - modulus));
        return static_cast<int32_t>(quadrant_bits);
    }

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
        //void octree(const span<const Vector> points, const span<OctreeNode> nodes);
    }
}
