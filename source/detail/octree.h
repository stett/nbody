#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <numeric>
#include <optional>
#include "detail/morton.h"
#include "detail/radix.h"
#include "nbody/vector.h"
#include "nbody/bounds.h"


namespace nbody::detail
{
    using std::span;
    using std::vector;
    using std::exclusive_scan;

    struct OctreeNode
    {
        int32_t parent = 0;
        int32_t next = 0;
        int32_t child = 0;

        // TODO: pack this value into the sign bit for child
        bool is_leaf = false;

        bool operator==(const OctreeNode& rhs) const = default;
    };

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

    namespace scalar
    {
        // do a prefix sum to compute the offset into the octree node array
        // corresponding to each radix node.
        //
        // return the total number of octree nodes.
        int32_t octree_node_offsets(const span<const int32_t> node_cpl_deltas, const span<int32_t> node_offsets);

        // Map each key to the octree node that holds it, by scattering from the radix nodes
        // that name it as a leaf child.
        //
        // Every key is the leaf child of exactly one radix node, so one pass over the radix
        // nodes fills the whole map with no collisions. It replaces walking the child indices
        // forward to hand out leaf slots in array order, which only worked while the radix
        // nodes were visited in order -- and it is the inverse map the escape pointer needs,
        // since that has a key in hand and wants the node.
        inline void octree_leaf_nodes(
            const span<const RadixNode> radix_nodes,
            const span<const NodeCount> node_counts,
            const span<const int32_t> node_offsets,
            const span<int32_t> leaf_nodes)
        {
            for (int32_t i_radix = 0; i_radix < static_cast<int32_t>(radix_nodes.size()); ++i_radix)
            {
                // a radix node's leafs sit after its internals, child0 before child1
                const int32_t i_leaf_0 = 1 + node_offsets[i_radix] + node_counts[i_radix].internals;
                int32_t slot = 0;

                // a non-negative child index is a key rather than a node
                if (radix_nodes[i_radix].child0_index >= 0)
                    leaf_nodes[radix_nodes[i_radix].child0_index] = i_leaf_0 + slot++;
                if (radix_nodes[i_radix].child1_index >= 0)
                    leaf_nodes[radix_nodes[i_radix].child1_index] = i_leaf_0 + slot++;
            }
        }

        template <typename MortonT>
        void build_octree(
            span<const MortonT> keys,
            span<const RadixNode> radix_nodes,
            span<const int32_t> radix_parents,
            span<const NodeCount> node_counts,
            span<const int32_t> node_range_ends,
            span<const int32_t> node_totals,
            span<const int32_t> node_offsets,
            span<const int32_t> leaf_nodes,
            span<OctreeNode> octree_nodes)
        {

            // count the number of octree nodes, allocate.
            // make sure it matches the output we expect
            const int32_t num_nodes = 1 + node_offsets.back() + node_totals.back();
            assert(octree_nodes.size() == num_nodes);

            // The octree node a radix node's chain hangs from.
            //
            // Not `i_node_0 - 1`: within a chain the previous slot is the parent, and the loop
            // below uses that, but a chain *head* has no such luck. `i_node_0 - 1` is the last
            // slot of the previous radix node's block, which is neither this radix node's parent
            // in general nor, when it happens to be, an internal node -- a block's leafs come
            // after its internals. No layout fixes that either: a node has up to 2^modulus
            // children and only one of them can sit at parent + 1.
            const auto find_octree_parent = [&](const int32_t i_radix) -> int32_t
            {
                // radix node 0 covers every key, so its chain begins at level 1 and hangs
                // straight from the root. It has no radix parent to consult.
                if (i_radix == 0)
                    return 0;

                // Walk up past ancestors that resolved no octree level of their own: those
                // produced no internal node, so this chain hangs from whatever is above them.
                //
                // Bounded by modulus rather than by depth. A zero-internal radix node means the
                // split fell *inside* an octree level, and only `modulus` binary splits fit in
                // one level, so the walk is 2 steps at most for an octree.
                int32_t i_radix_parent = radix_parents[i_radix];
                while (i_radix_parent > 0 && node_counts[i_radix_parent].internals == 0)
                    i_radix_parent = radix_parents[i_radix_parent];

                // The parent is the *last internal* of that ancestor's chain. Its block starts
                // at 1 + node_offsets[p] and its internals precede its leafs, so the last
                // internal is 1 + node_offsets[p] + internals - 1.
                //
                // Keyed on the ancestor's own count, not on `i_radix_parent > 0`: radix node 0
                // is not special, and when it resolves a level its chain tail is a real node.
                // Landing on a node that resolved nothing means there was no internal ancestor
                // at all, and the parent is the root.
                const NodeCount parent_count = node_counts[i_radix_parent];
                return parent_count.internals > 0
                    ? node_offsets[i_radix_parent] + parent_count.internals
                    : 0;
            };

            // The first octree node of radix node i_radix's key range.
            //
            // The downward mirror of the walk above: a radix node that resolved no octree level
            // of its own contributes nothing at the level its parent expects, so descend to its
            // first child. Bounded by modulus for the same reason.
            const auto find_first_octree_node = [&](int32_t i_radix) -> int32_t
            {
                while (node_counts[i_radix].internals == 0 && radix_nodes[i_radix].child0_index < 0)
                    i_radix = -radix_nodes[i_radix].child0_index;

                // either it resolved a level, and its chain begins the range, or it resolved
                // none and its own first leaf does
                return (node_counts[i_radix].internals > 0)
                    ? 1 + node_offsets[i_radix]
                    : leaf_nodes[radix_nodes[i_radix].child0_index];
            };

            // The octree index of one of a radix node's children, leaf or internal.
            const auto find_octree_child = [&](const int32_t i_child) -> int32_t
            {
                return (i_child >= 0) ? leaf_nodes[i_child] : find_first_octree_node(-i_child);
            };

            // The escape pointer for any node whose key range ends at key i_key_end: the first
            // octree node of the range starting one key later.
            //
            // Taken on the range rather than on the node, because every node of a chain covers
            // the same range and so shares one escape -- which is what `i_node_end` was reaching
            // for. It missed because a block ends where its *subtree* ends only for a radix node
            // with no internal children; otherwise the subtree continues past the block.
            const auto find_octree_next = [&](const int32_t i_key_end) -> int32_t
            {
                // nothing follows the last key, and 0 is the root, so it doubles as the
                // "traversal finished" sentinel
                if (i_key_end >= static_cast<int32_t>(keys.size()) - 1)
                    return 0;

                // A radix node's index is one end of its range, so node_range_ends[m] > m says
                // m's range *begins* at m: the sibling spans two or more keys and is that radix
                // node. Otherwise the sibling is the lone key m.
                const int32_t m = i_key_end + 1;
                return (m < static_cast<int32_t>(radix_nodes.size()) && node_range_ends[m] > m)
                    ? find_first_octree_node(m)
                    : leaf_nodes[m];
            };

            // the root is the level 0 node of radix node 0's chain: it covers every key, so
            // nothing follows it, and its first child is whatever radix node 0 begins with
            octree_nodes[0] = {
                .parent = 0,
                .next = 0,
                .child = (node_counts[0].internals > 0)
                    ? 1 + node_offsets[0]
                    : find_octree_child(radix_nodes[0].child0_index),
                .is_leaf = false,
            };

            // run through all radix nodes, populating octree nodes from them
            for (int32_t i_radix = 0; i_radix < static_cast<int32_t>(radix_nodes.size()); ++i_radix)
            {
                const NodeCount node_count = node_counts[i_radix];

                // A radix node that resolved no level and has no leaf children produces nothing,
                // and has no slot of its own to write to: its block is empty, so 1 + offset is
                // the *next* node's block -- or one past the array when it is the last radix
                // node, which is a write past the end.
                if (node_count.internals + node_count.leafs == 0)
                    continue;

                // get the first octree node index
                const int32_t i_node_0 = 1 + node_offsets[i_radix];

                // every node of this chain covers the same key range, so one escape serves them all
                const int32_t i_next = find_octree_next(node_range_ends[i_radix]);

                // populate corresponding intermediate nodes
                for (int32_t i_internal = 0; i_internal < node_count.internals; ++i_internal)
                {
                    const int32_t i_node = i_node_0 + i_internal;

                    // the chain is contiguous, so every node past the first is parented by the
                    // previous one. only the head has to be looked up.
                    octree_nodes[i_node].parent = (i_internal == 0)
                        ? find_octree_parent(i_radix)
                        : i_node - 1;

                    octree_nodes[i_node].next = i_next;

                    // a chain node above the tail has exactly one child, the next link in the
                    // chain. the tail's children are the radix node's own, and the first of
                    // those is child0 -- which may live in another block entirely.
                    octree_nodes[i_node].child = (i_internal + 1 < node_count.internals)
                        ? i_node + 1
                        : find_octree_child(radix_nodes[i_radix].child0_index);

                    // TODO: pack this value into the child node's sign bit
                    octree_nodes[i_node].is_leaf = false;
                }

                // the leafs hang from the deepest node of the chain, or from whatever is above
                // the block when this radix node resolved no level of its own
                const int32_t i_leaf_parent = (node_count.internals > 0)
                    ? i_node_0 + node_count.internals - 1
                    : find_octree_parent(i_radix);

                // populate corresponding leaf nodes after the internals
                for (int32_t i_child = 0; i_child < 2; ++i_child)
                {
                    // a negative child index is an internal node, which gets its own block
                    const int32_t i_key = i_child
                        ? radix_nodes[i_radix].child1_index
                        : radix_nodes[i_radix].child0_index;
                    if (i_key < 0)
                        continue;

                    assert(i_key < static_cast<int32_t>(keys.size()));
                    const int32_t i_node = leaf_nodes[i_key];
                    assert(i_node >= i_node_0 + node_count.internals);

                    octree_nodes[i_node].parent = i_leaf_parent;

                    // a leaf's range is the single key, so its escape is the range starting at
                    // the next one -- the same rule the chain uses
                    octree_nodes[i_node].next = find_octree_next(i_key);

                    // for leaf nodes, the child index indicates an index into the original keys array
                    octree_nodes[i_node].child = i_key;

                    // TODO: pack this value into the child node's sign bit
                    octree_nodes[i_node].is_leaf = true;
                }
            }
        }

        // Build an octree from a set of points, populating a span of nodes in a flat array.
        //
        // The entire span of points being read from must be provided, but the span of nodes being
        // written to can be a subset of the total, so that the octree can be built in parallel. In
        // this case, the node_offset parameter must be set to the index of the first node in theh
        // span being written to.
        template <typename MortonT>
        vector<OctreeNode> build_octree(span<const MortonT> keys)
        {
            // build the radix tree
            vector<RadixNode> radix_nodes(keys.size() - 1);
            vector<int32_t> radix_parents(radix_nodes.size());
            vector<NodeCount> node_counts(radix_nodes.size());
            vector<int32_t> node_range_ends(radix_nodes.size());
            scalar::radix_tree<MortonT>(keys, radix_nodes, radix_parents, node_counts, node_range_ends);

            // compute node count totals
            vector<int32_t> node_count_totals(radix_nodes.size());
            transform(node_counts.begin(), node_counts.end(), node_count_totals.begin(), [](const NodeCount& n) -> int32_t { return n.internals + n.leafs; });

            // get octree node offests for each radix tree node
            vector<int32_t> offsets(radix_nodes.size());
            exclusive_scan(node_count_totals.begin(), node_count_totals.end(), offsets.begin(), 0);

            // count the number of octree nodes, and allocate.
            const int32_t num_octree_nodes = 1 + offsets.back() + node_count_totals.back();
            vector<OctreeNode> octree_nodes(num_octree_nodes);

            // map each key to the octree node holding it. depends on the offsets, so it cannot
            // be folded into the radix pass above.
            vector<int32_t> leaf_nodes(keys.size());
            scalar::octree_leaf_nodes(radix_nodes, node_counts, offsets, leaf_nodes);

            // build and return the octree
            scalar::build_octree<MortonT>(keys, radix_nodes, radix_parents, node_counts,
                node_range_ends, node_count_totals, offsets, leaf_nodes, octree_nodes);
            return octree_nodes;
        }
    }
}
