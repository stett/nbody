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

        template <typename MortonT>
        void build_octree(
            span<const MortonT> keys,
            span<const RadixNode> radix_nodes,
            span<const int32_t> radix_parents,
            span<const NodeCount> node_counts,
            span<const int32_t> node_totals,
            span<const int32_t> node_offsets,
            span<OctreeNode> octree_nodes)
        {

            // count the number of octree nodes, allocate.
            // make sure it matches the output we expect
            const int32_t num_nodes = 1 + node_offsets.back() + node_totals.back();
            assert(octree_nodes.size() == num_nodes);

            // populate the root node
            octree_nodes[0] = { .parent = 0, .next = 0, .child = 1 };

            const auto find_octree_parent = [&](const int32_t i_radix) -> int32_t
            {
                // find the first parent radix node index which produced a chain of octree nodes
                int32_t i_radix_parent = radix_parents[i_radix];
                while (i_radix_parent > 0 && node_counts[i_radix_parent].internals == 0)
                    i_radix_parent = radix_parents[i_radix_parent];

                // find the index of the last oct node in the chain produced by this radix node
                const int32_t i_octree_parent
                    = (i_radix_parent > 0)
                    ? (node_offsets[i_radix_parent] + node_totals[i_radix_parent] + 1)
                    : 0;

                // return the parent octree node index
                return i_octree_parent;
            };

            // run through all radix nodes, populating octree nodes from them
            for (int32_t i_radix = 0; i_radix < radix_nodes.size(); ++i_radix)
            {
                // get the first octree node index
                const int32_t i_node_0 = 1 + node_offsets[i_radix];
                const NodeCount node_count = node_counts[i_radix];
                const int32_t i_node_end = i_node_0 + node_count.internals + node_count.leafs;
                //int32_t i_node = i_node_0;

                // populate the first node's parent pointer
                octree_nodes[i_node_0].parent = find_octree_parent(i_radix);

                // populate corresponding intermediate nodes
                for (int32_t i_internal = 0; i_internal < node_count.internals; ++i_internal)
                {
                    const int32_t i_node = i_node_0 + i_internal;

                    // store the parent. except for the first node, the parent will always be the previously added node
                    if (i_node > i_node_0)
                        octree_nodes[i_node].parent = i_node - 1;

                    // the "next" pointer will be the next node past the end of our range, or the root if there is none
                    octree_nodes[i_node].next = i_node_end < octree_nodes.size() ? i_node_end : 0;

                    // the "child" pointer is the pointer to our first child. since we're intermediate, there should
                    // always be at least one child coming up next, either the next internal or the first leaf
                    octree_nodes[i_node].child = i_node + 1;

                    // TODO: pack this value into the child node's sign bit
                    octree_nodes[i_node].is_leaf = false;
                }

                // make a temp function for getting the next leaf index
                int32_t i_radix_leaf_node = i_radix; // index to track the current radix leaf index
                bool i_radix_leaf_child = 0;
                const auto next_leaf = [&]() -> int32_t
                {
                    while (true)
                    {
                        const RadixNode& radix_leaf_node = radix_nodes[i_radix_leaf_node];
                        const int32_t i_key = i_radix_leaf_child == 0 ? radix_leaf_node.child0_index : radix_leaf_node.child1_index;
                        i_radix_leaf_child = !i_radix_leaf_child;
                        if (!i_radix_leaf_child)
                        {
                            ++i_radix_leaf_node;
                        }

                        if (i_key >= 0)
                        {
                            assert(i_key < keys.size());
                            return i_key;
                        }
                    }
                };

                // populate corresponding leaf nodes after the internals
                for (int32_t i_leaf = 0; i_leaf < node_count.leafs; ++i_leaf)
                {
                    const int32_t i_node = i_node_0 + node_count.internals + i_leaf;

                    // the parent for all the children will be the last intermediate that was added,
                    // or i_node_0's parent (which we already found) if there's no intermediate ancestor.
                    if (i_node > i_node_0)
                        octree_nodes[i_node].parent
                            = node_count.internals > 0
                            ? i_node_0 + node_count.internals - 1
                            : octree_nodes[i_node_0].parent;

                    // the "next" pointer will be the next node - it'll either be the next leaf, or one past the end
                    // of our range. if this is the last node, we'll give the root. This is exactly the same as
                    // with intermediate nodes
                    octree_nodes[i_node].next = i_node + 1 < octree_nodes.size() ? i_node + 1 : 0;

                    // for leaf nodes, the child index will indicate an index into the original keys array.
                    //
                    // this octree node might span many leaf nodes
                    //const RadixNode& leaf_radix = radix_nodes[i_radix + (i_leaf / 2)];
                    //if (leaf_radix.child0_index)
                    //octree_nodes[i_node].child = (i_leaf % 2) ? leaf_radix.child0_index : leaf_radix.child1_index;
                    octree_nodes[i_node].child = next_leaf();

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
            scalar::radix_tree<MortonT>(keys, radix_nodes, radix_parents, node_counts);

            // compute node count totals
            vector<int32_t> node_count_totals(radix_nodes.size());
            transform(node_counts.begin(), node_counts.end(), node_count_totals.begin(), [](const NodeCount& n) -> int32_t { return n.internals + n.leafs; });

            // get octree node offests for each radix tree node
            vector<int32_t> offsets(radix_nodes.size());
            exclusive_scan(node_count_totals.begin(), node_count_totals.end(), offsets.begin(), 0);

            // count the number of octree nodes, and allocate.
            const int32_t num_octree_nodes = 1 + offsets.back() + node_count_totals.back();
            vector<OctreeNode> octree_nodes(num_octree_nodes);

            // build and return the octree
            scalar::build_octree<MortonT>(keys, radix_nodes, radix_parents, node_counts, node_count_totals, offsets, octree_nodes);
            return octree_nodes;
        }
    }
}
