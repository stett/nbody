#pragma once
#include <concepts>
#include <cstdint>
#include <cassert>
#include <span>
#include <utility>
#include "detail/parallel.h"
#include "nbody/profile.h"
#include "detail/morton.h"

// references:
// (1) [Karas 2012](https://dl.acm.org/doi/10.5555/2383795.2383801)

namespace nbody::detail
{
    using std::span;
    using std::pair;
    using std::get;
    using std::min;
    using std::max;

    struct RadixNode
    {
        // Indices of left and right children.
        // Negative values indicate that the child is an internal node.
        // Positive values indicate that the child is a leaf node, and the value is the index of the key in the sorted list.
        int32_t child0_index;
        int32_t child1_index;
    };

    struct NodeCount
    {
        int32_t internals = 0;
        int32_t leafs = 0;
    };

    /*
    inline int32_t operator+(const NodeCount lhs, const NodeCount rhs)
    {
        return lhs.internals + lhs.leafs + rhs.internals + rhs.leafs;
    }
    */

    //int32_t std::plus<NodeCount>(NodeCount a, NodeCount b)

    // Two interchangeable implementations of the same construction, differing only in how
    // they compute it. Both are held to one oracle by the differential tests in
    // tests/source/test_radix.cpp, which run every case against each of them, so the pair
    // can be compared for agreement as well as for speed.
    //
    // `parallel` below is the inline namespace, so unqualified nbody::detail::radix_tree names
    // the thread pool version. Both are templates now, so a dependent call sees the two through
    // ADL and has to say which it wants: qualify as scalar:: or parallel::.
    namespace scalar
    {
        // compute the length of the common prefix between two keys
        //
        // int32_t rather than BitsT because the index form below reports a missing neighbour as
        // -1, and every comparison against that sentinel is signed.
        template <std::unsigned_integral BitsT>
        int32_t cpl(const BitsT a, const BitsT b)
        {
            return std::countl_zero(static_cast<BitsT>(a ^ b));
        }

        // Matched on the interface rather than on Morton's parameter list, which stops matching
        // whenever the class gains a parameter -- and would then fall through to the overload
        // above, where `a ^ b` is not even valid.
        template <typename MortonT>
            requires requires(const MortonT& m) { m.bits(); }
        int32_t cpl(const MortonT& m0, const MortonT& m1)
        {
            return cpl(m0.bits(), m1.bits());
        }

        // same as cpl above, except operates on indices of a container of sorted keys.
        template <typename MortonT>
        int32_t cpl(const span<const MortonT> sorted_keys, const int32_t i, const int32_t j)
        {
            const int32_t num_keys = static_cast<int32_t>(sorted_keys.size());
            assert(0 <= i && i < num_keys);
            if (j < 0 || j >= num_keys)
                return -1;
            return cpl(sorted_keys[i], sorted_keys[j]);
        }

        inline int32_t sign(const int32_t x)
        {
            return (x > 0) - (x < 0);
        }

        // return either +1 or -1, indicating which neighbor of the key at index i has a longer common prefix with it.
        template <typename MortonT>
        inline int32_t radix_node_direction(const span<const MortonT> sorted_keys, const int32_t i)
        {
            // get the common prefix length of the current key with its neighbors
            const int32_t d0 = cpl(sorted_keys, i, i + 1);
            const int32_t d1 = cpl(sorted_keys, i, i - 1);

            // the "direction" of the node is determined by which neighbor has a longer common prefix
            return sign(d0 - d1);
        }

        template <typename MortonT>
        inline int32_t radix_node_range_length(const span<const MortonT> sorted_keys, const int32_t i, const int32_t direction, const int32_t cpl_parent)
        {
            // lmax is the current approximation for the top of the length of range of keys that shares
            // this prefix. start with 2 and double it until we find a key whose cpl is less than the parent's cpl.
            int32_t lmax = 2;
            while (cpl(sorted_keys, i, i + (lmax * direction)) > cpl_parent)
                lmax <<= 1;

            // use a binary search to find the exact length of the range, along with
            // the index of the last key in the range (might be > i or < i)
            // 1) "div" is the temporary divisor for the binary search, starting at 2 and doubling each iteration
            // 2) "l" is the current length of the range (minus one), starting at 0 and increasing each iteration
            // 3) "t" is the amount by which we're considering increasing the length of the range
            int32_t div = 2;
            int32_t t = 0;
            int32_t l = 0;
            do {
                t = lmax / div;
                div <<= 1;
                if (cpl(sorted_keys, i, i + ((l + t) * direction)) > cpl_parent)
                    l += t;
            } while (t > 1);

            // return the range length
            return l;
        }

        template <typename MortonT>
        inline int32_t radix_node_split(const span<const MortonT> sorted_keys, const int32_t i, const int32_t direction, const int32_t node_len, const int32_t depth)
        {
            // use a binary search to find the split position of the range.
            // this is the index of the last key whose bit following the common prefix is 0.
            int32_t div = 2;
            int32_t t = 0;
            int32_t s = 0;
            do {
                // the division must round up, otherwise the last step of the search can be
                // skipped and the split lands short of its true position. unlike "lmax" above,
                // "l" is not a power of two, so this is not free.
                t = (node_len + div - 1) / div;
                div <<= 1;
                if (cpl(sorted_keys, i, i + ((s + t) * direction)) > depth)
                    s += t;
            } while (t > 1);

            // return the split position
            return i + (s * direction) + min(direction, 0);
        }

        inline std::pair<int32_t, int32_t> radix_node_children(const int32_t i_min, const int32_t i_max, const int32_t i_split)
        {
            // the sign of the node's child indices indicates leaf or internal node, which can
            // be determined by comparing each index at the ends of the range to the split index i_split;
            const int32_t child0 = (i_min == i_split ? 1 : -1) * i_split;
            const int32_t child1 = (i_max == i_split + 1 ? 1 : -1) * (i_split + 1);
            return { child0, child1 };
        }

        inline int32_t radix_node_internal_count(const int32_t depth, const int32_t cpl_parent, const int32_t modulus)
        {
            // "depth" in this case is the number of bits in the common prefix of all keys in the range (ie, the cpl).
            // the number of octree levels resolved by this node is the number of bits in the common prefix divided by
            // the number of bits per level (modulus).
            const int32_t octree_levels = depth / modulus;

            // "cpl_parent" is the common prefix length of the parent node relative to the current node. if we're more
            // than one octree level down from the parent, then the parent node resolves some octree levels of its own.
            //
            // we apply a max(..., 0) clamp here because cpl_parent can be -1 in the case of the root node.
            const int32_t octree_levels_parent = max(cpl_parent, 0) / modulus;

            // to avoid double-counting the octree levels resolved by the parent from the number of octree levels
            // resolved by this node, we subtract the parent's octree levels from our own.
            return octree_levels - octree_levels_parent;
        }

        inline int32_t radix_node_leaf_count(std::pair<int32_t, int32_t> children)
        {
            return (children.first >= 0) + (children.second >= 0);
        }

        // Build a radix tree from a sorted list of keys, populating a span of internal nodes in a flat array.
        //
        // This algorithm can be run on a section of nodes, so that the tree can be built in parallel, but must
        // always receive the full range of sorted keys.
        //
        // Algorithm from (1), section 3.2
        //template <typename MortonT = Morton<>>
        template <typename MortonT>
        void radix_tree(
            const span<const MortonT> sorted_keys,
            const span<RadixNode> nodes,
            const span<int32_t> node_parents,
            const span<NodeCount> node_counts,
            const span<int32_t> node_range_ends,
            const int32_t node_offset = 0)
        {
            NBODY_PROFILE_ZONE_NAMED("radix_tree (scalar)");

            assert(nodes.size() + 1 <= sorted_keys.size());

            for (int32_t i_first = node_offset; i_first < static_cast<int32_t>(nodes.size()) + node_offset; ++i_first)
            {
                // either "+1" or "-1", indicating the direction of the node's range of keys,
                // which is the side with the longer common prefix
                const int32_t dir = radix_node_direction(sorted_keys, i_first);

                // the common prefix length of the parent node is the cpl between this node and
                // it's neighbor in the opposite direction of the range. this is the minimum cpl
                // of all keys in this node's range, and is used to find the top of the range of
                // keys that share this prefix.
                const int32_t cpl_parent = cpl(sorted_keys, i_first, i_first - dir);

                // get the length of this node range
                const int32_t node_len = radix_node_range_length(sorted_keys, i_first, dir, cpl_parent);

                // find the index of the last range that shares this prefix
                const int32_t i_last = i_first + (node_len * dir);
                const int32_t i_min = min(i_first, i_last);
                const int32_t i_max = max(i_first, i_last);

                // compute the "depth" of the radix node - that is, the number of bits in the
                // common prefix of all keys in the range (ie, the cpl).
                const int32_t depth = cpl(sorted_keys, i_first, i_last);

                // use a binary search to find the split position of the range.
                // this is the index of the last key whose bit following the common prefix is 0.
                const int32_t i_split = radix_node_split(sorted_keys, i_first, dir, node_len, depth);

                // the sign of the node's child indices indicates leaf or internal node, which can
                // be determined by comparing each index at the ends of the range to the split index i_split;
                const auto i_children = radix_node_children(i_min, i_max, i_split);
                const int32_t node_index = i_first - node_offset;
                nodes[node_index].child0_index = i_children.first;
                nodes[node_index].child1_index = i_children.second;
                node_counts[node_index].internals = radix_node_internal_count(depth, cpl_parent, MortonT::modulus);
                node_counts[node_index].leafs = radix_node_leaf_count(i_children);
                node_range_ends[node_index] = i_max; // used in build_octree
                if (i_children.first  <= 0) node_parents[-i_children.first]  = i_first;
                if (i_children.second <= 0) node_parents[-i_children.second] = i_first;
            }
        }

        //void radix_tree_order_dfs(const span<const int32_t> node_range_ends);
    }

    inline namespace parallel
    {
        // Parallelized version of any of the radix_tree implementations, which can be selected by the `impl` template parameter.
        //
        // This version of the algorithm must receive the full range of keys and nodes.
        // KeyT is named first so the default implementation can refer to it: the level
        // arithmetic now comes from KeyT::modulus, so an implementation is only selectable
        // once the key type is fixed.
        // The trailing spans are indexed by node, so a partial build writes into a subspan of
        // each alongside the nodes themselves. node_parents is the exception: it is indexed by
        // *child*, so it is passed whole.
        template <typename KeyT, auto* impl = scalar::radix_tree<KeyT>>
        void radix_tree_thread_pool(BS::thread_pool& pool, const span<const KeyT> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<NodeCount> node_counts, const span<int32_t> node_range_ends, int32_t node_offset = 0)
        {
            NBODY_PROFILE_ZONE_NAMED("radix_tree (parallel)");
            assert(node_offset == 0);
            assert(nodes.size() == sorted_keys.size() - 1);
            assert(node_counts.size() == nodes.size());
            assert(node_range_ends.size() == nodes.size());
            detail::parallel_blocks(pool, nodes.size(),
                [&](const size_t begin, const size_t end)
                {
                    impl(sorted_keys, nodes.subspan(begin, end - begin), node_parents,
                        node_counts.subspan(begin, end - begin),
                        node_range_ends.subspan(begin, end - begin), static_cast<int32_t>(begin));
                }
            );
        }

        template <typename KeyT = Morton<>, auto* impl = scalar::radix_tree<KeyT>>
        void radix_tree(const span<const KeyT> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<NodeCount> node_counts, const span<int32_t> node_range_ends, int32_t node_offset = 0)
        {
            static BS::thread_pool pool;
            radix_tree_thread_pool<KeyT, impl>(pool, sorted_keys, nodes, node_parents, node_counts, node_range_ends, node_offset);
        }
    }
}
