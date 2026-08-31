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

            for (int32_t i = node_offset; i < static_cast<int32_t>(nodes.size()) + node_offset; ++i)
            {
                // get the common prefix length of the current key with its neighbors
                const int32_t d0 = cpl(sorted_keys, i, i + 1);
                const int32_t d1 = cpl(sorted_keys, i, i - 1);

                // the "direction" of the node is determined by which neighbor has a longer common prefix
                const int32_t d = sign(d0 - d1);

                // find the top of the range of keys that share this prefix
                // 1) "i-d" is the index of the first key to one side of the range
                // 2) "dmin" is smaller than the prefix length of elements within this range - this correlates
                //           to the common prefix length of the parent node.
                // 3) "lmax" is the max possible length of the range, a power of two
                const int32_t dmin = cpl(sorted_keys, i, i - d);
                int32_t lmax = 2;
                while (cpl(sorted_keys, i, i + (lmax * d)) > dmin)
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
                    if (cpl(sorted_keys, i, i + ((l + t) * d)) > dmin)
                        l += t;
                } while (t > 1);
                const int32_t j = i + (l * d);

                // use a binary search to find the split position of the range.
                // this is the index of the last key whose bit following the common prefix is 0.
                const int32_t dnode = cpl(sorted_keys, i, j);
                div = 2;
                t = 0;
                int32_t s = 0;
                do {
                    // the division must round up, otherwise the last step of the search can be
                    // skipped and the split lands short of its true position. unlike "lmax" above,
                    // "l" is not a power of two, so this is not free.
                    t = (l + div - 1) / div;
                    div <<= 1;
                    if (cpl(sorted_keys, i, i + ((s + t) * d)) > dnode)
                        s += t;
                } while (t > 1);
                const int32_t k = i + (s * d) + min(d, 0);

                // the sign of the node's child indices indicates leaf or internal node, which can
                // be determined by comparing each index at the ends of the range to the split index k;
                const int32_t node_index = i - node_offset;
                const int32_t child0 = (min(i, j) == k ? 1 : -1) * k;
                const int32_t child1 = (max(i, j) == k + 1 ? 1 : -1) * (k + 1);
                static constexpr size_t modulus = MortonT::modulus;
                nodes[node_index].child0_index = child0;
                nodes[node_index].child1_index = child1;
                //node_cpl_deltas[i] = (dnode / modulus) - (max(dmin, 0) / modulus);
                //node_child_counts[i] = (child0 >= 0) + (child1 >= 0);
                node_counts[node_index].internals = (dnode / modulus) - (max(dmin, 0) / modulus);
                node_counts[node_index].leafs = (child0 >= 0) + (child1 >= 0);

                // The last key of the node's range. Already computed above -- the range is
                // [min(i,j), max(i,j)] -- and previously discarded, but the octree needs it: a
                // node's escape pointer is the first node of the range starting one key past
                // its own, and nothing else recovers where a range ends.
                //
                // A node's index is always one end of its range, so this doubles as the test
                // for which end: node_range_ends[m] > m says m's range *begins* at m, and so
                // that the range starting at m spans two or more keys rather than being the
                // lone key m.
                node_range_ends[node_index] = max(i, j);
                if (child0 <= 0) node_parents[-child0 - node_offset] = i;
                if (child1 <= 0) node_parents[-child1 - node_offset] = i;
            }
        }
    }

    /*
    namespace simd
    {
        // compute the length of the common prefix between two keys
        int32_t cpl(const uint32_t a, const uint32_t b);

        int32_t sign(const int32_t x);

        // Vectorized counterpart of scalar::radix_tree, with the same contract.
        //
        // Algorithm from (1), section 3.2
        void radix_tree(const span<const uint32_t> sorted_keys, const span<pair<int32_t, int32_t>> nodes, int32_t node_offset = 0);
    }
    */

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
