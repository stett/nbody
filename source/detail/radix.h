#pragma once
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
        // Difference between the parent's common prefix lenght and this node's
        //int32_t depth_delta;
        //int32_t cpl;
        //int32_t cpl_parent;

        // Indices of left and right children.
        // Negative values indicate that the child is an internal node.
        // Positive values indicate that the child is a leaf node, and the value is the index of the key in the sorted list.
        int32_t child0_index;
        int32_t child1_index;
    };

    // Two interchangeable implementations of the same construction, differing only in how
    // they compute it. Both are held to one oracle by the differential tests in
    // tests/source/test_radix.cpp, which run every case against each of them, so the pair
    // can be compared for agreement as well as for speed.
    //
    // `scalar` is an inline namespace, so unqualified nbody::detail::radix_tree still names
    // it and callers get the plain implementation unless they ask for another by name.
    namespace scalar
    {
        // compute the length of the common prefix between two keys
        template <typename BitsT>
        BitsT cpl(const BitsT& a, const BitsT& b)
        {
            return std::countl_zero(a ^ b);
        }

        template <typename BitsT, size_t modulus>
        BitsT cpl(const Morton<BitsT, modulus>& m0, const Morton<BitsT, modulus>& m1)
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

        int32_t sign(const int32_t x)
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
        void radix_tree(const span<const MortonT> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_offset = 0)
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
                nodes[node_index].child0_index = child0;
                nodes[node_index].child1_index = child1;
                node_cpl_deltas[i] = (dnode / MortonT::modulus) - (max(dmin, 0) / MortonT::modulus);
                node_parents[k] = i;
                node_parents[k + 1] = i;
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
        template <typename KeyT, auto* impl = scalar::radix_tree>
        void radix_tree_thread_pool(BS::thread_pool& pool, const span<const KeyT> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus = 1, int32_t node_offset = 0)
        {
            NBODY_PROFILE_ZONE_NAMED("radix_tree (parallel)");
            assert(node_offset == 0);
            assert(nodes.size() == sorted_keys.size() - 1);
            detail::parallel_blocks(pool, nodes.size(),
                [&](const size_t begin, const size_t end)
                {
                    impl(sorted_keys, nodes.subspan(begin, end - begin), node_parents, node_cpl_deltas, node_cpl_modulus, begin);
                }
            );
        }

        template <typename KeyT = Morton<>, auto* impl = scalar::radix_tree>
        void radix_tree(const span<const KeyT> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus = 1, int32_t node_offset = 0)
        {
            static BS::thread_pool pool;
            radix_tree_thread_pool(pool, sorted_keys, nodes, node_parents, node_cpl_deltas, node_cpl_modulus, node_offset);
        }
    }
}
