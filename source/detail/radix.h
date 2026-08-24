#pragma once
#include <cstdint>
#include <cassert>
#include <span>
#include <utility>
#include "detail/parallel.h"
#include "nbody/profile.h"

// references:
// (1) [Karas 2012](https://dl.acm.org/doi/10.5555/2383795.2383801)

namespace nbody::detail
{
    using std::span;
    using std::pair;

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
        int32_t cpl(const uint32_t a, const uint32_t b);

        int32_t sign(const int32_t x);

        // Build a radix tree from a sorted list of keys, populating a span of internal nodes in a flat array.
        //
        // This algorithm can be run on a section of nodes, so that the tree can be built in parallel, but must
        // always receive the full range of sorted keys.
        //
        // Algorithm from (1), section 3.2
        void radix_tree(const span<const uint32_t> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus = 1, const int32_t node_offset = 0);

        // Find the parent indices for each node in a radix tree
        //void radix_tree_parents(const span<const RadixNode> nodes, const span<int32_t> parents, int32_t node_offset = 0);
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
        template <auto* impl = scalar::radix_tree>
        void radix_tree_thread_pool(BS::thread_pool& pool, const span<const uint32_t> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus = 1, int32_t node_offset = 0)
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

        template <auto* impl = scalar::radix_tree>
        void radix_tree(const span<const uint32_t> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus = 1, int32_t node_offset = 0)
        {
            static BS::thread_pool pool;
            radix_tree_thread_pool(pool, sorted_keys, nodes, node_parents, node_cpl_deltas, node_cpl_modulus, node_offset);
        }
    }
}
