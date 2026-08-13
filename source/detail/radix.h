#pragma once
#include <cassert>
#include <cmath>
#include <span>
#include <bit>

// references:
// (1) [Karas 2012](https://dl.acm.org/doi/10.5555/2383795.2383801)

namespace nbody::detail
{
    using std::span;
    using std::pair;
    using std::get;
    using std::min;

    namespace
    {
        // compute the length of the common prefix between two keys
        int32_t cpl(const uint32_t a, const uint32_t b)
        {
            return std::countl_zero(a ^ b);
        }

        int32_t sign(const int32_t x)
        {
            return (x > 0) - (x < 0);
        }

    }

    // Build a radix tree from a sorted list of keys, populating a span of internal nodes in a flat array.
    // This algorithm can be run on a section of keys and nodes, so that the tree can be built in parallel.
    // The keys must be sorted in ascending order.
    //
    // Algorithm from (1), section 3.2
    void radix_tree(const span<const uint32_t> sorted_keys, const span<pair<int32_t, int32_t>> nodes)
    {
        assert(sorted_keys.size() == nodes.size() + 1);

        const auto index_cpl = [&](const int32_t i, const int32_t j) -> int32_t
        {
            const int32_t num_keys = static_cast<int32_t>(sorted_keys.size());
            assert(0 <= i && i < num_keys);
            if (j < 0 || j >= num_keys)
                return -1;
            return cpl(sorted_keys[i], sorted_keys[j]);
        };

        for (int32_t i = 0; i < static_cast<int32_t>(nodes.size()); ++i)
        {
            // get the common prefix length of the current key with its neighbors
            const int32_t d0 = index_cpl(i, i + 1);
            const int32_t d1 = index_cpl(i, i - 1);

            // the "direction" of the node is determined by which neighbor has a longer common prefix
            const int32_t d = sign(d0 - d1);

            // find the top of the range of keys that share this prefix
            // 1) "i-d" is the index of the first key to one side of the range
            // 2) "dmin" is smaller than the prefix length of elements within this range
            // 3) "lmax" is the max possible length of the range, a power of two
            //const uint32_t dmin = i - d >= 0 ? cpl(sorted_keys[i], sorted_keys[i - d]) : -1;
            const int32_t dmin = index_cpl(i, i - d);
            int32_t lmax = 2;
            while (index_cpl(i, i + (lmax * d)) > dmin)
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
                if (index_cpl(i, i + ((l + t) * d)) > dmin)
                    l += t;
            } while (t > 1);
            const int32_t j = i + (l * d);

            // use a binary search to find the split position of the range.
            // this is the index of the last key whose bit following the common prefix is 0.
            const int32_t dnode = index_cpl(i, j);
            div = 2;
            t = 0;
            int32_t s = 0;
            do {
                t = l / div;
                div <<= 1;
                if (index_cpl(i, i + ((s + t) * d)) > dnode)
                    s += t;
            } while (t > 1);
            const int32_t k = i + (s * d) + min(d, 0);

            // get the range indices in order, and store them in the node.
            // the sign of the node's child indices indicates leaf or internal node, which can
            // be determined by comparing each index at the ends of the range to the split index k;
            int32_t i0 = i;
            int32_t i1 = j;
            if (i0 > i1)
                std::swap(i0, i1);
            get<0>(nodes[i]) = (i0 == k ? 1 : -1) * k;
            get<1>(nodes[i]) = (i1 == k + 1 ? 1 : -1) * (k + 1);
        }
    }

    template <size_t N>
    void radix_tree(const span<const uint32_t, N> keys, const span<pair<int32_t, int32_t>, N-1> nodes)
    {
        radix_tree(keys, nodes);
    }
}
