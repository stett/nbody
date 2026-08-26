#include "detail/radix.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <simde/x86/avx2.h>

namespace nbody::detail
{
    using std::get;
    using std::min;
    using std::max;

    namespace scalar
    {
        /*
        // "common prefix length"
        //
        // This gives the number of common leading bits in two keys.
        int32_t cpl(const uint32_t a, const uint32_t b)
        {
            return std::countl_zero(a ^ b);
        }

        // same as cpl above, except operates on indices of a container of sorted keys.
        int32_t cpl(const span<const uint32_t> sorted_keys, const int32_t i, const int32_t j)
        {
            const int32_t num_keys = static_cast<int32_t>(sorted_keys.size());
            assert(0 <= i && i < num_keys);
            if (j < 0 || j >= num_keys)
                return -1;
            return cpl(sorted_keys[i], sorted_keys[j]);
        }

        // "signum"
        int32_t sign(const int32_t x)
        {
            return (x > 0) - (x < 0);
        }

        void radix_tree(const span<const uint32_t> sorted_keys, const span<RadixNode> nodes, const span<int32_t> node_parents, const span<int32_t> node_cpl_deltas, const int32_t node_cpl_modulus, const int32_t node_offset)
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
                node_cpl_deltas[i] = (dnode / node_cpl_modulus) - (max(dmin, 0) / node_cpl_modulus);
                node_parents[k] = i;
                node_parents[k + 1] = i;
            }
        }
        */
    }

    /*
    namespace simd
    {
        int32_t cpl(const uint32_t a, const uint32_t b)
        {
            return std::countl_zero(a ^ b);
        }

        int32_t sign(const int32_t x)
        {
            return (x > 0) - (x < 0);
        }

        void radix_tree(const span<const uint32_t> sorted_keys, const span<pair<int32_t, int32_t>> nodes, int32_t node_offset)
        {
            NBODY_PROFILE_ZONE_NAMED("radix_tree (simd)");

            assert(nodes.size() + 1 <= sorted_keys.size());

            const int32_t num_keys = static_cast<int32_t>(sorted_keys.size());

            const auto index_cpl = [num_keys, &sorted_keys](const int32_t i, const int32_t j) -> int32_t
            {
                assert(0 <= i && i < num_keys);
                if (j < 0 || j >= num_keys)
                    return -1;
                return cpl(sorted_keys[i], sorted_keys[j]);
            };

            for (int32_t i = node_offset; i < static_cast<int32_t>(nodes.size()) + node_offset; ++i)
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
                    // the division must round up, otherwise the last step of the search can be
                    // skipped and the split lands short of its true position. unlike "lmax" above,
                    // "l" is not a power of two, so this is not free.
                    t = (l + div - 1) / div;
                    div <<= 1;
                    if (index_cpl(i, i + ((s + t) * d)) > dnode)
                        s += t;
                } while (t > 1);
                const int32_t k = i + (s * d) + min(d, 0);

                // get the range indices in order, and store them in the node.
                // the sign of the node's child indices indicates leaf or internal node, which can
                // be determined by comparing each index at the ends of the range to the split index k;
                get<0>(nodes[i - node_offset]) = (min(i, j) == k ? 1 : -1) * k;
                get<1>(nodes[i - node_offset]) = (max(i, j) == k + 1 ? 1 : -1) * (k + 1);
            }
        }
    }
    */
}
