#include <atomic>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <bit>
#include "detail/radix.h"

namespace
{
    using std::vector;
    using std::pair;
    using std::countl_zero;
	using nbody::detail::radix_tree;

    using Node = pair<int32_t, int32_t>;
    using Nodes = vector<Node>;
    using Keys = vector<uint32_t>;

    int32_t key_cpl(const uint32_t a, const uint32_t b)
    {
        return static_cast<int32_t>(countl_zero(a ^ b));
    }

    // Reference tree builder, used as an oracle for the implementation under test.
    //
    // Deliberately naive: recurse top-down over key ranges and find each split with a linear
    // scan. It shares no logic with radix_tree() - no direction, no range search, no binary
    // search - so the two agreeing is real evidence rather than a repeated mistake.
    //
    // The node index of a child is known up front because the children of a range split at k
    // are always k and k+1, which is the same convention radix_tree() emits.
    void reference_tree(const Keys& keys, const int32_t a, const int32_t b, const int32_t index, Nodes& nodes)
    {
        assert(a < b);

        // every key in the range shares this prefix; the split is where the following bit flips
        const int32_t prefix = key_cpl(keys[a], keys[b]);

        // last index still on the 0 side of the split bit
        int32_t k = a;
        while (k + 1 < b && key_cpl(keys[a], keys[k + 1]) > prefix)
            ++k;

        // negative child index means internal node, positive means leaf
        nodes[index] = {
            (k == a ? 1 : -1) * k,
            (k + 1 == b ? 1 : -1) * (k + 1),
        };

        if (k != a)
            reference_tree(keys, a, k, k, nodes);
        if (k + 1 != b)
            reference_tree(keys, k + 1, b, k + 1, nodes);
    }

    Nodes reference_tree(const Keys& keys)
    {
        Nodes nodes(keys.size() - 1, Node{ INT32_MIN, INT32_MIN });
        reference_tree(keys, 0, static_cast<int32_t>(keys.size()) - 1, 0, nodes);
        return nodes;
    }

    Nodes actual_tree(const Keys& keys)
    {
        Nodes nodes(keys.size() - 1, Node{ INT32_MIN, INT32_MIN });
        radix_tree(keys, nodes);
        return nodes;
    }

    std::string describe(const Keys& keys)
    {
        std::string s;
        for (const uint32_t key : keys)
            s += std::to_string(key) + " ";
        return s;
    }

    // Structural invariants that hold for any valid radix tree over n distinct keys, checked
    // without reference to either builder: walking from the root must reach all n leaves and
    // all n-1 internal nodes, each exactly once.
    void check_structure(const Keys& keys, const Nodes& nodes)
    {
        const int32_t num_keys = static_cast<int32_t>(keys.size());
        vector<int> leaf_visits(num_keys, 0);
        vector<int> node_visits(nodes.size(), 0);

        vector<int32_t> stack{ 0 };
        while (!stack.empty())
        {
            const int32_t index = stack.back();
            stack.pop_back();

            REQUIRE(index >= 0);
            REQUIRE(index < static_cast<int32_t>(nodes.size()));
            REQUIRE(++node_visits[index] == 1);

            for (const int32_t child : { nodes[index].first, nodes[index].second })
            {
                // the sign convention is ambiguous at zero, but only in principle: internal
                // node 0 is always the root, so it is never anyone's child, and a child of 0
                // always means leaf 0
                if (child >= 0)
                {
                    // leaf
                    REQUIRE(child < num_keys);
                    REQUIRE(++leaf_visits[child] == 1);
                }
                else
                {
                    // internal
                    const int32_t next = -child;
                    REQUIRE(next > 0);
                    stack.push_back(next);
                }
            }
        }

        for (const int visits : node_visits)
            REQUIRE(visits == 1);
        for (const int visits : leaf_visits)
            REQUIRE(visits == 1);
    }

    void check_tree(const Keys& keys)
    {
        INFO("keys: " << describe(keys));
        const Nodes expected = reference_tree(keys);
        const Nodes actual = actual_tree(keys);
        for (size_t i = 0; i < expected.size(); ++i)
        {
            INFO("node " << i
                << ": expected {" << expected[i].first << ", " << expected[i].second << "}"
                << " actual {" << actual[i].first << ", " << actual[i].second << "}");
            REQUIRE(actual[i] == expected[i]);
        }
        check_structure(keys, actual);
    }

    // Karras' construction requires strictly increasing keys.
    Keys sanitize(Keys keys)
    {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }
}

TEST_CASE("radix tree", "[radix]")
{
    SECTION("hand computed")
    {
        // must be ordered
        const Keys keys{
            0b00001,
            0b00010,
            0b01001,
            0b01110,
            0b11010,
            0b11101,
        };

        // expected outcome
        const Nodes nodes_expected{
            { -3, -4 },
            { 0, 1 },
            { 2, 3 },
            { -1, -2 },
            { 4, 5 },
        };

        // the reference builder must reproduce the one case we worked out by hand, otherwise
        // it is not fit to serve as an oracle for everything below
        REQUIRE(reference_tree(keys) == nodes_expected);

        const Nodes nodes = actual_tree(keys);
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            INFO("node " << i << ": " << nodes[i].first << ", " << nodes[i].second);
            REQUIRE(nodes[i] == nodes_expected[i]);
        }
    }

    SECTION("two keys")
    {
        check_tree(Keys{ 0b0, 0b1 });
        check_tree(Keys{ 0x00000000u, 0xFFFFFFFFu });
    }

    SECTION("dense low keys")
    {
        // 0..n-1 for a range of n: maximally deep left leaning chains
        for (int32_t num_keys = 2; num_keys <= 40; ++num_keys)
        {
            Keys keys(num_keys);
            for (int32_t i = 0; i < num_keys; ++i)
                keys[i] = static_cast<uint32_t>(i);
            check_tree(keys);
        }
    }

    SECTION("single bit keys")
    {
        // every key differs from its neighbors at a different bit depth
        Keys keys{ 0 };
        for (int32_t bit = 0; bit < 30; ++bit)
            keys.push_back(1u << bit);
        check_tree(sanitize(keys));
    }

    SECTION("wide split")
    {
        // two tight clusters separated at the top bit: the root split is far from both ends
        Keys keys;
        for (uint32_t i = 0; i < 16; ++i)
            keys.push_back(i);
        for (uint32_t i = 0; i < 16; ++i)
            keys.push_back(0x20000000u | i);
        check_tree(sanitize(keys));
    }

    SECTION("lopsided split")
    {
        // one key alone on the far side of the root split, forcing a long unbalanced spine
        Keys keys;
        for (uint32_t i = 0; i < 31; ++i)
            keys.push_back(i);
        keys.push_back(0x3FFFFFFFu);
        check_tree(sanitize(keys));
    }

    SECTION("random keys")
    {
        // full 30 bit morton range, uniform: mostly shallow trees with random shapes
        for (uint32_t seed = 0; seed < 200; ++seed)
        {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<uint32_t> key_dist(0, (1u << 30) - 1);
            std::uniform_int_distribution<size_t> size_dist(2, 64);

            Keys keys(size_dist(rng));
            for (uint32_t& key : keys)
                key = key_dist(rng);

            keys = sanitize(keys);
            if (keys.size() < 2)
                continue;
            check_tree(keys);
        }
    }

    SECTION("random clustered keys")
    {
        // keys drawn from a few tight clusters, imitating the morton codes of a galaxy:
        // long shared prefixes, so deep chains of internal nodes with single leaf children
        for (uint32_t seed = 0; seed < 200; ++seed)
        {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<uint32_t> cluster_dist(0, (1u << 30) - 1);
            std::uniform_int_distribution<uint32_t> spread_dist(0, 1u << 8);
            std::uniform_int_distribution<size_t> count_dist(1, 12);

            Keys keys;
            for (int32_t cluster = 0; cluster < 4; ++cluster)
            {
                const uint32_t center = cluster_dist(rng);
                const size_t count = count_dist(rng);
                for (size_t i = 0; i < count; ++i)
                    keys.push_back((center + spread_dist(rng)) & ((1u << 30) - 1));
            }

            keys = sanitize(keys);
            if (keys.size() < 2)
                continue;
            check_tree(keys);
        }
    }

    SECTION("partial node range")
    {
        // radix_tree() is documented as buildable over a section of the node array; a prefix
        // of the nodes must come out the same as when the whole array is built at once
        std::mt19937 rng(1234);
        std::uniform_int_distribution<uint32_t> key_dist(0, (1u << 30) - 1);

        Keys keys(64);
        for (uint32_t& key : keys)
            key = key_dist(rng);
        keys = sanitize(keys);

        const Nodes expected = actual_tree(keys);
        for (size_t count = 1; count < keys.size(); ++count)
        {
            Nodes nodes(count, Node{ INT32_MIN, INT32_MIN });
            radix_tree(keys, nodes);
            INFO("first " << count << " nodes");
            for (size_t i = 0; i < count; ++i)
                REQUIRE(nodes[i] == expected[i]);
        }
    }
}
