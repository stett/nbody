#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <span>
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
    using std::span;
    using std::countl_zero;
    using namespace nbody::detail;

    using Node = RadixNode;
    using Nodes = vector<Node>;
    using Deltas = vector<int32_t>;
    using Keys = vector<uint32_t>;

    // A node's prefix increment used to be RadixNode::depth_delta; it now comes back in a
    // span running parallel to the nodes, so a built tree is the two of them together and
    // every case has to check both halves.
    struct Tree
    {
        Nodes nodes;
        Deltas cpl_deltas;
    };

    // Fillers for what no builder has written yet. Every field is checked, so these have to
    // be values no correct build can produce.
    constexpr Node unwritten{ INT32_MIN, INT32_MIN };
    constexpr int32_t unwritten_delta = INT32_MIN;

    // The increments are compared as raw bit counts, which is what a modulus of one asks the
    // builder for. Folding several bits into one octree level is a different contract and is
    // not covered here.
    constexpr int32_t cpl_modulus = 1;

    // Compared field by field rather than through an operator==, which would have to live in
    // nbody::detail to be found by ADL from inside std::equal.
    bool same(const Node& a, const Node& b)
    {
        return a.child0_index == b.child0_index &&
            a.child1_index == b.child1_index;
    }

    std::string describe(const Node& node, const int32_t cpl_delta)
    {
        return "{ delta " + std::to_string(cpl_delta)
            + ", left " + std::to_string(node.child0_index)
            + ", right " + std::to_string(node.child1_index) + " }";
    }

    // Every implementation of the construction must produce the same tree, so each one is
    // held to the same oracle rather than to the others. Naming them here means a new
    // implementation is covered by every case below the moment it is added to this list,
    // and a failure reports which one diverged.
    struct Builder
    {
        const char* name;
        void (*build)(span<const uint32_t>, span<Node>, span<int32_t>, span<int32_t>, int32_t, int32_t);
        bool supports_partial_build;
    };

    constexpr Builder builders[]{
        { "scalar",             &scalar::radix_tree, true },
        //{ "simd",               &simd::radix_tree  , true },
        { "parallel-scalar",    &parallel::radix_tree<scalar::radix_tree>, false },
        //{ "parallel-simd",      &parallel::radix_tree<simd::radix_tree>  , false },
    };

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
    //
    // The prefix increment comes for free here. radix_tree() has to recover the parent's
    // prefix length from its neighbours (as `dmin`), but a top-down recursion already holds
    // it, so the two arrive at the same number by genuinely different routes.
    void reference_tree(const Keys& keys, const int32_t a, const int32_t b, const int32_t index,
                        const int32_t parent_prefix, Tree& tree)
    {
        assert(a < b);

        // every key in the range shares this prefix; the split is where the following bit flips
        const int32_t prefix = key_cpl(keys[a], keys[b]);

        // last index still on the 0 side of the split bit
        int32_t k = a;
        while (k + 1 < b && key_cpl(keys[a], keys[k + 1]) > prefix)
            ++k;

        // negative child index means internal node, positive means leaf
        tree.nodes[index] = {
            (k == a ? 1 : -1) * k,
            (k + 1 == b ? 1 : -1) * (k + 1),
        };

        // radix_tree() divides both prefix lengths by the modulus before subtracting, which
        // is the same subtraction only while the modulus is one
        static_assert(cpl_modulus == 1, "the oracle measures increments in whole bits");
        tree.cpl_deltas[index] = prefix - parent_prefix;

        if (k != a)
            reference_tree(keys, a, k, k, prefix, tree);
        if (k + 1 != b)
            reference_tree(keys, k + 1, b, k + 1, prefix, tree);
    }

    Tree empty_tree(const size_t num_nodes)
    {
        return { Nodes(num_nodes, unwritten), Deltas(num_nodes, unwritten_delta) };
    }

    Tree reference_tree(const Keys& keys)
    {
        Tree tree = empty_tree(keys.size() - 1);

        // The root has no parent. radix_tree() reads the missing neighbour through index_cpl,
        // which reports out of range as -1, so the root's increment is measured from -1 and
        // the oracle has to start from the same place.
        reference_tree(keys, 0, static_cast<int32_t>(keys.size()) - 1, 0, -1, tree);
        return tree;
    }

    Tree actual_tree(const Builder& builder, const Keys& keys, const size_t num_nodes)
    {
        Tree tree = empty_tree(num_nodes);

        // Parents are written at child indices, and the last leaf's index is one past the last
        // internal node, so this buffer is a key long rather than a node long. Its contents are
        // not asserted on here.
        vector<int32_t> parents(keys.size(), unwritten_delta);

        builder.build(keys, tree.nodes, parents, tree.cpl_deltas, cpl_modulus, 0);
        return tree;
    }

    Tree actual_tree(const Builder& builder, const Keys& keys)
    {
        return actual_tree(builder, keys, keys.size() - 1);
    }

    std::string describe(const Keys& keys)
    {
        std::string s;
        for (const uint32_t key : keys)
            s += std::to_string(key) + " ";
        return s;
    }

    // Invariants that hold for any valid radix tree over n distinct keys, checked against the
    // keys themselves rather than against either builder.
    //
    // Walking down from the root reaches all n leaves and all n-1 internal nodes exactly once.
    // The walk also carries each node's key range, which makes the child indices, the leaf
    // flags and the increments all checkable in passing: the range determines where the split
    // must fall, and the prefix length of the range determines what the increments must sum to.
    void check_structure(const Keys& keys, const Tree& tree)
    {
        const int32_t num_keys = static_cast<int32_t>(keys.size());
        const int32_t num_nodes = static_cast<int32_t>(tree.nodes.size());
        vector<int> leaf_visits(num_keys, 0);
        vector<int> node_visits(tree.nodes.size(), 0);

        struct Frame
        {
            int32_t index;      // internal node
            int32_t first;      // first key of its range
            int32_t last;       // last key of its range
            int32_t delta_sum;  // sum of the increments from the root down to and including this node
        };

        vector<Frame> stack{ { 0, 0, num_keys - 1, tree.cpl_deltas[0] } };
        while (!stack.empty())
        {
            const Frame frame = stack.back();
            stack.pop_back();

            INFO("node " << frame.index << " covering keys [" << frame.first << ", " << frame.last << "]");
            REQUIRE(frame.index >= 0);
            REQUIRE(frame.index < num_nodes);
            REQUIRE(++node_visits[frame.index] == 1);

            const Node& node = tree.nodes[frame.index];

            // a child always resolves at least one bit more than its parent, so the increment
            // is never zero
            REQUIRE(tree.cpl_deltas[frame.index] >= 1);

            // the increments telescope: summed from the root they reach this range's own prefix
            // length, offset by one because the root measures from the absent parent's -1
            REQUIRE(frame.delta_sum == key_cpl(keys[frame.first], keys[frame.last]) + 1);

            // both children name the split, so either index recovers it and they must agree
            const int32_t k = std::abs(node.child0_index);
            REQUIRE(k >= frame.first);
            REQUIRE(k < frame.last);
            REQUIRE(std::abs(node.child1_index) == k + 1);

            // a child is a leaf exactly when its side of the split is a single key, and the
            // sign of the stored index has to say so. index 0 is unambiguous despite -0 == 0,
            // because internal node 0 is always the root and so is never a child.
            const bool left_is_leaf = (k == frame.first);
            const bool right_is_leaf = (k + 1 == frame.last);
            REQUIRE((node.child0_index >= 0) == left_is_leaf);
            REQUIRE((node.child1_index >= 0) == right_is_leaf);

            if (left_is_leaf)
            {
                REQUIRE(++leaf_visits[k] == 1);
            }
            else
            {
                REQUIRE(k < num_nodes);
                stack.push_back({ k, frame.first, k, frame.delta_sum + tree.cpl_deltas[k] });
            }

            if (right_is_leaf)
            {
                REQUIRE(++leaf_visits[k + 1] == 1);
            }
            else
            {
                REQUIRE(k + 1 < num_nodes);
                stack.push_back({ k + 1, k + 1, frame.last, frame.delta_sum + tree.cpl_deltas[k + 1] });
            }
        }

        for (const int visits : node_visits)
            REQUIRE(visits == 1);
        for (const int visits : leaf_visits)
            REQUIRE(visits == 1);
    }

    // Compares a prefix of a built tree against the oracle, node and increment together.
    void check_nodes(const Tree& actual, const Tree& expected, const size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            INFO("node " << i
                << ": expected " << describe(expected.nodes[i], expected.cpl_deltas[i])
                << " actual " << describe(actual.nodes[i], actual.cpl_deltas[i]));
            REQUIRE(same(actual.nodes[i], expected.nodes[i]));
            REQUIRE(actual.cpl_deltas[i] == expected.cpl_deltas[i]);
        }
    }

    void check_tree(const Keys& keys)
    {
        INFO("keys: " << describe(keys));

        // built once: the oracle does not depend on which implementation is under test
        const Tree expected = reference_tree(keys);

        for (const Builder& builder : builders)
        {
            INFO("builder: " << builder.name);
            const Tree actual = actual_tree(builder, keys);
            check_nodes(actual, expected, expected.nodes.size());
            check_structure(keys, actual);
        }
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

        // Expected outcome, worked out by hand from the key bits.
        //
        // The ranges and splits are: node 0 covers keys [0,5] and splits at 3, node 3 covers
        // [0,3] splitting at 1, node 1 covers [0,1] splitting at 0, node 2 covers [2,3], and
        // node 4 covers [4,5]. Their prefix lengths (as 32 bit counts, so 27 means the keys
        // first differ at bit 4) are 27, 28, 30, 29 and 29, and each increment is that prefix
        // minus its parent's - the root measuring from -1, giving 27 - (-1) = 28.
        const Tree expected{
            Nodes{
                { -3, -4 },
                {  0,  1 },
                {  2,  3 },
                { -1, -2 },
                {  4,  5 },
            },
            Deltas{ 28, 2, 1, 1, 2 },
        };

        // the reference builder must reproduce the case we worked out by hand, otherwise it is
        // not fit to serve as an oracle for everything below
        const Tree reference = reference_tree(keys);
        REQUIRE(reference.nodes.size() == expected.nodes.size());
        {
            INFO("builder: reference");
            check_nodes(reference, expected, expected.nodes.size());
        }

        for (const Builder& builder : builders)
        {
            INFO("builder: " << builder.name);
            check_nodes(actual_tree(builder, keys), expected, expected.nodes.size());
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

    SECTION("deep prefix chains")
    {
        // Keys sharing all but the last few bits, so prefix lengths crowd up against 32 and
        // most increments are 1. The increment is the value most likely to be wrong at the
        // extremes.
        for (uint32_t base = 0; base < 8; ++base)
        {
            Keys keys;
            const uint32_t center = base * 0x04000000u;
            for (uint32_t i = 0; i < 24; ++i)
                keys.push_back(center | i);
            check_tree(sanitize(keys));
        }
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
        // of the nodes must come out the same as the whole tree built at once.
        //
        // Sweeping every prefix length is also what covers a vectorized implementation's
        // remainder handling, since each length leaves a different number of nodes over after
        // the last full batch of lanes.
        std::mt19937 rng(1234);
        std::uniform_int_distribution<uint32_t> key_dist(0, (1u << 30) - 1);

        Keys keys(64);
        for (uint32_t& key : keys)
            key = key_dist(rng);
        keys = sanitize(keys);

        const Tree expected = reference_tree(keys);
        for (const Builder& builder : builders)
        {
            if (builder.supports_partial_build)
            {
                INFO("builder: " << builder.name);
                for (size_t count = 1; count < keys.size(); ++count)
                {
                    INFO("first " << count << " nodes");
                    check_nodes(actual_tree(builder, keys, count), expected, count);
                }
            }
        }
    }
}
