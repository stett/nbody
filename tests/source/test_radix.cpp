#include <algorithm>
#include <bit>
#include <cstdlib>
#include <random>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "detail/morton.h"
#include "detail/radix.h"

namespace
{
    using std::vector;
    using std::span;
    using std::countl_zero;
    using namespace nbody::detail;

    using Node = RadixNode;
    using Nodes = vector<Node>;
    using NodeCounts = vector<NodeCount>;

    // Cases write their keys as raw interleaved patterns rather than deriving them from
    // positions, since the tree only ever reads the bits. Morton's constructor left-aligns what
    // it is handed by its own padding, and that convention is what makes cpl / modulus a level.
    using Bits = uint32_t;
    using Keys = vector<Bits>;

    // A node's prefix increment used to be RadixNode::depth_delta, then briefly a span of its
    // own; it now arrives as NodeCount::internals, the count of octree levels the node resolves,
    // alongside NodeCount::leafs. That span runs parallel to the nodes, so a built tree is the
    // two of them together and every case has to check both halves.
    struct Tree
    {
        Nodes nodes;
        NodeCounts node_counts;
    };

    // Fillers for what no builder has written yet. Every field is checked, so these have to be
    // values no correct build can produce.
    constexpr Node unwritten{ INT32_MIN, INT32_MIN };
    constexpr int32_t unwritten_parent = INT32_MIN;
    constexpr NodeCount unwritten_node_count{ INT32_MIN, INT32_MIN };
    constexpr int32_t unwritten_range_end = INT32_MIN;

    // The bits a key may set: everything Morton's constructor will not shift off the top. Two
    // patterns that differ only above this are the same key, so the mask has to be applied
    // before the uniqueness pass rather than after it.
    template <typename MortonT>
    constexpr Bits key_mask = static_cast<Bits>(~Bits{ 0 } >> MortonT::padding);

    // The level a prefix length lands in: whole groups of `modulus` bits counted down from the
    // top of the word. An out of range neighbour reports -1, which is the domain root, level 0 --
    // the same clamp radix_tree applies, and the reason the increments no longer carry the extra
    // step they used to when the root measured from -1.
    template <typename MortonT>
    int32_t level(const int32_t cpl)
    {
        return std::max(cpl, 0) / static_cast<int32_t>(MortonT::modulus);
    }

    template <typename MortonT>
    int32_t key_cpl(const MortonT& a, const MortonT& b)
    {
        return static_cast<int32_t>(countl_zero(static_cast<Bits>(a.bits() ^ b.bits())));
    }

    // Compared field by field rather than through an operator==, which would have to live in
    // nbody::detail to be found by ADL from inside std::equal.
    bool same(const Node& a, const Node& b)
    {
        return a.child0_index == b.child0_index &&
            a.child1_index == b.child1_index;
    }

    bool same(const NodeCount& a, const NodeCount& b)
    {
        return a.internals == b.internals && a.leafs == b.leafs;
    }

    std::string describe(const Node& node, const NodeCount& count)
    {
        return "{ internals " + std::to_string(count.internals)
            + ", leafs " + std::to_string(count.leafs)
            + ", left " + std::to_string(node.child0_index)
            + ", right " + std::to_string(node.child1_index) + " }";
    }

    // Every implementation of the construction must produce the same tree, so each one is held
    // to the same oracle rather than to the others. Naming them here means a new implementation
    // is covered by every case below the moment it is added to this list, and a failure reports
    // which one diverged.
    template <typename MortonT>
    struct Builder
    {
        const char* name;
        void (*build)(span<const MortonT>, span<Node>, span<int32_t>, span<NodeCount>, span<int32_t>, int32_t);
        bool supports_partial_build;
    };

    template <typename MortonT>
    constexpr Builder<MortonT> builders[]{
        { "scalar",             &scalar::radix_tree<MortonT>, true },
        //{ "simd",             &simd::radix_tree<MortonT>  , true },
        //{ "parallel-scalar",    &parallel::radix_tree<MortonT>, false },
    };

    template <typename MortonT>
    vector<MortonT> morton_keys(const Keys& bits)
    {
        vector<MortonT> keys;
        keys.reserve(bits.size());
        for (const Bits pattern : bits)
            keys.push_back(MortonT(pattern));
        return keys;
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
    // The prefix increment comes for free here. radix_tree() has to recover the parent's prefix
    // length from its neighbours (as `dmin`), but a top-down recursion already holds it, so the
    // two arrive at the same number by genuinely different routes.
    template <typename MortonT>
    void reference_tree(const vector<MortonT>& keys, const int32_t a, const int32_t b,
                        const int32_t index, const int32_t parent_prefix, Tree& tree)
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

        // the increment counts level boundaries crossed, so both ends are folded to a level
        // before subtracting -- not after, which would count bits instead. the leaf count falls
        // out of the split: a side of it is a leaf exactly when it is a single key.
        tree.node_counts[index] = {
            level<MortonT>(prefix) - level<MortonT>(parent_prefix),
            (k == a) + (k + 1 == b),
        };

        if (k != a)
            reference_tree(keys, a, k, k, prefix, tree);
        if (k + 1 != b)
            reference_tree(keys, k + 1, b, k + 1, prefix, tree);
    }

    Tree empty_tree(const size_t num_nodes)
    {
        return { Nodes(num_nodes, unwritten), NodeCounts(num_nodes, unwritten_node_count) };
    }

    template <typename MortonT>
    Tree reference_tree(const vector<MortonT>& keys)
    {
        Tree tree = empty_tree(keys.size() - 1);

        // The root has no parent. radix_tree() reads the missing neighbour through cpl(), which
        // reports out of range as -1 and then clamps it to level 0, so the oracle starts from
        // the same place.
        reference_tree(keys, 0, static_cast<int32_t>(keys.size()) - 1, 0, -1, tree);
        return tree;
    }

    template <typename MortonT>
    Tree actual_tree(const Builder<MortonT>& builder, const vector<MortonT>& keys, const size_t num_nodes)
    {
        Tree tree = empty_tree(num_nodes);

        // Parents are written at child indices, and the last leaf's index is one past the last
        // internal node, so this buffer is a key long rather than a node long. Its contents are
        // not asserted on here.
        vector<int32_t> parents(keys.size(), unwritten_parent);

        // The last key of each node's range, which the octree needs for its escape pointers.
        // Not asserted on here either; test_octree.cpp is what holds it to anything.
        vector<int32_t> range_ends(num_nodes, unwritten_range_end);

        builder.build(keys, tree.nodes, parents, tree.node_counts, range_ends, 0);
        return tree;
    }

    std::string describe(const Keys& keys)
    {
        std::string s;
        for (const Bits key : keys)
            s += std::to_string(key) + " ";
        return s;
    }

    // Invariants that hold for any valid radix tree over n distinct keys, checked against the
    // keys themselves rather than against either builder.
    //
    // Walking down from the root reaches all n leaves and all n-1 internal nodes exactly once.
    // The walk also carries each node's key range, which makes the child indices, the leaf flags
    // and the increments all checkable in passing: the range determines where the split must
    // fall, and the prefix length of the range determines what the increments must sum to.
    template <typename MortonT>
    void check_structure(const vector<MortonT>& keys, const Tree& tree)
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

        vector<Frame> stack{ { 0, 0, num_keys - 1, tree.node_counts[0].internals } };
        while (!stack.empty())
        {
            const Frame frame = stack.back();
            stack.pop_back();

            INFO("node " << frame.index << " covering keys [" << frame.first << ", " << frame.last << "]");
            REQUIRE(frame.index >= 0);
            REQUIRE(frame.index < num_nodes);
            REQUIRE(++node_visits[frame.index] == 1);

            const Node& node = tree.nodes[frame.index];

            // A child resolves at least one bit more than its parent, but a bit is only a level
            // once `modulus` of them have accumulated, so an increment is zero exactly when a
            // node splits inside its parent's level. At a modulus of one every node is its own
            // level, so only the root may be zero there: it measures from the domain root at
            // level 0, and its own prefix is empty when the outermost keys differ in the top
            // bit. Node 0 is always the root.
            const int32_t min_delta = (MortonT::modulus == 1 && frame.index != 0) ? 1 : 0;
            REQUIRE(tree.node_counts[frame.index].internals >= min_delta);

            // the increments telescope: summed from the root they reach the level this range's
            // own prefix length falls in
            REQUIRE(frame.delta_sum == level<MortonT>(key_cpl(keys[frame.first], keys[frame.last])));

            // both children name the split, so either index recovers it and they must agree
            const int32_t k = std::abs(node.child0_index);
            REQUIRE(k >= frame.first);
            REQUIRE(k < frame.last);
            REQUIRE(std::abs(node.child1_index) == k + 1);

            // a child is a leaf exactly when its side of the split is a single key, and the sign
            // of the stored index has to say so. index 0 is unambiguous despite -0 == 0, because
            // internal node 0 is always the root and so is never a child.
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
                stack.push_back({ k, frame.first, k, frame.delta_sum + tree.node_counts[k].internals });
            }

            if (right_is_leaf)
            {
                REQUIRE(++leaf_visits[k + 1] == 1);
            }
            else
            {
                REQUIRE(k + 1 < num_nodes);
                stack.push_back({ k + 1, k + 1, frame.last, frame.delta_sum + tree.node_counts[k + 1].internals });
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
                << ": expected " << describe(expected.nodes[i], expected.node_counts[i])
                << " actual " << describe(actual.nodes[i], actual.node_counts[i]));
            REQUIRE(same(actual.nodes[i], expected.nodes[i]));
            REQUIRE(same(actual.node_counts[i], expected.node_counts[i]));
        }
    }

    // Karras' construction requires strictly increasing keys, and the mask has to come first:
    // patterns differing only in the bits Morton shifts away are the same key.
    template <typename MortonT>
    Keys sanitize(Keys keys)
    {
        for (Bits& key : keys)
            key &= key_mask<MortonT>;
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    template <typename MortonT>
    void check_tree(const Keys& raw)
    {
        const Keys bits = sanitize<MortonT>(raw);
        if (bits.size() < 2)
            return;

        INFO("modulus: " << MortonT::modulus);
        INFO("keys: " << describe(bits));

        const vector<MortonT> keys = morton_keys<MortonT>(bits);

        // built once: the oracle does not depend on which implementation is under test
        const Tree expected = reference_tree(keys);

        for (const Builder<MortonT>& builder : builders<MortonT>)
        {
            INFO("builder: " << builder.name);
            const Tree actual = actual_tree(builder, keys, keys.size() - 1);
            check_nodes(actual, expected, expected.nodes.size());
            check_structure(keys, actual);
        }
    }

    // Every case runs at each modulus. The level arithmetic is the only thing the modulus
    // changes, and it is exactly where the radix tree meets the octree, so a case that only ran
    // at one of them would leave that seam untested.
    void check_tree_all_moduli(const Keys& keys)
    {
        check_tree<Morton<uint32_t, 1>>(keys);
        check_tree<Morton<uint32_t, 2>>(keys);
        check_tree<Morton<uint32_t, 3>>(keys);
    }
}

TEST_CASE("radix tree", "[radix]")
{
    SECTION("hand computed, bit level")
    {
        // A modulus of one makes every resolved bit its own level, so the increments are plain
        // prefix length differences and can be read straight off the key bits.
        //
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
        // minus its parent's -- the root measuring from the domain root at level 0, so 27.
        const Tree expected{
            Nodes{
                { -3, -4 },
                {  0,  1 },
                {  2,  3 },
                { -1, -2 },
                {  4,  5 },
            },
            NodeCounts{ { 27, 0 }, { 2, 2 }, { 1, 2 }, { 1, 0 }, { 2, 2 } },
        };

        using MortonT = Morton<uint32_t, 1>;
        const vector<MortonT> morton = morton_keys<MortonT>(keys);

        // the reference builder must reproduce the case we worked out by hand, otherwise it is
        // not fit to serve as an oracle for everything below
        const Tree reference = reference_tree(morton);
        REQUIRE(reference.nodes.size() == expected.nodes.size());
        {
            INFO("builder: reference");
            check_nodes(reference, expected, expected.nodes.size());
        }

        for (const Builder<MortonT>& builder : builders<MortonT>)
        {
            INFO("builder: " << builder.name);
            check_nodes(actual_tree(builder, morton, expected.nodes.size()), expected, expected.nodes.size());
        }
    }

    SECTION("hand computed quadtree")
    {
        // The same case test_octree.cpp builds on: 2d morton codes, three levels of two bits,
        // left-aligned so the level boundaries start at the top of the word.
        //
        //   0. (1,5) -> (001, 101) -> 010011
        //   1. (4,1) -> (100, 001) -> 100101
        //   2. (5,1) -> (101, 001) -> 100111
        constexpr int32_t key_bits = 6;
        const Keys keys{
            0b010011u << (32 - key_bits),
            0b100101u << (32 - key_bits),
            0b100111u << (32 - key_bits),
        };

        // Keys 1 and 2 share the first two levels, so node 1 sits at level 2 and creates both
        // of them; the root splits inside level 1 and so creates nothing beyond the domain root.
        const Tree expected{
            Nodes{
                { 0, -1 },
                { 1,  2 },
            },
            NodeCounts{ { 0, 1 }, { 2, 2 } },
        };

        using MortonT = Morton<uint32_t, 2>;
        const vector<MortonT> morton = morton_keys<MortonT>(keys);

        const Tree reference = reference_tree(morton);
        {
            INFO("builder: reference");
            check_nodes(reference, expected, expected.nodes.size());
        }

        for (const Builder<MortonT>& builder : builders<MortonT>)
        {
            INFO("builder: " << builder.name);
            check_nodes(actual_tree(builder, morton, expected.nodes.size()), expected, expected.nodes.size());
        }
    }

    SECTION("two keys")
    {
        check_tree_all_moduli(Keys{ 0b0, 0b1 });
        check_tree_all_moduli(Keys{ 0x00000000u, 0xFFFFFFFFu });
    }

    SECTION("dense low keys")
    {
        // 0..n-1 for a range of n: maximally deep left leaning chains
        for (int32_t num_keys = 2; num_keys <= 40; ++num_keys)
        {
            Keys keys(num_keys);
            for (int32_t i = 0; i < num_keys; ++i)
                keys[i] = static_cast<Bits>(i);
            check_tree_all_moduli(keys);
        }
    }

    SECTION("single bit keys")
    {
        // every key differs from its neighbors at a different bit depth
        Keys keys{ 0 };
        for (int32_t bit = 0; bit < 30; ++bit)
            keys.push_back(1u << bit);
        check_tree_all_moduli(keys);
    }

    SECTION("wide split")
    {
        // two tight clusters separated at the top bit: the root split is far from both ends
        Keys keys;
        for (uint32_t i = 0; i < 16; ++i)
            keys.push_back(i);
        for (uint32_t i = 0; i < 16; ++i)
            keys.push_back(0x20000000u | i);
        check_tree_all_moduli(keys);
    }

    SECTION("lopsided split")
    {
        // one key alone on the far side of the root split, forcing a long unbalanced spine
        Keys keys;
        for (uint32_t i = 0; i < 31; ++i)
            keys.push_back(i);
        keys.push_back(0x3FFFFFFFu);
        check_tree_all_moduli(keys);
    }

    SECTION("deep prefix chains")
    {
        // Keys sharing all but the last few bits, so prefix lengths crowd up against 32 and most
        // increments are 0 or 1. The increment is the value most likely to be wrong at the
        // extremes.
        for (uint32_t base = 0; base < 8; ++base)
        {
            Keys keys;
            const uint32_t center = base * 0x04000000u;
            for (uint32_t i = 0; i < 24; ++i)
                keys.push_back(center | i);
            check_tree_all_moduli(keys);
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
            for (Bits& key : keys)
                key = key_dist(rng);

            check_tree_all_moduli(keys);
        }
    }

    SECTION("random clustered keys")
    {
        // keys drawn from a few tight clusters, imitating the morton codes of a galaxy: long
        // shared prefixes, so deep chains of internal nodes with single leaf children
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

            check_tree_all_moduli(keys);
        }
    }

    SECTION("partial node range")
    {
        // radix_tree() is documented as buildable over a section of the node array; a prefix of
        // the nodes must come out the same as the whole tree built at once.
        //
        // Sweeping every prefix length is also what covers a vectorized implementation's
        // remainder handling, since each length leaves a different number of nodes over after
        // the last full batch of lanes.
        using MortonT = Morton<uint32_t, 3>;

        std::mt19937 rng(1234);
        std::uniform_int_distribution<uint32_t> key_dist(0, (1u << 30) - 1);

        Keys keys(64);
        for (Bits& key : keys)
            key = key_dist(rng);
        keys = sanitize<MortonT>(keys);

        const vector<MortonT> morton = morton_keys<MortonT>(keys);
        const Tree expected = reference_tree(morton);
        for (const Builder<MortonT>& builder : builders<MortonT>)
        {
            if (builder.supports_partial_build)
            {
                INFO("builder: " << builder.name);
                for (size_t count = 1; count < keys.size(); ++count)
                {
                    INFO("first " << count << " nodes");
                    check_nodes(actual_tree(builder, morton, count), expected, count);
                }
            }
        }
    }
}
