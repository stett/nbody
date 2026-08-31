#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include "nbody/vector.h"
#include "detail/radix.h"
#include "detail/octree.h"
#include "detail/morton.h"

TEST_CASE("create quadtree (flat octree)", "[octree]")
{
    using std::vector;
    using std::ranges::sort;
    using std::ranges::transform;
    using std::exclusive_scan;
    using namespace nbody;
    using namespace nbody::detail;

    // morton codes, sorted, manually generated from a set of 2d coordinates:
    // 0. (1, 5) -> (001, 101) -> 010011
    // 1. (4, 3) -> (100, 011) -> 100101
    // 2. (5, 3) -> (101, 011) -> 100111
    //
    //   |
    // 5 |   0
    // 4 |
    // 3 |            1  2
    // 2 |
    // 1 |
    // 0 |
    //   +------------------
    //    0  1  2  3  4  5
    //
    // two bits per level for a quadtree, and three levels of them: the codes below are written
    // as six significant bits, which Morton left-aligns into the word
    /*using MortonT = detail::Morton<uint32_t, 2, 6>;
    const vector<MortonT> keys{
        0b010011,
        0b100101,
        0b100111,
    };
    */
    using MortonT = detail::Morton<uint32_t, 2, 4>;
    const vector<MortonT> keys{
        0b0110,
        0b1001,
        0b1011,
    };

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

    // count the number of octree nodes, allocate.
    // make sure it matches the output we expect
    const int32_t num_octree_nodes = 1 + offsets.back() + node_count_totals.back();
    vector<OctreeNode> octree_nodes(num_octree_nodes);
    REQUIRE(num_octree_nodes == 5);

    // build the octree
    scalar::build_octree<MortonT>(keys, radix_nodes, radix_parents, node_counts, node_count_totals, offsets, octree_nodes);

	// Here's what we expect:
	/*
               ~0.
              /   \
            ~1.   ~2.
            /    /   \
          (0)  ~3.   ~4.
               /     /
			 (1)   (2)

	nodes:
		0: parent = 0, next = 0, child = -1   [root]
		1: parent = 0, next = 2, child = 0    [leaf]
		2: parent = 0, next = 0, child = -3   [intermediate]
		3: parent = 2, next = 4, child = 1    [leaf]
		4: parent = 2, next = 0, child = 2    [leaf]
	*/

    REQUIRE(octree_nodes[0].parent == 0);
    REQUIRE(octree_nodes[1].parent == 0);
    REQUIRE(octree_nodes[2].parent == 0);
    REQUIRE(octree_nodes[3].parent == 2);
    REQUIRE(octree_nodes[4].parent == 2);

	REQUIRE(octree_nodes[0].next == 0);
	REQUIRE(octree_nodes[1].next == 2);
	REQUIRE(octree_nodes[2].next == 0);
	REQUIRE(octree_nodes[3].next == 4);
	REQUIRE(octree_nodes[4].next == 0);

	REQUIRE(octree_nodes[0].child == 1);
	REQUIRE(octree_nodes[1].child == 0);
	REQUIRE(octree_nodes[2].child == 3);
	REQUIRE(octree_nodes[3].child == 1);
	REQUIRE(octree_nodes[4].child == 2);

	REQUIRE(octree_nodes[0].is_leaf == 0);
	REQUIRE(octree_nodes[1].is_leaf == 1);
	REQUIRE(octree_nodes[2].is_leaf == 0);
	REQUIRE(octree_nodes[3].is_leaf == 1);
	REQUIRE(octree_nodes[4].is_leaf == 1);
}
