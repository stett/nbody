#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include "nbody/vector.h"
#include "detail/radix.h"
#include "detail/octree.h"
#include "detail/morton.h"

namespace
{
	using std::vector;
	using std::ranges::sort;
	using std::ranges::transform;
	using std::exclusive_scan;
	using namespace nbody;
	using namespace nbody::detail;
}

TEST_CASE("create 3-element quadtree (flat octree)", "[octree]")
{
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
    using MortonT = detail::Morton<uint32_t, 2, 4>;
    const vector<MortonT> keys{
        0b0110,
        0b1001,
        0b1011,
    };

    // build the octree
	vector<OctreeNode> octree_nodes = scalar::build_octree<MortonT>(keys);
	REQUIRE(octree_nodes.size() == 5);

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

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 2, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 0, .next = 0, .child = 3, .is_leaf = 0 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 2, .next = 4, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 2, .next = 0, .child = 2, .is_leaf = 1 });
}

TEST_CASE("create 6-element quadtree (flat octree)", "[octree]")
{
	using MortonT = detail::Morton<uint32_t, 2, 6>;
	const vector<MortonT> keys{
		0b000111,
		0b010100,
		0b100011,
		0b101000,
		0b110001,
		0b110100
	};

	// build the octree
	vector<OctreeNode> octree_nodes = scalar::build_octree<MortonT>(keys);
	REQUIRE(octree_nodes.size() == 9);

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 2, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 0, .next = 3, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 0, .next = 6, .child = 4, .is_leaf = 0 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 3, .next = 5, .child = 2, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 3, .next = 6, .child = 3, .is_leaf = 1 });
	REQUIRE(octree_nodes[6] == OctreeNode{ .parent = 0, .next = 0, .child = 7, .is_leaf = 0 });
	REQUIRE(octree_nodes[7] == OctreeNode{ .parent = 6, .next = 8, .child = 4, .is_leaf = 1 });
	REQUIRE(octree_nodes[8] == OctreeNode{ .parent = 6, .next = 0, .child = 5, .is_leaf = 1 });
}
