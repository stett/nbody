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
	/*

	| 7 |   |   |   |   |   |   |   |   |
	| 6 | A |   |   |   | C |   |   |   |
	| 5 |   |   |   |   | D |   |   |   |
	| 4 |   |   |   |   |   |   |   |   |
	| 3 |   | B |   |   |   |   |   |   |
	| 2 |   |   |   |   |   |   |   |   |
	| 1 |   |   |   |   |   | E |   |   |
	| 0 |   |   |   |   |   |   | F |   |
	|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

	*/

	// Build a sorted list of morton keys for the points in the graph above
	using MortonT = detail::Morton<uint32_t, 2, 6>;
	const vector<MortonT> keys{
		0b000111, // B
		0b010100, // A
		0b100011, // E
		0b101000, // F
		0b110001, // D
		0b110100  // C
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

TEST_CASE("create 3-element quadtree sharing a level 1 quadrant (flat octree)", "[octree]")
{
	/*

	The simplest set of points the construction gets wrong. All three sit inside one level 1
	quadrant, so the whole tree hangs off a single child of the root:

	| **7** |       |       |       |       |       |       |       |       |
	| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
	| **6** |       |       |       |       |       |       |       |       |
	| **5** |       |       |       |       |       |       |       |       |
	| **4** |       |       |       |       |       |       |       |       |
	| **3** |       | (C)   |       |       |       |       |       |       |
	| **2** | (B)   |       |       |       |       |       |       |       |
	| **1** |       |       |       |       |       |       |       |       |
	| **0** | (A)   |       |       |       |       |       |       |       |
	|       | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** |

	Letters run in morton order, so a leaf's child index is also its letter: A is key 0, B is
	key 1, C is key 2.

	A quadtree code is `x2 y2 | x1 y1 | x0 y0` -- one bit per axis per level, x in the high bit
	of each pair:

		A (0,0) -> (000, 000) -> 00 00 00
		B (0,2) -> (000, 010) -> 00 01 00
		C (1,3) -> (001, 011) -> 00 01 11

	Read down the levels: all three share level 1 (quadrant 00), A splits away from B and C at
	level 2, and B and C split from each other at level 3.

	Sharing level 1 is what no case above exercises. In both of those the outermost keys differ
	in their first level, so radix node 0 resolves no level of its own and contributes nothing
	but the root. Here it resolves level 1, and produces an octree node for it.

	The radix tree over these keys, with the counts it reports:

		radix 0: children (A, radix 1)   prefix level 1, internals = 1, leafs = 1
		radix 1: children (B, C)         prefix level 2, internals = 1, leafs = 2

	So radix 0 is allotted the block oct[1..2] and radix 1 the block oct[3..5], and the octree
	those describe is:

	                   ~0.            level 0, the whole domain
	                    |
	                   ~1.            level 1, quadrant 00
	                  /   \
	                (A)   ~3.         level 2: A alone, and quadrant 01 holding B and C
	                     /   \
	                   (B)   (C)      level 3

	nodes:
		0: parent = 0, next = 0, child = 1   [root]
		1: parent = 0, next = 0, child = 2   [internal, level 1]
		2: parent = 1, next = 3, child = 0   [leaf A]
		3: parent = 1, next = 0, child = 4   [internal, level 2]
		4: parent = 3, next = 5, child = 1   [leaf B]
		5: parent = 3, next = 0, child = 2   [leaf C]

	Those block allotments are already the DFS preorder of that tree, so the array *layout* is
	not what this case is about -- and the construction still gets two fields wrong:

		node 1's `next` comes out 3 rather than 0. `next` is written as `i_node_end`, one past
		the block radix 0 produced, on the assumption that whatever follows the block also
		follows the subtree. Here node 3 is *inside* node 1's subtree: it is node 1's second
		child. A block ends where its subtree ends only for a radix node with no internal
		children, which is every radix node in the cases above.

		node 3's `parent` comes out 0 rather than 1. find_octree_parent() walks radix parents
		up to radix node 0 and then returns octree node 0, treating the radix root as having
		produced nothing. That holds while its `internals` is 0, as in the cases above, but
		here it produced node 1.

	*/

	using MortonT = detail::Morton<uint32_t, 2, 6>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000100, // B
		0b000111  // C
	};

	// build the octree
	vector<OctreeNode> octree_nodes = scalar::build_octree<MortonT>(keys);
	REQUIRE(octree_nodes.size() == 6);

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 1, .next = 3, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 1, .next = 0, .child = 4, .is_leaf = 0 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 3, .next = 5, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 3, .next = 0, .child = 2, .is_leaf = 1 });
}
