#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>
#include "nbody/constants.h"
#include "nbody/vector.h"
#include "detail/physics.h"
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

	// Shared fixture for apply_octree tests below: builds nodes/bounds/masses for a small,
	// explicit set of bodies, using the real production pipeline (morton-encode, sort,
	// build_octree, build_octree_masses) rather than hand-picked node layouts, so these
	// tests exercise apply_octree exactly as cpu_morton_barnes_hut.h does.
	//
	// Positions and masses are re-sorted together by key here, unlike
	// cpu_morton_barnes_hut.h, which sorts _keys alone and then indexes its *unsorted*
	// _body_positions/_body_masses by the sorted position -- a separate, real bug (the
	// mapping from a sorted key back to the body that produced it is lost the moment the
	// keys are sorted on their own). Keeping that bug out of this fixture is deliberate:
	// it isolates apply_octree so a failure here points at apply_octree itself, not at
	// that upstream mismatch. Worth fixing separately in cpu_morton_barnes_hut.h.
	using ApplyMorton = Morton<uint64_t, 3>;

	struct AppliedOctree
	{
		vector<OctreeNode> nodes;
		vector<OctreeBounds<3>> bounds;
		vector<OctreeNodeMass> masses;
	};

	AppliedOctree build_applied_octree(const vector<Vector>& positions, const vector<float>& masses, const float size)
	{
		struct Entry { ApplyMorton key; Vector pos; float mass; };

		const float size_inv = 1.f / size;
		vector<Entry> entries(positions.size());
		for (size_t i = 0; i < positions.size(); ++i)
		{
			const Vector& p = positions[i];
			entries[i] = {
				ApplyMorton(
					std::clamp((p.x * size_inv) + .5f, 0.f, 1.f),
					std::clamp((p.y * size_inv) + .5f, 0.f, 1.f),
					std::clamp((p.z * size_inv) + .5f, 0.f, 1.f)),
				p,
				masses[i],
			};
		}
		std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });

		vector<ApplyMorton> keys(entries.size());
		vector<Vector> sorted_positions(entries.size());
		vector<float> sorted_masses(entries.size());
		for (size_t i = 0; i < entries.size(); ++i)
		{
			keys[i] = entries[i].key;
			sorted_positions[i] = entries[i].pos;
			sorted_masses[i] = entries[i].mass;
		}

		OctreeCache cache;
		AppliedOctree result;
		scalar::build_octree<ApplyMorton>(keys, cache, result.nodes, result.bounds);

		result.masses.resize(result.nodes.size());
		vector<std::atomic<uint8_t>> counters(result.nodes.size());
		scalar::build_octree_masses(result.nodes, cache.leaf_nodes, sorted_positions, sorted_masses, result.masses, counters);

		return result;
	}
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
    using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
    const vector<MortonT> keys{
        0b0110,
        0b1001,
        0b1011,
    };

    // build the octree
	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 5);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

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

	// Bounds. Cells are the unit cube halved once per level, so a node at level L is 2^-L
	// across and its center is read straight out of any key it holds. Level 0 is the whole
	// domain, which is why the root is centered on (0.5, 0.5) rather than on the origin.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.75f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.625f, 0.375f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.875f, 0.375f }, .half_extent = 0.125f });
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
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000111, // B
		0b010100, // A
		0b100011, // E
		0b101000, // F
		0b110001, // D
		0b110100  // C
	};

	// build the octree
	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 9);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 2, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 0, .next = 3, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 0, .next = 6, .child = 4, .is_leaf = 0 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 3, .next = 5, .child = 2, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 3, .next = 6, .child = 3, .is_leaf = 1 });
	REQUIRE(octree_nodes[6] == OctreeNode{ .parent = 0, .next = 0, .child = 7, .is_leaf = 0 });
	REQUIRE(octree_nodes[7] == OctreeNode{ .parent = 6, .next = 8, .child = 4, .is_leaf = 1 });
	REQUIRE(octree_nodes[8] == OctreeNode{ .parent = 6, .next = 0, .child = 5, .is_leaf = 1 });

	// Bounds. The root's four children are the four level 1 quadrants that hold anything.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.25f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.75f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.625f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.875f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[6] == OctreeBoundsT{ .center = { 0.75f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[7] == OctreeBoundsT{ .center = { 0.625f, 0.625f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[8] == OctreeBoundsT{ .center = { 0.625f, 0.875f }, .half_extent = 0.125f });
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
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000100, // B
		0b000111  // C
	};

	// build the octree
	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 6);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 1, .next = 3, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 1, .next = 0, .child = 4, .is_leaf = 0 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 3, .next = 5, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 3, .next = 0, .child = 2, .is_leaf = 1 });

	// Bounds. Each level down halves the cell, so the chain reads 0.5 -> 0.25 -> 0.125.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.125f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.125f, 0.375f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.0625f, 0.3125f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.1875f, 0.4375f }, .half_extent = 0.0625f });
}

TEST_CASE("create 2-element quadtree at the deepest level (flat octree)", "[octree]")
{
	/*

	The smallest input the construction accepts, and at the same time the deepest chain this
	morton type can produce: two adjacent points, so they share every level but the last.

	| **7** |       |       |       |       |       |       |       |       |
	| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
	| **6** |       |       |       |       |       |       |       |       |
	| **5** |       |       |       |       |       |       |       |       |
	| **4** |       |       |       |       |       |       |       |       |
	| **3** |       |       |       |       |       |       |       |       |
	| **2** |       |       |       |       |       |       |       |       |
	| **1** | (B)   |       |       |       |       |       |       |       |
	| **0** | (A)   |       |       |       |       |       |       |       |
	|       | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** |

		A (0,0) -> (000, 000) -> 00 00 00
		B (0,1) -> (000, 001) -> 00 00 01

	Two keys leave exactly one radix node, which resolves levels 1 and 2 and then splits into
	two leafs at level 3:

		radix 0: children (A, B)   internals = 2, leafs = 2, range end = 1

	It is the only case here whose chain runs more than one node deep, so it is the only one
	that exercises a chain node's parent being the previous slot and its child being the next
	one. Nothing else in the file distinguishes those from the looked-up forms.

	           ~0.        level 0
	            |
	           ~1.        level 1
	            |
	           ~2.        level 2
	           /  \
	         (A)  (B)     level 3

	*/

	using MortonT = detail::Morton<uint32_t, 2, 6>;
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000001  // B
	};

	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 5);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 1, .next = 0, .child = 3, .is_leaf = 0 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 2, .next = 4, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 2, .next = 0, .child = 1, .is_leaf = 1 });

	// Bounds. Two levels of chain, then the leafs at 2^-3. Only y separates them.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.125f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.0625f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.0625f, 0.1875f }, .half_extent = 0.0625f });
}

TEST_CASE("create 3-element quadtree whose nodes are not in traversal order (flat octree)", "[octree]")
{
	/*

	Three points in one level 2 cell, arranged so that the radix node holding them has an
	internal child *and* a leaf child -- the one shape whose block cannot be laid out in
	traversal order.

	| **7** |       |       |       |       |       |       |       |       |
	| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
	| **6** |       |       |       |       |       |       |       |       |
	| **5** |       |       |       |       |       |       |       |       |
	| **4** |       |       |       |       |       |       |       |       |
	| **3** |       |       |       |       |       |       |       |       |
	| **2** |       |       |       |       |       |       |       |       |
	| **1** | (B)   |       |       |       |       |       |       |       |
	| **0** | (A)   | (C)   |       |       |       |       |       |       |
	|       | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** |

		A (0,0) -> (000, 000) -> 00 00 00
		B (0,1) -> (000, 001) -> 00 00 01
		C (1,0) -> (001, 000) -> 00 00 10

	All three share levels 1 and 2 and separate at level 3, so the tree is a two node chain
	above a single node with three leaf children, in quadrant order A (00), B (01), C (10).

		radix 0: children (radix 1, C)   internals = 2, leafs = 1, range end = 2
		radix 1: children (A, B)         internals = 0, leafs = 2, range end = 1

	Radix 0's own leaf is C, and its block is [internal, internal, C] = oct[1..3]. But C is the
	*last* of the three children in quadrant order, and A and B live in radix 1's block at
	oct[4..5], after it. So the array cannot be in traversal order here no matter how the blocks
	are placed: a radix node's leafs always follow its internals, while its internal child's
	subtree is a separate block.

	That makes this the case that pins the pointers down as pointers. The sibling chain runs
	oct 4 -> oct 5 -> oct 3, backwards through the array, and node 2's child is 4 rather than
	the slot next to it.

	              ~0.               level 0
	               |
	              ~1.               level 1
	               |
	              ~2.               level 2
	             /  |  \
	          (A) (B) (C)           level 3, held at oct 4, oct 5 and oct 3

	*/

	using MortonT = detail::Morton<uint32_t, 2, 6>;
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000001, // B
		0b000010  // C
	};

	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 6);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });

	// child is 4, in radix 1's block, not the adjacent slot
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 1, .next = 0, .child = 4, .is_leaf = 0 });

	// C, reached last in the sibling chain despite sitting first in the array
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 2, .next = 0, .child = 2, .is_leaf = 1 });

	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 2, .next = 5, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 2, .next = 3, .child = 1, .is_leaf = 1 });

	// Bounds. The three leafs share a level 2 cell and split at level 3, so they differ by one eighth.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.125f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.1875f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.0625f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.0625f, 0.1875f }, .half_extent = 0.0625f });
}

TEST_CASE("create 4-element quadtree with a childless block tail (flat octree)", "[octree]")
{
	/*

	Two pairs of points in adjacent level 2 cells, which gives a radix node that resolves a
	level and owns no leaf at all -- both of its children are internal.

	| **7** |       |       |       |       |       |       |       |       |
	| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
	| **6** |       |       |       |       |       |       |       |       |
	| **5** |       |       |       |       |       |       |       |       |
	| **4** |       |       |       |       |       |       |       |       |
	| **3** |       |       |       |       |       |       |       |       |
	| **2** | (C)   | (D)   |       |       |       |       |       |       |
	| **1** |       |       |       |       |       |       |       |       |
	| **0** | (A)   | (B)   |       |       |       |       |       |       |
	|       | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** |

		A (0,0) -> (000, 000) -> 00 00 00
		B (1,0) -> (001, 000) -> 00 00 10
		C (0,2) -> (000, 010) -> 00 01 00
		D (1,2) -> (001, 010) -> 00 01 10

	All four share level 1, then split into level 2 cells 00 (A, B) and 01 (C, D):

		radix 0: children (radix 1, radix 2)   internals = 1, leafs = 0, range end = 3
		radix 1: children (A, B)               internals = 1, leafs = 2, range end = 1
		radix 2: children (C, D)               internals = 1, leafs = 2, range end = 3

	Radix 0's block is one node long and holds no leaf, so the node ending it has to reach
	outside its own block for a child. That is what separates `child` from `next` here: node 1's
	child is 2, the head of radix 1's block, while its escape is 0 -- nothing follows it. A block
	ends where its subtree ends only for a radix node with no internal children, and this node
	has two.

	              ~0.                 level 0
	               |
	              ~1.                 level 1, holds no leaf of its own
	             /   \
	          ~2.     ~5.             level 2, cells 00 and 01
	         /  \     /  \
	      (A) (B)   (C) (D)           level 3

	*/

	using MortonT = detail::Morton<uint32_t, 2, 6>;
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000010, // B
		0b000100, // C
		0b000110  // D
	};

	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 8);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });

	// child reaches into the next block; next is 0. Deriving both from the block's end would
	// make them the same number.
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });

	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 1, .next = 5, .child = 3, .is_leaf = 0 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 2, .next = 4, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 2, .next = 5, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 1, .next = 0, .child = 6, .is_leaf = 0 });
	REQUIRE(octree_nodes[6] == OctreeNode{ .parent = 5, .next = 7, .child = 2, .is_leaf = 1 });
	REQUIRE(octree_nodes[7] == OctreeNode{ .parent = 5, .next = 0, .child = 3, .is_leaf = 1 });

	// Bounds. The two level 2 cells are side by side in y, a quarter apart, each an eighth across.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.125f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.0625f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.1875f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.125f, 0.375f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[6] == OctreeBoundsT{ .center = { 0.0625f, 0.3125f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[7] == OctreeBoundsT{ .center = { 0.1875f, 0.3125f }, .half_extent = 0.0625f });
}

TEST_CASE("create 5-element quadtree whose last radix node produces nothing (flat octree)", "[octree]")
{
	/*

	A radix node can resolve no level of its own and own no leaf, in which case it produces no
	octree node and its block is empty. When that node is the *last* one, 1 + its offset is one
	past the end of the array, so a write placed at the head of a block lands outside it.

	Five points is the smallest input that puts such a node last: no 3 or 4 key set over this
	grid does it. The construction no longer writes anything for such a node, so no assertion
	here can catch that write coming back -- what this case is really for is the tree it
	produces, which no other case in this file has: a root with three children whose array
	order runs backwards.

	| **7** |       |       |       |       |       |       |       |       |
	| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
	| **6** |       |       |       |       |       |       |       |       |
	| **5** | (D)   |       |       |       |       |       |       |       |
	| **4** | (C)   |       |       |       |       |       |       |       |
	| **3** |       |       |       |       |       |       |       |       |
	| **2** |       |       |       |       |       |       |       |       |
	| **1** |       | (B)   |       |       |       |       |       |       |
	| **0** | (A)   |       |       |       | (E)   |       |       |       |
	|       | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** |

		A (0,0) -> (000, 000) -> 00 00 00
		B (1,1) -> (001, 001) -> 00 00 11
		C (0,4) -> (000, 100) -> 01 00 00
		D (1,5) -> (001, 101) -> 01 00 11
		E (4,0) -> (100, 000) -> 10 00 00

	Level 1 puts A and B in cell 00, C and D in cell 01, and E alone in cell 10, so the root has
	three children. Each pair then shares level 2 and separates at level 3.

		radix 0: children (radix 3, E)         internals = 0, leafs = 1, range end = 4
		radix 1: children (A, B)               internals = 2, leafs = 2, range end = 1
		radix 2: children (C, D)               internals = 2, leafs = 2, range end = 3
		radix 3: children (radix 1, radix 2)   internals = 0, leafs = 0, range end = 3

	Radix 3 is last and produces nothing. Its offset is 9 and the array holds 10 nodes, so its
	block would begin at oct 10 -- one past the end.

	The case is worth keeping for what it asserts as well: E's leaf sits at oct 1, ahead of the
	subtrees of every key that sorts before it, so the root's children run oct 2 -> oct 6 ->
	oct 1 in quadrant order.

	                    ~0.                     level 0
	                  /  |  \
	              ~2.   ~6.  (E)                 level 1, cells 00, 01 and 10
	               |     |
	              ~3.   ~7.                      level 2
	             /  \   /  \
	          (A) (B) (C) (D)                    level 3

	*/

	using MortonT = detail::Morton<uint32_t, 2, 6>;
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000000, // A
		0b000011, // B
		0b010000, // C
		0b010011, // D
		0b100000  // E
	};

	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 10);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	// the root's first child is oct 2, not oct 1: radix 0 resolves no level, so the first node
	// of the range is found by descending past radix 3, which resolves none either
	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 2, .is_leaf = 0 });

	// E, the root's last child, held ahead of both subtrees
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 0, .child = 4, .is_leaf = 1 });

	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 0, .next = 6, .child = 3, .is_leaf = 0 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 2, .next = 6, .child = 4, .is_leaf = 0 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 3, .next = 5, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 3, .next = 6, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[6] == OctreeNode{ .parent = 0, .next = 1, .child = 7, .is_leaf = 0 });
	REQUIRE(octree_nodes[7] == OctreeNode{ .parent = 6, .next = 1, .child = 8, .is_leaf = 0 });
	REQUIRE(octree_nodes[8] == OctreeNode{ .parent = 7, .next = 9, .child = 2, .is_leaf = 1 });
	REQUIRE(octree_nodes[9] == OctreeNode{ .parent = 7, .next = 1, .child = 3, .is_leaf = 1 });

	// Bounds. E's cell is a level 1 quadrant, the other two children are level 1 quadrants subdivided twice.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.75f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.125f, 0.125f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.0625f, 0.0625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.1875f, 0.1875f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[6] == OctreeBoundsT{ .center = { 0.25f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[7] == OctreeBoundsT{ .center = { 0.125f, 0.625f }, .half_extent = 0.125f });
	REQUIRE(octree_bounds[8] == OctreeBoundsT{ .center = { 0.0625f, 0.5625f }, .half_extent = 0.0625f });
	REQUIRE(octree_bounds[9] == OctreeBoundsT{ .center = { 0.1875f, 0.6875f }, .half_extent = 0.0625f });
}

TEST_CASE("create 8-element octree, one body per octant (flat octree)", "[octree]")
{
	/*

	The first case here at a modulus of three, and the only one that is an octree rather than a
	quadtree. Modulus is what converts a common prefix length into a level, and it also bounds
	how far the construction has to skip past radix nodes that resolve no level, so a file of
	quadtrees leaves both untested at the width the solver actually runs at.

	One level, one body in each of the eight octants. A code is `x y z`, one bit per axis, x
	highest:

		octant | code | position
		-------|------|----------
		   0   | 000  | (0,0,0)
		   1   | 001  | (0,0,1)
		   2   | 010  | (0,1,0)
		   3   | 011  | (0,1,1)
		   4   | 100  | (1,0,0)
		   5   | 101  | (1,0,1)
		   6   | 110  | (1,1,0)
		   7   | 111  | (1,1,1)

	Every pair of keys differs inside level 1, so no radix node resolves a level of its own and
	the octree is one node deep: a root with eight leaf children.

		radix 0: children (radix 3, radix 4)   internals = 0, leafs = 0
		radix 1: children (0, 1)               internals = 0, leafs = 2
		radix 2: children (2, 3)               internals = 0, leafs = 2
		radix 3: children (radix 1, radix 2)   internals = 0, leafs = 0
		radix 4: children (radix 5, radix 6)   internals = 0, leafs = 0
		radix 5: children (4, 5)               internals = 0, leafs = 2
		radix 6: children (6, 7)               internals = 0, leafs = 2

	Three of the seven radix nodes produce nothing, and they are the ones that force the skips
	to run two deep in both directions -- the bound being modulus - 1, which is 1 for a quadtree
	and so never reaches two anywhere else in this file:

		the root descends past radix 3 and then radix 1 to find its first child at oct 1
		leaf 0 walks up past radix 3 and then radix 0 to find the root

	The eight leaves also make the longest sibling chain here, one escape pointer per octant.

	*/

	using MortonT = detail::Morton<uint32_t, 3, 3>;
	using OctreeBoundsT = OctreeBounds<MortonT::modulus>;
	const vector<MortonT> keys{
		0b000, 0b001, 0b010, 0b011,
		0b100, 0b101, 0b110, 0b111
	};

	vector<OctreeNode> octree_nodes;
	vector<OctreeBoundsT> octree_bounds;
	scalar::build_octree<MortonT>(keys, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 9);
	REQUIRE(octree_bounds.size() == octree_nodes.size());

	REQUIRE(octree_nodes[0] == OctreeNode{ .parent = 0, .next = 0, .child = 1, .is_leaf = 0 });
	REQUIRE(octree_nodes[1] == OctreeNode{ .parent = 0, .next = 2, .child = 0, .is_leaf = 1 });
	REQUIRE(octree_nodes[2] == OctreeNode{ .parent = 0, .next = 3, .child = 1, .is_leaf = 1 });
	REQUIRE(octree_nodes[3] == OctreeNode{ .parent = 0, .next = 4, .child = 2, .is_leaf = 1 });
	REQUIRE(octree_nodes[4] == OctreeNode{ .parent = 0, .next = 5, .child = 3, .is_leaf = 1 });
	REQUIRE(octree_nodes[5] == OctreeNode{ .parent = 0, .next = 6, .child = 4, .is_leaf = 1 });
	REQUIRE(octree_nodes[6] == OctreeNode{ .parent = 0, .next = 7, .child = 5, .is_leaf = 1 });
	REQUIRE(octree_nodes[7] == OctreeNode{ .parent = 0, .next = 8, .child = 6, .is_leaf = 1 });
	REQUIRE(octree_nodes[8] == OctreeNode{ .parent = 0, .next = 0, .child = 7, .is_leaf = 1 });

	// Bounds. One body per octant, so every leaf is a corner of the unit cube.

	REQUIRE(octree_bounds[0] == OctreeBoundsT{ .center = { 0.5f, 0.5f, 0.5f }, .half_extent = 0.5f });
	REQUIRE(octree_bounds[1] == OctreeBoundsT{ .center = { 0.25f, 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[2] == OctreeBoundsT{ .center = { 0.25f, 0.25f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[3] == OctreeBoundsT{ .center = { 0.25f, 0.75f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[4] == OctreeBoundsT{ .center = { 0.25f, 0.75f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[5] == OctreeBoundsT{ .center = { 0.75f, 0.25f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[6] == OctreeBoundsT{ .center = { 0.75f, 0.25f, 0.75f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[7] == OctreeBoundsT{ .center = { 0.75f, 0.75f, 0.25f }, .half_extent = 0.25f });
	REQUIRE(octree_bounds[8] == OctreeBoundsT{ .center = { 0.75f, 0.75f, 0.75f }, .half_extent = 0.25f });
}

TEST_CASE("build_octree_masses propagates leaf masses to their parents", "[octree]")
{
	// Reuses the "3-element quadtree" fixture from the top of this file, whose node layout
	// is already pinned down by that test:
	//   node 0: root, parent of 1 and 2
	//   node 1: leaf, key 0 -- a direct child of the root
	//   node 2: internal, parent of 3 and 4
	//   node 3: leaf, key 1
	//   node 4: leaf, key 2
	// so leaf_nodes = [1, 3, 4]: every leaf's octree index is *larger* than its key index.
	// That is exactly what a leaf-mass pass indexing positions/masses by the octree node
	// index rather than by the key gets wrong -- key 1 and key 2 would read past the end of
	// a 3-element positions/masses array.
	using MortonT = detail::Morton<uint32_t, 2, 4>;
	const vector<MortonT> keys{
		0b0110,
		0b1001,
		0b1011,
	};

	OctreeCache cache;
	vector<OctreeNode> octree_nodes;
	vector<OctreeBounds<MortonT::modulus>> octree_bounds;
	scalar::build_octree<MortonT>(keys, cache, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 5);
	REQUIRE(cache.leaf_nodes == vector<int32_t>{ 1, 3, 4 });

	// one body per key, in key order, with masses and positions chosen so each is
	// trivially distinguishable from the others
	const vector<Vector> positions{
		Vector(1, 0, 0),
		Vector(0, 1, 0),
		Vector(0, 0, 1),
	};
	const vector<float> masses{ 1.f, 2.f, 4.f };

	vector<OctreeNodeMass> node_masses(octree_nodes.size());
	vector<std::atomic<uint8_t>> node_counters(octree_nodes.size());
	scalar::build_octree_masses(octree_nodes, cache.leaf_nodes, positions, masses, node_masses, node_counters);

	// every leaf holds its own body's mass and position, unchanged
	REQUIRE(node_masses[1].mass == masses[0]);
	REQUIRE(node_masses[1].center == positions[0]);
	REQUIRE(node_masses[3].mass == masses[1]);
	REQUIRE(node_masses[3].center == positions[1]);
	REQUIRE(node_masses[4].mass == masses[2]);
	REQUIRE(node_masses[4].center == positions[2]);

	// node 2 aggregates leaves 3 and 4, i.e. bodies 1 and 2
	const float expect_2_mass = masses[1] + masses[2];
	const Vector expect_2_center = ((masses[1] * positions[1]) + (masses[2] * positions[2])) / expect_2_mass;
	REQUIRE(node_masses[2].mass == Catch::Approx(expect_2_mass));
	REQUIRE(node_masses[2].center.x == Catch::Approx(expect_2_center.x));
	REQUIRE(node_masses[2].center.y == Catch::Approx(expect_2_center.y));
	REQUIRE(node_masses[2].center.z == Catch::Approx(expect_2_center.z));

	// the root aggregates all three bodies
	const float expect_total_mass = masses[0] + masses[1] + masses[2];
	REQUIRE(node_masses[0].mass == Catch::Approx(expect_total_mass));
}

TEST_CASE("build_octree_masses aggregates many siblings under one root", "[octree]")
{
	// Reuses the "eight octants, one level" fixture above: node k+1 is the leaf for key k,
	// and all eight leaves hang directly off the root. This checks the root's accumulation
	// over many siblings at once, rather than through the two-level chain of the test above.
	using MortonT = detail::Morton<uint32_t, 3, 3>;
	const vector<MortonT> keys{
		0b000, 0b001, 0b010, 0b011,
		0b100, 0b101, 0b110, 0b111
	};

	OctreeCache cache;
	vector<OctreeNode> octree_nodes;
	vector<OctreeBounds<MortonT::modulus>> octree_bounds;
	scalar::build_octree<MortonT>(keys, cache, octree_nodes, octree_bounds);
	REQUIRE(octree_nodes.size() == 9);
	for (int32_t key = 0; key < static_cast<int32_t>(keys.size()); ++key)
		REQUIRE(cache.leaf_nodes[key] == key + 1);

	const vector<Vector> positions{
		Vector(0, 0, 0), Vector(0, 0, 1), Vector(0, 1, 0), Vector(0, 1, 1),
		Vector(1, 0, 0), Vector(1, 0, 1), Vector(1, 1, 0), Vector(1, 1, 1),
	};
	const vector<float> masses{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

	vector<OctreeNodeMass> node_masses(octree_nodes.size());
	vector<std::atomic<uint8_t>> node_counters(octree_nodes.size());
	scalar::build_octree_masses(octree_nodes, cache.leaf_nodes, positions, masses, node_masses, node_counters);

	// every leaf holds its own body's mass and position, unchanged
	for (int32_t key = 0; key < static_cast<int32_t>(keys.size()); ++key)
	{
		const OctreeNodeMass& leaf = node_masses[cache.leaf_nodes[key]];
		REQUIRE(leaf.mass == masses[key]);
		REQUIRE(leaf.center == positions[key]);
	}

	// the root aggregates every body: total mass, and the mass-weighted centroid
	float expect_mass = 0.f;
	Vector expect_center{ 0, 0, 0 };
	for (size_t i = 0; i < masses.size(); ++i)
	{
		expect_center += masses[i] * positions[i];
		expect_mass += masses[i];
	}
	expect_center = expect_center / expect_mass;

	REQUIRE(node_masses[0].mass == Catch::Approx(expect_mass));
	REQUIRE(node_masses[0].center.x == Catch::Approx(expect_center.x));
	REQUIRE(node_masses[0].center.y == Catch::Approx(expect_center.y));
	REQUIRE(node_masses[0].center.z == Catch::Approx(expect_center.z));
}

TEST_CASE("apply_octree on one body applies no force", "[octree][apply_octree]")
{
	// The smallest input apply_octree accepts: one body, so build_octree hands back the
	// degenerate single-node tree (root is its own leaf, see try_build_degenerate_octree).
	// There is nothing else to feel a force from, and the only node apply_octree can visit
	// is the body's own leaf -- gravity() must report that as zero, not NaN, since the
	// self-distance is exactly zero and the default radius is also zero.
	const vector<Vector> positions{ Vector(100, 0, 0) };
	const vector<float> masses{ 5.f };
	const AppliedOctree tree = build_applied_octree(positions, masses, 1000.f);
	REQUIRE(tree.nodes.size() == 1);

	// theta this large forces the loop past every "far enough, approximate" branch it can
	// take, so it always drills down to individual leaves -- see the huge-theta comment on
	// the two- and three-body tests below for why that is exhaustive here rather than the
	// more usual small-theta reading of the opening angle.
	const float theta = 1.0e6f;

	int32_t call_count = 0;
	Vector acc{ 0, 0, 0 };
	scalar::apply_octree(tree.nodes, tree.bounds, tree.masses, positions[0],
		[&](const int32_t node_index)
		{
			++call_count;
			const OctreeNodeMass& node_mass = tree.masses[node_index];
			acc += gravity(positions[0], 0.f, node_mass.center, node_mass.mass, G);
		}, theta, 1000.f);

	REQUIRE(call_count == 1);
	REQUIRE(acc.x == 0.f);
	REQUIRE(acc.y == 0.f);
	REQUIRE(acc.z == 0.f);
}

TEST_CASE("apply_octree on two bodies matches a direct pairwise force", "[octree][apply_octree]")
{
	// Two bodies is the smallest case with an actual force to get right, and the smallest
	// case with a real sibling to traverse to (the "3-element quadtree" and "eight octants"
	// fixtures above are what to reach for if a deeper tree turns out to be where this
	// breaks). With theta this large, apply_octree must drill all the way to both leaves
	// rather than approximate either one, so the result has to equal gravity() applied
	// directly between the two bodies -- any mismatch here is apply_octree's traversal
	// getting the wrong node, the wrong count of nodes, or the wrong mass/center off a node,
	// not an approximation artifact.
	const vector<Vector> positions{ Vector(-100, 0, 0), Vector(100, 0, 0) };
	const vector<float> masses{ 3.f, 7.f };
	const AppliedOctree tree = build_applied_octree(positions, masses, 1000.f);

	const float theta = 1.0e6f;

	int32_t call_count = 0;
	Vector acc{ 0, 0, 0 };
	scalar::apply_octree(tree.nodes, tree.bounds, tree.masses, positions[0],
		[&](const int32_t node_index)
		{
			++call_count;
			const OctreeNodeMass& node_mass = tree.masses[node_index];
			acc += gravity(positions[0], 0.f, node_mass.center, node_mass.mass, G);
		}, theta, 1000.f);

	// one call for the queried body's own (zero-force) leaf, one for the other body
	REQUIRE(call_count == 2);

	const Vector expect_acc = gravity(positions[0], 0.f, positions[1], masses[1], G);
	REQUIRE(acc.x == Catch::Approx(expect_acc.x));
	REQUIRE(acc.y == Catch::Approx(expect_acc.y));
	REQUIRE(acc.z == Catch::Approx(expect_acc.z));
}

TEST_CASE("apply_octree on three bodies matches a brute-force sum", "[octree][apply_octree]")
{
	// Three non-collinear bodies are the smallest case with a real internal node above more
	// than one leaf (unlike the two-body case, whose only internal node is the root), so
	// this is what exercises apply_octree walking a sibling chain more than one hop deep --
	// see the "next" field's role in the fixtures above if this is where it breaks instead
	// of the two-body case above.
	const vector<Vector> positions{ Vector(-100, 0, 0), Vector(100, 0, 0), Vector(0, 100, 0) };
	const vector<float> masses{ 3.f, 7.f, 5.f };
	const AppliedOctree tree = build_applied_octree(positions, masses, 1000.f);

	const float theta = 1.0e6f;

	int32_t call_count = 0;
	Vector acc{ 0, 0, 0 };
	scalar::apply_octree(tree.nodes, tree.bounds, tree.masses, positions[0],
		[&](const int32_t node_index)
		{
			++call_count;
			const OctreeNodeMass& node_mass = tree.masses[node_index];
			acc += gravity(positions[0], 0.f, node_mass.center, node_mass.mass, G);
		}, theta, 1000.f);

	// one call per leaf: the queried body's own (zero-force) leaf, plus the other two
	REQUIRE(call_count == 3);

	const Vector expect_acc = gravity(positions[0], 0.f, positions[1], masses[1], G)
		+ gravity(positions[0], 0.f, positions[2], masses[2], G);
	REQUIRE(acc.x == Catch::Approx(expect_acc.x));
	REQUIRE(acc.y == Catch::Approx(expect_acc.y));
	REQUIRE(acc.z == Catch::Approx(expect_acc.z));
}
