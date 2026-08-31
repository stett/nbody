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
    const int32_t num_nodes = 1 + offsets.back() + node_count_totals.back();
    vector<OctreeNode> nodes(num_nodes);
    REQUIRE(num_nodes == 5);

    // populate the root node
    nodes[0] = { .parent = 0, .next = 0, .child = 1 };

    const auto find_octree_parent = [&](const int32_t i_radix) -> int32_t
    {
        // find the first parent radix node index which produced a chain of octree nodes
        int32_t i_radix_parent = radix_parents[i_radix];
        while (i_radix_parent > 0 && node_counts[i_radix_parent].internals == 0)
            i_radix_parent = radix_parents[i_radix_parent];

        // find the index of the last oct node in the chain produced by this radix node
        const int32_t i_octree_parent
            = (i_radix_parent > 0)
            ? (offsets[i_radix_parent] + node_counts[i_radix_parent].internals + node_counts[i_radix_parent].leafs + 1)
            : 0;

        // return the parent octree node index
        return i_octree_parent;
    };

    // run through all radix nodes, populating octree nodes from them
    for (int32_t i_radix = 0; i_radix < radix_nodes.size(); ++i_radix)
    {
        // get the first octree node index
        const int32_t i_node_0 = 1 + offsets[i_radix];
        const NodeCount node_count = node_counts[i_radix];
        const int32_t i_node_end = i_node_0 + node_count.internals + node_count.leafs;
        int32_t i_node = i_node_0;

        // populate the first node's parent pointer
        nodes[i_node_0].parent = find_octree_parent(i_radix);

        // populate corresponding intermediate nodes
        for (; i_node < i_node_0 + node_count.internals; ++i_node)
        {
            // store the parent. except for the first node, the parent will always be the previously added node
            if (i_node > i_node_0)
                nodes[i_node].parent = i_node - 1;

            // the "next" pointer will be the next node past the end of our range, or the root if there is none
            nodes[i_node].next = i_node_end < nodes.size() ? i_node_end : 0;

            // the "child" pointer is the pointer to our first child. since we're intermediate, there should
            // always be at least one child coming up next, either the next internal or the first leaf
            nodes[i_node].child = i_node + 1;
        }

        // populate corresponding leaf nodes after the internals
        for (; i_node < i_node_0 + node_count.internals + node_count.leafs; ++i_node)
        {
            // the parent for all the children will be the last intermediate that was added,
            // or i_node_0's parent (which we already found) if there's no intermediate ancestor.
            if (i_node > i_node_0)
                nodes[i_node].parent
                    = node_count.internals > 0
                    ? i_node_0 + node_count.internals - 1
                    : nodes[i_node_0].parent;

            // the "next" pointer will be the next node - it'll either be the next leaf, or one past the end
            // of our range. if this is the last node, we'll give the root. This is exactly the same as
            // with intermediate nodes
            nodes[i_node].next = i_node + 1 < nodes.size() ? i_node + 1 : 0;

            // for leaf nodes, the child index will indicate an index into the original keys array
            // TODO: how do we determine this?
            nodes[i_node].child = -1;
        }
    }

    /*
    // create the octree node array and add the root
    vector<OctreeNode> octree_nodes(octree_node_count);
    {
        OctreeNode& node = octree_nodes[0];
        node.parent = 0; // root's parent is "self"
        node.next = 0; // root's next is "self"
    }
    */

    //
    //for (size_t i_radix = 0; i_radix < radix_nodes.size(); ++i_radix)
    //{
        /*
        // find the location of the first octree node that will be produced by this radix node
        const int32_t count = counts[i_radix];
        if (count == 0)
            continue;

        // find the radix location of the first parent that produced an octree node
        int32_t i_radix_parent = radix_parents[i_radix];
        int32_t count_parent = counts[i_radix_parent];
        while (i_radix_parent >= 0 && counts[i_radix_parent] == 0)
            i_radix_parent = radix_parents[i_radix_parent];
        const int32_t i_octree_parent
            = i_radix_parent >= 0
            ? offsets[i_radix_parent] + counts[i_radix_parent] - 1
            : -1; // what should this bee in this case? is this root?

        // find the octree-location of the first node
        const int32_t i_octree_0 = offsets[i_radix];
        int32_t i_octree = i_octree_0;

        // point the first node's parent to it
        {
            //OctreeNode& node = octree_nodes[i_octree_parent];
            //node.child = i_octree;

            QuadNode& node = nodes[i_octree_parent];
            const int32_t q = quadrant<MortonT>(keys[i_radix_parent], keys[i_radix]);
            node.children[q] = i_octree;
        }

        /*
        // emit the first octree node
        {
            OctreeNode& node = octree_nodes[i_octree];
            node.parent = i_octree_parent;
            //node0.next = ???;
            node.child = ++i_octree;
        }

        // emit the middle octree nodes
        while (i_octree < i_octree_0 + count - 1)
        {
            OctreeNode& node = octree_nodes[i_octree];
            node.parent = i_octree - 1;
            //node.next = ???;
            node.child = ++i_octree;
        }

        // emit the last octree node
        // NOTE: this is done separately so as not to be in a race condition
        // for setting the node's child.
        if (count > 1)
        {
            OctreeNode& node = octree_nodes[i_octree];
            node.parent = i_octree - 1;
            //node.next = ???;
        }
        */
    //}

    // TEMP

    //REQUIRE(octree_nodes[0].level == 1);
    //REQUIRE(octree_nodes[0].parent == 0);
    //REQUIRE(octree_nodes[0].prefix == 0);
}
