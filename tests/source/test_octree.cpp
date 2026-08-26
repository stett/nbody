#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include "nbody/vector.h"
#include "detail/radix.h"
#include "detail/octree.h"
#include "detail/morton.h"

TEST_CASE("create quadtree (flat octree)", "[octree]")
{
    using std::vector;
    using std::ranges::sort;
    using namespace nbody;
    using namespace nbody::detail;

    /*
    const vector<Vector> points {
        { 0, 6, 0 },
        { 1, 3, 0 },
        { 4, 6, 0 },
        { 4, 5, 0 },
        { 5, 1, 0 },
        { 6, 0, 0 },
    };
    vector<Vector> points_norm(points.size());

    // normalize the points on a specific bound
    const float bounds_size = 10.f;
    const float bounds_size_inv = 1.f / bounds_size;
    std::transform(points.begin(), points.end(), points_norm.begin(), [s = bounds_size_inv](const Vector& v) { return v * s; });
    */

    /*
    // compute and sort morton codes
    // tested elsewhere, so we don't make assertions here.
    vector<uint32_t> keys(points.size());
    scalar::morton(points_norm, keys);
    sort(keys);
    */

    // morton codes, generated from the 2d coordinates:
    // 0. (1,5) -> (001, 101) -> 010011
    // 1.
    // 2.
    const vector<uint32_t> keys{
        0b010011,
        0b100101,
        0b100111,
    };

    // build the radix tree
    vector<RadixNode> radix_nodes(keys.size() - 1);
    vector<int32_t> radix_node_parents(radix_nodes.size());
    vector<int32_t> radix_node_cpl_deltas(radix_nodes.size());
    scalar::radix_tree(keys, radix_nodes, radix_node_parents, radix_node_cpl_deltas, 2);

    // get octree node offests for each radix tree node
    vector<int32_t> node_offests(radix_nodes.size());
    const int32_t octree_node_count = scalar::octree_node_offsets(radix_node_cpl_deltas, node_offests);
    REQUIRE(octree_node_count == 3);

    //
    vector<OctreeNode> octree_nodes(octree_node_count);

    // TEMP

    REQUIRE(octree_nodes[0].get_level() == 1);
    REQUIRE(octree_nodes[0].get_parent() == 0);
    REQUIRE(octree_nodes[0].get_prefix() == 0);
}
