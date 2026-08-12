#include <atomic>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <bit>
#include "detail/radix.h"

TEST_CASE("radix tree", "[radix]")
{
    using std::vector;
    using std::pair;
    using std::countl_zero;
	using nbody::detail::radix_tree;

    SECTION("empty")
    {
        // must be ordered
        const vector<uint32_t> keys{
            0b00001,
            0b00010,
            0b01001,
            0b01110,
            0b11010,
            0b11101,
        };

        // expected outcome
        const vector<pair<int32_t, int32_t>> nodes_expected{
            { -3, -4 },
            { 0, 1 },
            { 2, 3 },
            { -1, -2 },
            { 4, 5 },
        };

        // compute the actual nodes
        vector<pair<int32_t, int32_t>> nodes(keys.size() - 1);
        radix_tree(keys, nodes);

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            REQUIRE(nodes[i] == nodes_expected[i]);
        }
    }
}
