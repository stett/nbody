#include <atomic>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "BS_thread_pool.hpp"
#include "detail/parallel.h"

// The partitioning in the old Sim::visit() dropped the trailing n % num_threads
// elements, and processed nothing at all when n < num_threads. On a 128-core machine
// that meant a 16-body simulation never moved.
TEST_CASE("parallel_for visits every index exactly once", "[parallel]")
{
    BS::thread_pool pool;

    const size_t n = GENERATE(size_t{0}, 1, 2, 3, 7, 8, 9, 127, 128, 129, 1023, 4096);

    std::vector<std::atomic<int>> hits(n);
    for (std::atomic<int>& hit : hits)
        hit.store(0);

    nbody::detail::parallel_for(pool, n, [&hits](const size_t i) { ++hits[i]; });

    for (const std::atomic<int>& hit : hits)
        REQUIRE(hit.load() == 1);
}

TEST_CASE("parallel_blocks covers the whole range exactly once", "[parallel]")
{
    BS::thread_pool pool;

    const size_t n = GENERATE(size_t{0}, 1, 5, 127, 128, 129, 1000);

    std::vector<std::atomic<int>> hits(n);
    for (std::atomic<int>& hit : hits)
        hit.store(0);

    // NOTE: Catch2's assertion macros are not thread safe — never REQUIRE from inside
    // a worker. Record findings atomically and assert on the main thread.
    std::atomic<size_t> covered{0};
    std::atomic<bool> bad_range{false};
    nbody::detail::parallel_blocks(pool, n, [&](const size_t begin, const size_t end)
    {
        if (begin > end)
        {
            bad_range = true;
            return;
        }
        covered += end - begin;
        for (size_t i = begin; i < end; ++i)
            ++hits[i];
    });

    REQUIRE_FALSE(bad_range.load());
    REQUIRE(covered.load() == n);
    for (const std::atomic<int>& hit : hits)
        REQUIRE(hit.load() == 1);
}
