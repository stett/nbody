#pragma once
#include <cstddef>
#include <utility>
#include "BS_thread_pool.hpp"

namespace nbody::detail
{
    // Run `block(begin, end)` over a partition of [0, n) across the pool, and wait.
    //
    // This replaces the old hand-rolled partitioning in Sim::visit(), which had three
    // bugs: the trailing n % num_threads elements were never visited, n < num_threads
    // produced a chunk size of 0 so nothing ran at all, and hardware_concurrency()
    // returning 0 divided by zero. BS::thread_pool::blocks already clamps the block
    // count to the range length and distributes the remainder, so deferring to
    // submit_blocks fixes all three.
    template <typename Block>
    void parallel_blocks(BS::thread_pool& pool, const size_t n, Block&& block)
    {
        if (n == 0)
            return;
        pool.submit_blocks(size_t{0}, n, std::forward<Block>(block)).wait();
    }

    // Per-index convenience wrapper. Prefer parallel_blocks when the body can hoist
    // work out of the inner loop.
    template <typename Fn>
    void parallel_for(BS::thread_pool& pool, const size_t n, Fn&& fn)
    {
        parallel_blocks(pool, n, [&fn](const size_t begin, const size_t end)
        {
            for (size_t i = begin; i < end; ++i)
                fn(i);
        });
    }
}
