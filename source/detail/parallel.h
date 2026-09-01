#pragma once
#define BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include "nbody/profile.h"
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

    template <typename T>
    void parallel_sort(BS::thread_pool& pool, const std::span<T> vec)
    {
        const size_t total_size = vec.size();
        if (total_size < 2)
            return;

        // Chunk boundaries are computed once and reused for both the sort and the merge
        // below, so the two phases can never disagree about where a chunk starts -- unlike
        // deriving them from parallel_blocks() for the sort and independently assuming a
        // uniform chunk_size for the merge, which disagree whenever total_size does not
        // divide evenly by the chunk count (BS::thread_pool spreads the remainder over the
        // first few blocks), leaving inplace_merge to run on ranges that are not actually
        // two sorted runs and silently producing an unsorted array.
        const size_t num_chunks = std::min(static_cast<size_t>(pool.get_thread_count()), total_size);
        const size_t chunk_size = total_size / num_chunks;
        const size_t remainder = total_size % num_chunks;
        const auto start = [&](const size_t chunk) { return chunk * chunk_size + std::min(chunk, remainder); };

        {
            NBODY_PROFILE_ZONE_NAMED("inner sort");
            parallel_for(pool, num_chunks, [&](const size_t chunk)
            {
                NBODY_PROFILE_ZONE_NAMED("inner sort chunk");
                std::sort(vec.begin() + start(chunk), vec.begin() + start(chunk + 1));
            });
        }

        {
            // in-place iterative merge phase: double the run length each pass until one
            // sorted run of chunks remains.
            NBODY_PROFILE_ZONE_NAMED("sorted chunk merge");
            for (size_t run = 1; run < num_chunks; run *= 2)
            {
                for (size_t left_chunk = 0; left_chunk < num_chunks; left_chunk += 2 * run)
                {
                    const size_t mid_chunk = std::min(left_chunk + run, num_chunks);
                    const size_t right_chunk = std::min(left_chunk + 2 * run, num_chunks);
                    if (mid_chunk < right_chunk)
                        std::inplace_merge(vec.begin() + start(left_chunk), vec.begin() + start(mid_chunk), vec.begin() + start(right_chunk));
                }
            }
        }
    }
}
