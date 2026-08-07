#pragma once
#include <memory>
#include "BS_thread_pool.hpp"

namespace nbody
{
    class GpuDevice;   // source/gpu.h -- keeps vulkan headers out of here

    // Resources shared by every solver a Sim runs, outliving any individual one.
    //
    // The thread pool lives here rather than in State (which would make State
    // uncopyable) or in each solver (which would spawn and join hardware_concurrency()
    // threads on every variant switch).
    struct Context
    {
        std::shared_ptr<BS::thread_pool> pool = std::make_shared<BS::thread_pool>();

        // Created on the first successful switch to a GPU variant and cached: both GPU
        // variants share one set of shaders and select between them with a push constant,
        // so switching has nothing to rebuild.
        std::shared_ptr<GpuDevice> gpu;

        // Returns `gpu`, creating it if needed. Throws if the device cannot be brought
        // up; the caller turns that into an unavailable variant with a reason.
        std::shared_ptr<GpuDevice> require_gpu();
    };
}
