#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "body.h"
#include "constants.h"

namespace nbody
{
    // The canonical, backend-independent simulation state: everything that defines the
    // simulation and must survive a change of variant. Derived data (the barnes-hut
    // tree) and resources (the thread pool, the vulkan device) deliberately live
    // elsewhere -- the first because not every variant has one, the second because a
    // resource is not state and would make State uncopyable.
    //
    // Every variant speaks this format. A variant that works directly on `bodies` uses
    // it in place at no cost; one that keeps its own representation (device buffers,
    // struct-of-arrays, a morton-sorted copy) converts to and from it at the sync
    // points defined by Solver.
    struct State
    {
        // world extent; space wraps over [-size/2, +size/2] when `wrap` is set
        float size = 10000.f;

        // barnes-hut opening angle; ignored by brute-force variants
        float theta = .5f;

        // gravitational constant
        float gravity = G;

        // whether space wraps into a 3-torus
        bool wrap = true;

        // The canonical body array. For a variant that works on this vector directly it
        // IS the simulation; for one holding its own representation it is a cache that
        // Solver::state() materializes on demand. Either way it is guaranteed current
        // when Sim::state() or Sim::bodies() returns.
        std::vector<Body> bodies;

        // Bumped whenever a caller takes mutating access. Sim compares this against the
        // revision the active solver last ingested to decide whether that solver needs
        // to re-converge, so this is the single source of truth for staleness in the
        // caller -> solver direction and solvers do not track their own copy.
        uint64_t revision = 0;
        void touch() noexcept { ++revision; }
    };

    using StateRef = std::shared_ptr<State>;
}
