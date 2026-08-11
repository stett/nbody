#pragma once
#include "solvers/cpu_solver.h"
#include "nbody/profile.h"

namespace nbody
{
    // Exact O(n^2) summation. Slow by design: this is the reference the approximate
    // variants are checked against, and it builds no tree.
    class CpuBruteForceSolver final : public CpuSolver
    {
    public:

        using CpuSolver::CpuSolver;

        void adopt(StateRef state) override { _state = std::move(state); }

        void accelerate() override
        {
            NBODY_PROFILE_ZONE();
            const float G = _state->gravity;
            const std::vector<Body>& bodies = _state->bodies;
            detail::parallel_blocks(*_context->pool, bodies.size(),
                [this, G, &bodies](const size_t begin, const size_t end)
                {
                    NBODY_PROFILE_ZONE_NAMED("brute force block");
                    for (size_t i = begin; i < end; ++i)
                    {
                        Body& body = _state->bodies[i];
                        const Vector pos = body.pos;
                        const float radius = body.radius;
                        Vector acc = { 0, 0, 0 };
                        for (size_t j = 0; j < bodies.size(); ++j)
                        {
                            // Skip self-interaction. detail::gravity returns zero at zero
                            // distance anyway, so this just saves the call.
                            if (j == i)
                                continue;
                            acc += detail::gravity(pos, radius, bodies[j].pos, bodies[j].mass, G);
                        }
                        body.acc = acc;
                    }
                });
        }
    };
}
