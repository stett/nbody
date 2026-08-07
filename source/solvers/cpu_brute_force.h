#pragma once
#include "solvers/cpu_solver.h"

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
            const float G = _state->gravity;
            const std::vector<Body>& bodies = _state->bodies;
            detail::parallel_blocks(*_context->pool, bodies.size(),
                [this, G, &bodies](const size_t begin, const size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        Body& body = _state->bodies[i];
                        const Vector pos = body.pos;
                        const float radius = body.radius;
                        Vector acc = { 0, 0, 0 };
                        for (size_t j = 0; j < bodies.size(); ++j)
                        {
                            // Skip self-interaction. detail::gravity already returns zero
                            // at zero distance, so this is an optimization rather than a
                            // correctness guard -- it saves a call per body, and only
                            // brute force has the index needed to do it.
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
