#pragma once
#include "context.h"
#include "solver.h"
#include "detail/parallel.h"
#include "detail/physics.h"

namespace nbody
{
    // Shared base for the CPU solvers. Both work on State::bodies in place, so the
    // standard-format conversion is free in both directions: state() is a plain getter
    // and ingest() has nothing to do.
    //
    // The integrator lives here rather than in each solver so that the barnes-hut and
    // brute-force variants cannot drift apart on anything but the force summation.
    class CpuSolver : public Solver
    {
    public:

        // Public rather than inherited: an inheriting constructor keeps the access of
        // the base declaration, and Solver's is protected, so make_unique could not
        // reach it through a `using` chain.
        CpuSolver(std::shared_ptr<Context> context, StateRef state)
            : Solver(std::move(context), std::move(state))
        {}

        [[nodiscard]] StateRef state() const override { return _state; }

        void integrate(const float dt) override
        {
            const float size = _state->size;
            const bool wrap = _state->wrap;
            detail::parallel_blocks(*_context->pool, _state->bodies.size(),
                [this, dt, size, wrap](const size_t begin, const size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                        detail::integrate_euler(_state->bodies[i], dt, size, wrap);
                });
        }
    };
}
