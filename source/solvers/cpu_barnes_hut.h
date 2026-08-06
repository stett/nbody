#pragma once
#include "solvers/cpu_solver.h"
#include "detail/tree.h"

namespace nbody
{
    // O(n log n) barnes-hut approximation: build the tree, then sum forces against
    // nodes far enough away to be treated as a single mass.
    class CpuBarnesHutSolver final : public CpuSolver
    {
    public:

        using CpuSolver::CpuSolver;

        void adopt(StateRef state) override
        {
            _state = std::move(state);
            _tree.clear({ .size = _state->size });   // last variant's tree is meaningless here
        }

        void accelerate() override
        {
            detail::build_tree(_tree, _state->bodies, _state->size);

            const float theta = _state->theta;
            const float G = _state->gravity;
            detail::parallel_blocks(*_context->pool, _state->bodies.size(),
                [this, theta, G](const size_t begin, const size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        Body& body = _state->bodies[i];
                        body.acc = { 0, 0, 0 };
                        _tree.apply(body.pos, [this, &body, G](const bh::Node& node)
                        {
                            body.acc += detail::gravity(body.pos, body.radius, node.com, node.mass, G);
                        }, theta);
                    }
                });
        }

        [[nodiscard]] const bh::Tree* tree() const override { return &_tree; }

    private:

        bh::Tree _tree;
    };
}
