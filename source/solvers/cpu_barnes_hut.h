#pragma once
#include "solvers/cpu_solver.h"
#include "detail/tree.h"
#include "nbody/profile.h"

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
            NBODY_PROFILE_ZONE();
            detail::build_tree(_tree, _state->bodies, _state->size);

            const float theta = _state->theta;
            const float G = _state->gravity;
            detail::parallel_blocks(*_context->pool, _state->bodies.size(),
                [this, theta, G](const size_t begin, const size_t end)
                {
                    // Not zoned per traversal: hundreds of node visits per body.
                    NBODY_PROFILE_ZONE_NAMED("barnes-hut block");
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

        [[nodiscard]] size_t debug_node_count() const override { return _tree.nodes().size(); }

        size_t write_debug_nodes(const std::span<DebugNode> out) const override
        {
            return detail::write_debug_nodes(_tree, out);
        }

    private:

        bh::Tree _tree;
    };
}
