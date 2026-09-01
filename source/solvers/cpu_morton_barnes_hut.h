#pragma once
#include <cstddef>
#include <span>
#include <vector>
#include "solvers/cpu_solver.h"
#include "nbody/debug.h"
#include "nbody/profile.h"
#include "detail/morton.h"
#include "detail/octree.h"
#include "detail/parallel.h"

namespace nbody
{
    // O(n log n) barnes-hut approximation -- the same approximation CpuBarnesHutSolver
    // makes, over a flat octree built from morton codes rather than by inserting bodies one
    // at a time. Construction is the part of a barnes-hut frame that does not parallelize
    // today (see detail/tree.h), and a morton build is what fixes that.
    //
    // NOT YET WIRED INTO Variant, and accelerate() below builds nothing: this is a hole of
    // the right shape, not a working solver. sim.cpp includes it anyway so the compiler
    // keeps it honest against the Solver interface while the tree work lands.
    class CpuMortonBarnesHutSolver final : public CpuSolver
    {
    public:

        using CpuSolver::CpuSolver;
        using Morton = detail::Morton<uint64_t, 3>;

        void adopt(StateRef state) override
        {
            _state = std::move(state);

            // last variant's tree is meaningless here
            _nodes.clear();
            _bounds.clear();
        }

        void accelerate() override
        {
            NBODY_PROFILE_ZONE();

            // TODO: morton-encode the bodies, sort, and build into _nodes / _bounds with
            // detail::scalar::build_octree, then run a centre-of-mass pass and traverse.
            // Until then the bodies are left unaccelerated rather than half-accelerated.
            _keys.clear();
            _nodes.clear();
            _bounds.clear();

            {
                NBODY_PROFILE_ZONE_NAMED("Allocate Morton codes");
                _keys.resize(_state->bodies.size());
            }

            {
                NBODY_PROFILE_ZONE_NAMED("Compute Morton codes");
                const float size_inv = 1. / _state->size;
                std::ranges::transform(_state->bodies, _keys.begin(), [size_inv](const Body& b) -> Morton
                {
                    return {
                        std::clamp((b.pos.x * size_inv) + .5, 0., 1.),
                        std::clamp((b.pos.y * size_inv) + .5, 0., 1.),
                        std::clamp((b.pos.z * size_inv) + .5, 0., 1.),
                    };
                });
            }

            {
                NBODY_PROFILE_ZONE_NAMED("Sort Morton codes");
                std::sort(_keys.begin(), _keys.end());
            }

            {
                NBODY_PROFILE_ZONE_NAMED("Build octree");
                detail::scalar::build_octree<Morton>(_keys, _nodes, _bounds);
            }
        }

        [[nodiscard]] size_t debug_node_count() const override { return _bounds.size(); }

        size_t write_debug_nodes(const std::span<DebugNode> out) const override
        {
            NBODY_PROFILE_ZONE_NAMED("debug nodes");

            const float size = _state->size;
            const size_t count = std::min<size_t>(out.size(), _bounds.size());

            std::transform(_bounds.begin(), _bounds.begin() + count, out.begin(), [size](const detail::OctreeBounds<3>& b) -> DebugNode {
                const std::array<float, 3>& c = b.center;
                return {
                    .center = { .x = (c[0] - .5f) * size, .y = (c[1] - .5f) * size, .z = (c[2] - .5f) * size, },
                    .size = 2.f * b.half_extent * size,
                    .weight = 0.f,
                };
            });

            return count;
        }

    private:

        std::vector<Morton> _keys;
        std::vector<detail::OctreeNode> _nodes;
        std::vector<detail::OctreeBounds<3>> _bounds;
    };
}
