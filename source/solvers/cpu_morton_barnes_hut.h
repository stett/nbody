#pragma once
#include <cstddef>
#include <span>
#include <vector>
#include "solvers/cpu_solver.h"
#include "nbody/debug.h"
#include "nbody/profile.h"
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
            _nodes.clear();
            _bounds.clear();

            for (Body& body : _state->bodies)
                body.acc = { 0, 0, 0 };
        }

        [[nodiscard]] size_t debug_node_count() const override { return _bounds.size(); }

        size_t write_debug_nodes(const std::span<DebugNode> out) const override
        {
            // TODO: two conversions are needed here and neither is optional.
            //
            // detail::OctreeBounds lives in the unit cube -- that is the space to_morton()
            // normalizes into -- while DebugNode is world space, so this has to undo
            // whatever normalization the build used. It also stores a HALF extent where
            // DebugNode wants a full edge:
            //
            //     .center = (b.center - 0.5) * world_size
            //     .size   = 2 * b.half_extent * world_size
            //
            // weight stays 0 regardless: OctreeNode carries no mass or centre of mass, so
            // there is nothing to compute a potential from until that pass exists.
            (void)out;
            return 0;
        }

    private:

        std::vector<detail::OctreeNode> _nodes;
        std::vector<detail::OctreeBounds<3>> _bounds;
    };
}
