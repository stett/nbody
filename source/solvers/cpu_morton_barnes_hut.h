#pragma once
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>
#include <execution>
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

            {
                NBODY_PROFILE_ZONE_NAMED("build acceleration structure");

                // TODO: morton-encode the bodies, sort, and build into _nodes / _bounds with
                // detail::scalar::build_octree, then run a centre-of-mass pass and traverse.
                // Until then the bodies are left unaccelerated rather than half-accelerated.
                _keys.clear();
                _nodes.clear();
                _bounds.clear();
                _node_masses.clear();

                {
                    NBODY_PROFILE_ZONE_NAMED("allocate morton codes");
                    _keys.resize(_state->bodies.size());
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("compute morton codes");
                    const float size_inv = 1.f / _state->size;

                    detail::parallel_blocks(*_context->pool, _state->bodies.size(), [this, size_inv](const std::ptrdiff_t begin, const std::ptrdiff_t end)
                    {
                        NBODY_PROFILE_ZONE_NAMED("compute morton codes subset");
                        std::transform(_state->bodies.begin() + begin, _state->bodies.begin() + end, _keys.begin() + begin, [size_inv](const Body& b) -> Morton
                        {
                            return {
                                std::clamp((b.pos.x * size_inv) + .5, 0., 1.),
                                std::clamp((b.pos.y * size_inv) + .5, 0., 1.),
                                std::clamp((b.pos.z * size_inv) + .5, 0., 1.),
                            };
                        });
                    });
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("sort morton codes");
                    detail::parallel_sort<Morton>(*_context->pool, _keys);
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("build octree");
                    detail::parallel::build_octree<Morton>(*_context->pool, _keys, _cache, _nodes, _bounds);
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("build masses");

                    {
                        NBODY_PROFILE_ZONE_NAMED("split masses and positions");
                        // TODO: Do this only on transfer
                        _body_positions.resize(_state->bodies.size());
                        _body_masses.resize(_state->bodies.size());
                        detail::parallel_for(*_context->pool, _state->bodies.size(), [this](const size_t i)
                        {
                            _body_positions[i] = _state->bodies[i].pos;
                            _body_masses[i] = _state->bodies[i].mass;
                        });
                    }

                    {
                        NBODY_PROFILE_ZONE_NAMED("propagate leaf node masses");
                        _node_masses.resize(_nodes.size());
                        if (_node_counters.size() != _nodes.size())
                            _node_counters = std::vector<std::atomic<uint8_t>>(_nodes.size());
                        detail::scalar::build_octree_masses(_nodes, _cache.leaf_nodes, _body_positions, _body_masses, _node_masses, _node_counters);
                    }
                }
            }

            {
                NBODY_PROFILE_ZONE_NAMED("compute accelerations");
            }
        }

        [[nodiscard]] size_t debug_node_count() const override { return _bounds.size(); }

        size_t write_debug_nodes(const std::span<DebugNode> out) const override
        {
            NBODY_PROFILE_ZONE_NAMED("debug nodes");

            const float size = _state->size;
            const size_t count = std::min<size_t>(out.size(), _bounds.size());

            const float total_mass = _node_masses[0].mass;
            const float total_mass_inv = 1.f / total_mass;
            detail::parallel_for(*_context->pool, count, [&](const size_t i)
            {
                const detail::OctreeBounds<3>& bounds = _bounds[i];
                const detail::OctreeNodeMass& mass = _node_masses[i];
                const std::array<float, 3>& center = bounds.center;
                out[i] = {
                    // TODO: SIMD
                    .center = { .x = (center[0] - .5f) * size, .y = (center[1] - .5f) * size, .z = (center[2] - .5f) * size },
                    .size = 2.f * bounds.half_extent * size,
                    .weight = 1.f - (mass.mass * total_mass_inv),
                };
            });

            return count;
        }

    private:

        std::vector<Morton> _keys;
        detail::OctreeCache _cache;
        std::vector<detail::OctreeNode> _nodes;
        std::vector<detail::OctreeBounds<3>> _bounds;
        std::vector<detail::OctreeNodeMass> _node_masses;
        std::vector<std::atomic<uint8_t>> _node_counters;
        std::vector<Vector> _body_positions;
        std::vector<float> _body_masses;
    };
}
