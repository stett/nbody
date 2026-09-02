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

                _keyed.clear();
                _keys.clear();
                _nodes.clear();
                _bounds.clear();
                _node_masses.clear();

                {
                    NBODY_PROFILE_ZONE_NAMED("allocate morton codes");
                    _keyed.resize(_state->bodies.size());
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("compute morton codes");
                    const float size_inv = 1.f / _state->size;

                    detail::parallel_blocks(*_context->pool, _state->bodies.size(), [this, size_inv](const std::ptrdiff_t begin, const std::ptrdiff_t end)
                    {
                        NBODY_PROFILE_ZONE_NAMED("compute morton codes subset");
                        for (std::ptrdiff_t i = begin; i < end; ++i)
                        {
                            const Body& b = _state->bodies[i];
                            _keyed[i] = {
                                Morton(
                                    std::clamp((b.pos.x * size_inv) + .5, 0., 1.),
                                    std::clamp((b.pos.y * size_inv) + .5, 0., 1.),
                                    std::clamp((b.pos.z * size_inv) + .5, 0., 1.)),
                                static_cast<int32_t>(i),
                            };
                        }
                    });
                }

                {
                    // Sorting the key alongside the body index it came from is what keeps
                    // the gather below aligned to the right body: sorting a bare
                    // vector<Morton>, as before, has nowhere to carry that index and
                    // forgets it, which is the bug this replaces.
                    NBODY_PROFILE_ZONE_NAMED("sort morton codes");
                    detail::parallel_sort<KeyedMorton>(*_context->pool, _keyed);
                }

                {
                    // build_octree wants a plain span<const Morton>; this is a sequential
                    // copy over data that's already in its final sorted order, not a gather.
                    NBODY_PROFILE_ZONE_NAMED("extract sorted keys");
                    _keys.resize(_keyed.size());
                    detail::parallel_for(*_context->pool, _keyed.size(), [this](const size_t i)
                    {
                        _keys[i] = _keyed[i].key;
                    });
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("build octree");
                    detail::parallel::build_octree<Morton>(*_context->pool, _keys, _cache, _nodes, _bounds);
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("build masses");

                    {
                        // Leaf slot i_leaf needs the body that produced the i_leaf-th sorted
                        // key -- _keyed[i_leaf].body_index -- not body i_leaf itself. This is
                        // the one random-access pass the fix costs: everywhere else only
                        // touches the small (key, index) pairs, not the full body data.
                        NBODY_PROFILE_ZONE_NAMED("gather positions and masses into sorted order");
                        _body_positions.resize(_keyed.size());
                        _body_masses.resize(_keyed.size());
                        detail::parallel_for(*_context->pool, _keyed.size(), [this](const size_t i_leaf)
                        {
                            NBODY_PROFILE_ZONE_NAMED("gather positions block");
                            const int32_t i_body = _keyed[i_leaf].body_index;
                            _body_positions[i_leaf] = _state->bodies[i_body].pos;
                            _body_masses[i_leaf] = _state->bodies[i_body].mass;
                        });
                    }

                    {
                        NBODY_PROFILE_ZONE_NAMED("propagate leaf node masses");
                        _node_masses.resize(_nodes.size());
                        if (_node_counters.size() != _nodes.size())
                            _node_counters = std::vector<std::atomic<uint8_t>>(_nodes.size());
                        detail::parallel::build_octree_masses(*_context->pool, _nodes, _cache.leaf_nodes, _body_positions, _body_masses, _node_masses, _node_counters);
                    }
                }
            }

            {
                NBODY_PROFILE_ZONE_NAMED("compute accelerations");
                const float theta = _state->theta;
                const float G = _state->gravity;
                const float size = _state->size;
                detail::parallel_blocks(*_context->pool, _state->bodies.size(),
                    [this, theta, G, size](const size_t begin, const size_t end)
                    {
                        NBODY_PROFILE_ZONE_NAMED("barnes-hut block");
                        for (size_t i = begin; i < end; ++i)
                        {
                            Body& body = _state->bodies[i];
                            body.acc = { 0, 0, 0 };
                            detail::scalar::apply_octree(
                                _nodes,
                                _bounds,
                                _node_masses,
                                body.pos, [this, &body, G](const int32_t node_index)
                            {
                                const detail::OctreeNodeMass& node_mass = _node_masses[node_index];
                                body.acc += detail::gravity(body.pos, body.radius, node_mass.center, node_mass.mass, G);
                            }, theta, size);
                        }
                    }
                );
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

        // A morton key paired with the index of the body that produced it. Sorting these
        // together, by key alone (see operator< below), is what keeps that pairing intact
        // through the sort.
        struct KeyedMorton
        {
            Morton key;
            int32_t body_index;
            bool operator<(const KeyedMorton& rhs) const { return key < rhs.key; }
        };

        std::vector<KeyedMorton> _keyed;
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
