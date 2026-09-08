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
#include "detail/physics.h"

namespace nbody
{
    using std::vector;
    using std::span;

    // O(n log n) barnes-hut approximation -- the same approximation CpuBarnesHutSolver
    // makes, over a flat octree built from morton codes rather than by inserting bodies one
    // at a time. Construction is the part of a barnes-hut frame that does not parallelize
    // today (see detail/tree.h), and a morton build is what fixes that.
    class SimdMortonBarnesHutSolver final : public CpuSolver
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

            // set the dirty flags so that we'll digest this new state next time
            _external_dirty = true;
            _internal_dirty = false;
        }

        [[nodiscard]] StateRef state() const override
        {
            if (_internal_dirty)
            {
                _internal_dirty = false;
                scatter_bodies(_state->bodies);
            }
            return _state;
        }

        void ingest() const override
        {
            _external_dirty = true;
        }

        /*
        void update(const float dt) override
        {
            NBODY_PROFILE_ZONE();

            // if external state has changed, update simd vectors
            if (_external_dirty)
            {
                _external_dirty = false;
                gather_bodies(_state->bodies);
            }

            // do the regular update (call accelerate and integrate)
            CpuSolver::update(dt);
        }
        */

        void accelerate() override
        {
            NBODY_PROFILE_ZONE();

            // if external state has changed, update simd vectors
            if (_external_dirty)
            {
                _external_dirty = false;
                gather_bodies(_state->bodies);
            }

            // this function modifies internal state, so mark it dirty
            _internal_dirty = true;

            {
                NBODY_PROFILE_ZONE_NAMED("build acceleration structure");

                {
                    NBODY_PROFILE_ZONE_NAMED("allocations");
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
                    {
                        NBODY_PROFILE_ZONE_NAMED("allocations");
                        if (_keys.size() != _keyed.size())
                            _keys.resize(_keyed.size());
                    }
                    detail::parallel_for(*_context->pool, _keyed.size(), [this](const size_t i)
                        {
                            _keys[i] = _keyed[i].key;
                        });
                }

                {
                    NBODY_PROFILE_ZONE_NAMED("build octree");
                    detail::parallel::build_octree_cache<Morton>(*_context->pool, _keys, _cache);
                    if (_nodes.size() != _cache.num_octree_nodes)
                        _nodes.resize(_cache.num_octree_nodes);
                    if (_bounds.size() != _cache.num_octree_nodes)
                        _bounds.resize(_cache.num_octree_nodes);
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
                        {
                            NBODY_PROFILE_ZONE_NAMED("allocations");
                            _body_positions.resize(_keyed.size());
                            _body_masses.resize(_keyed.size());
                        }
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
                        {
                            NBODY_PROFILE_ZONE_NAMED("allocations");
                            if (_node_masses.size() < _nodes.size())
                                _node_masses.resize(_nodes.size());
                            if (_node_counters.size() < _nodes.size())
                                _node_counters = std::vector<std::atomic<uint8_t>>(_nodes.size());
                        }
                        {
                            NBODY_PROFILE_ZONE_NAMED("build node masses");
                            detail::parallel::build_octree_masses(*_context->pool, _nodes, _cache.leaf_nodes, _body_positions, _body_masses, std::span(_node_masses).subspan(0, _nodes.size()), std::span(_node_counters).subspan(0, _nodes.size()));
                        }
                    }
                }
            }

            {
                NBODY_PROFILE_ZONE_NAMED("compute accelerations");
                const float theta = _state->theta;
                const float G = _state->gravity;
                const float size = _state->size;
                detail::parallel_blocks(*_context->pool, x.size(),
                    [this, theta, G, size](const size_t begin, const size_t end)
                    {
                        NBODY_PROFILE_ZONE_NAMED("barnes-hut block");
                        for (size_t i = begin; i < end; ++i)
                        {
                            ax[i] = 0.f;
                            ay[i] = 0.f;
                            az[i] = 0.f;

                            Vector pos = { x[i], y[i], z[i] };

                            detail::scalar::apply_octree(
                                _nodes,
                                _bounds,
                                _node_masses,
                                pos, [this, &pos, i, G](const int32_t node_index)
                                {
                                    const detail::OctreeNodeMass& node_mass = _node_masses[node_index];
                                    Vector a = detail::gravity(pos, r[i], node_mass.center, node_mass.mass, G);
                                    ax[i] += a.x;
                                    ay[i] += a.y;
                                    az[i] += a.z;
                                }, theta, size
                            );
                        }
                    }
                );
            }
        }

        void integrate(const float dt) override
        {
            // if external state has changed, update simd vectors
            if (_external_dirty)
            {
                _external_dirty = false;
                gather_bodies(_state->bodies);
            }

            // this function modifies internal state, so mark it dirty
            _internal_dirty = true;

            NBODY_PROFILE_ZONE();
            const float size = _state->size;
            const bool wrap = _state->wrap;

            detail::parallel_blocks(*_context->pool, _state->bodies.size(),
            [this, dt, size, wrap](const size_t begin, const size_t end)
            {
                // Inside the block, so each worker's share shows on its own thread.
                NBODY_PROFILE_ZONE_NAMED("integrate block");
                const size_t i = begin;
                const size_t l = end - begin;
                detail::simd::integrate_euler(
                    span(x).subspan(i, l), span(y).subspan(i, l), span(z).subspan(i, l),
                    span(vx).subspan(i, l), span(vy).subspan(i, l), span(vz).subspan(i, l),
                    span(ax).subspan(i, l), span(ay).subspan(i, l), span(az).subspan(i, l),
                    dt, size, wrap);
            });
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
                    .center = {.x = (center[0] - .5f) * size, .y = (center[1] - .5f) * size, .z = (center[2] - .5f) * size },
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

        void resize_simd_vectors(const size_t num_bodies)
        {
            NBODY_PROFILE_ZONE();

            if (r.size() != num_bodies) r.resize(num_bodies);
            if (m.size() != num_bodies) m.resize(num_bodies);
            if (x.size() != num_bodies) x.resize(num_bodies);
            if (y.size() != num_bodies) y.resize(num_bodies);
            if (z.size() != num_bodies) z.resize(num_bodies);
            if (vx.size() != num_bodies) vx.resize(num_bodies);
            if (vy.size() != num_bodies) vy.resize(num_bodies);
            if (vz.size() != num_bodies) vz.resize(num_bodies);
            if (ax.size() != num_bodies) ax.resize(num_bodies);
            if (ay.size() != num_bodies) ay.resize(num_bodies);
            if (az.size() != num_bodies) az.resize(num_bodies);
        }

        // gather data from body states to simd vectors
        void gather_bodies(const span<const Body> bodies)
        {
            NBODY_PROFILE_ZONE();

            resize_simd_vectors(bodies.size());
            detail::parallel_for(*_context->pool, bodies.size(), [this, &bodies](const size_t i)
            {
                const Body& body = bodies[i];
                r[i] = body.radius;
                m[i] = body.mass;
                x[i] = body.pos.x;
                y[i] = body.pos.y;
                z[i] = body.pos.z;
                vx[i] = body.vel.x;
                vy[i] = body.vel.y;
                vz[i] = body.vel.z;
                ax[i] = body.acc.x;
                ay[i] = body.acc.y;
                az[i] = body.acc.z;
            });
        }

        // scatter data from simd vectors to body states
        void scatter_bodies(const span<const Body> bodies) const
        {
            NBODY_PROFILE_ZONE();

            detail::parallel_for(*_context->pool, bodies.size(), [this, &bodies](const size_t i)
            {
                Body& body = const_cast<Body&>(bodies[i]);
                body.radius = r[i];
                body.mass = m[i];
                body.pos.x = x[i];
                body.pos.y = y[i];
                body.pos.z = z[i];
                body.vel.x = vx[i];
                body.vel.y = vy[i];
                body.vel.z = vz[i];
                body.acc.x = ax[i];
                body.acc.y = ay[i];
                body.acc.z = az[i];
            });
        }

        vector<KeyedMorton> _keyed;
        vector<Morton> _keys;
        detail::OctreeCache _cache;
        vector<detail::OctreeNode> _nodes;
        vector<detail::OctreeBounds<3>> _bounds;
        vector<detail::OctreeNodeMass> _node_masses;
        vector<std::atomic<uint8_t>> _node_counters;
        vector<Vector> _body_positions;
        vector<float> _body_masses;

        mutable bool _external_dirty = false;   // has the shared state object been modified externally to this object
        mutable bool _internal_dirty = false;   // has the internal data changed since the last time the shared data was updated

        // body properties
        vector<float> r;            // radius
        vector<float> m;            // mass
        vector<float> x, y, z;      // position
        vector<float> vx, vy, vz;   // velocity
        vector<float> ax, ay, az;   // acceleration
    };
}
