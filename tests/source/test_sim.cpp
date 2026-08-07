#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "nbody/sim.h"
#include "nbody/util.h"

namespace
{
    // util::disk writes the central mass at the first element and orbiting stars after
    // it, so orbit assertions must skip index 0.
    constexpr size_t first_star = 1;

    float radius_of(const nbody::Body& body)
    {
        return std::sqrt(body.pos.dist_sq({ 0, 0, 0 }));
    }
}

TEST_CASE("orbits neither decay nor escape", "[sim]")
{
    // A disk of stars on keplerian orbits about a dominant central mass. Each star
    // should hold its own orbital radius; the test tracks per-body drift in BOTH
    // directions, because "doesn't decay" is a statement about collapse, and an
    // outward-only bound would be satisfied by a star spiralling into the center.
    const float rad = 100;
    const size_t num = 16;
    nbody::Sim sim;
    sim.mutable_bodies().resize(num);
    nbody::util::disk(sim.mutable_bodies().begin(), sim.mutable_bodies().end(), { .outer_radius = rad });

    std::vector<float> initial;
    std::vector<nbody::Vector> previous;
    for (const nbody::Body& body : sim.bodies())
    {
        initial.push_back(radius_of(body));
        previous.push_back(body.pos);
    }
    std::vector<float> path(sim.bodies().size(), 0.f);

    // At r=100 around this central mass one orbit is roughly 3.1s, so run about a full
    // orbit rather than the ~2.7% of one that a 10-step run covers. A short run cannot
    // distinguish a stable orbit from a frozen simulation.
    const float dt = 1.f / 120.f;
    for (size_t step = 0; step < 400; ++step)
    {
        sim.update(dt);

        // Accumulate arc length. Displacement from the start point is useless here:
        // after a full orbit a star is back where it began, so a healthy orbit and a
        // frozen one look identical by that measure.
        for (size_t i = 0; i < sim.bodies().size(); ++i)
        {
            path[i] += std::sqrt(sim.bodies()[i].pos.dist_sq(previous[i]));
            previous[i] = sim.bodies()[i].pos;
        }
    }

    float worst_drift = 0.f;
    float least_path = std::numeric_limits<float>::max();
    for (size_t i = first_star; i < sim.bodies().size(); ++i)
    {
        const float r0 = initial[i];
        if (r0 <= 0.f)
            continue;
        worst_drift = std::max(worst_drift, std::abs(radius_of(sim.bodies()[i]) - r0) / r0);
        least_path = std::min(least_path, path[i] / r0);
    }

    // Holding a radius is only meaningful if the stars actually went somewhere: a frozen
    // simulation trivially has zero drift, which is how this test's predecessor passed
    // against a completely inert sim. A full circular orbit is 2*pi radii of arc, so
    // require at least a half turn of real travel before judging the radius.
    REQUIRE(least_path > 3.14f);

    // Measured worst-case radial drift over this run is ~1.4%; the 10% bound leaves
    // room for integrator noise while still tripping on a genuine decay or blow-up.
    REQUIRE(worst_drift < 0.10f);
}

TEST_CASE("every body is integrated regardless of body count", "[sim]")
{
    // Regression test for the partitioning in the parallel body loop, which dropped the
    // trailing bodies.size() % num_threads elements and processed nothing at all when
    // the body count was below the thread count. On a many-core machine that left small
    // simulations completely inert, and nothing caught it because the only test in the
    // suite asserted nothing.
    //
    // Counts deliberately straddle typical thread counts, including non-multiples so a
    // dropped remainder is caught.
    const size_t num = GENERATE(size_t{2}, 3, 5, 16, 129);

    nbody::Sim sim;
    sim.mutable_bodies().resize(num);
    nbody::util::disk(sim.mutable_bodies().begin(), sim.mutable_bodies().end(), { .outer_radius = 50.f });

    std::vector<nbody::Vector> before;
    for (const nbody::Body& body : sim.bodies())
        before.push_back(body.pos);

    for (size_t i = 0; i < 10; ++i)
        sim.update(1.f / 120.f);

    // every orbiting star must have moved; the central mass at index 0 may not
    for (size_t i = first_star; i < sim.bodies().size(); ++i)
        REQUIRE(sim.bodies()[i].pos.dist_sq(before[i]) > 0.f);
}
