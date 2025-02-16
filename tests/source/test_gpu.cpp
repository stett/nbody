#include <random>
#include <catch2/catch_test_macros.hpp>
#include "nbody/sim.h"
#include "nbody/util.h"

TEST_CASE("can evolve a sim with gpu acceleration", "[gpu]")
{
    // make a disk of stars with radius 100
    const float rad = 100;
    const size_t num = 4096;
    nbody::Sim sim;
    sim.gpu = true;
    sim.bodies.resize(num);
    nbody::util::disk(sim.bodies.begin(), sim.bodies.end(), { .outer_radius = rad });

    // advance the simulation by 1000 ticks at 60hz
    for (size_t i = 0; i < 100; ++i)
        sim.update(1.f/120.f);

    // make sure all of the stars are still within the original radius
    for (const nbody::Body& body : sim.bodies)
        REQUIRE(body.pos.dist_sq({0,0,0}) <= rad * rad);
}
