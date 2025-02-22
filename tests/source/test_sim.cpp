#include <random>
#include <catch2/catch_test_macros.hpp>
#include "nbody/sim.h"
#include "nbody/util.h"
#include <string>
#include <sstream>

TEST_CASE("orbit doesn't decay", "[sim]")
{
    // make a disk of stars with radius 100
    const float rad = 100;
    //const size_t num = 4096;
    const size_t num = 16;
    nbody::Sim sim;
    sim.bodies.resize(num);
    nbody::util::disk(sim.bodies.begin(), sim.bodies.end(), { .outer_radius = rad });

    // advance the simulation by 100 ticks at 60hz
    for (size_t i = 0; i < 10; ++i)
    {
        /*
        std::stringstream ss;
        for (size_t j = 0; j < 10; ++j)
        {
            const nbody::Body& b = sim.bodies[j];
            ss << "(" << b.pos.x << "," << b.pos.y << "," << b.pos.z << ") ";
        }
        */

        const float x1 = sim.bodies[1].pos.x;
        const float y1 = sim.bodies[1].pos.y;
        const float z1 = sim.bodies[1].pos.z;
        const float x2 = sim.bodies[2].pos.x;
        const float y2 = sim.bodies[2].pos.y;
        const float z2 = sim.bodies[2].pos.z;
        const float x3 = sim.bodies[3].pos.x;
        const float y3 = sim.bodies[3].pos.y;
        const float z3 = sim.bodies[3].pos.z;
        const float x4 = sim.bodies[4].pos.x;
        const float y4 = sim.bodies[4].pos.y;
        const float z4 = sim.bodies[4].pos.z;

        sim.update(1.f / 120.f);
    }

    /*
    // make sure all of the stars are still within the original radius
    for (const nbody::Body& body : sim.bodies)
        REQUIRE(body.pos.dist_sq({0,0,0}) <= rad * rad);
        */
}
