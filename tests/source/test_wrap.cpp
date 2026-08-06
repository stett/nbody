#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include "nbody/sim.h"

// The GLSL integrate shader used to skip the toroidal wrap entirely while the CPU path
// applied it, so the two backends disagreed about where bodies ended up. These tests pin
// the shared behavior, and cover `wrap` being switchable at run time.
namespace
{
    // Sim holds a BS::thread_pool by value, which is neither copyable nor movable, so
    // configure in place rather than returning one.
    void setup_drifting(nbody::Sim& sim, const bool use_gpu)
    {
        sim.set_size(1000.f);
        sim.set_gravity(0.f);   // isolate the integrator from any gravitational force
        sim.mutable_bodies().resize(1);
        sim.mutable_bodies()[0] = nbody::Body{
            .pos = { 499.f, 0.f, 0.f },
            .radius = 1.f,
            .vel = { 10.f, 0.f, 0.f },
            .mass = 1.f,
        };
        if (use_gpu)
            sim.init_gpu();
    }
}

TEST_CASE("cpu wraps a body that leaves the world box", "[sim][wrap]")
{
    nbody::Sim sim;
    setup_drifting(sim, false);
    REQUIRE_FALSE(sim.using_gpu());

    sim.update(1.f);   // unwrapped this would land at ~509

    const nbody::Vector p = sim.bodies()[0].pos;
    REQUIRE(std::abs(p.x) <= 500.f);
    REQUIRE(std::abs(p.y) <= 500.f);
    REQUIRE(std::abs(p.z) <= 500.f);
}

TEST_CASE("cpu leaves the body alone when wrap is off", "[sim][wrap]")
{
    nbody::Sim sim;
    setup_drifting(sim, false);
    sim.set_wrap(false);

    sim.update(1.f);

    REQUIRE(sim.bodies()[0].pos.x > 500.f);
}

TEST_CASE("gpu wraps identically to the cpu", "[sim][wrap][gpu]")
{
    nbody::Sim gpu_sim;
    setup_drifting(gpu_sim, true);
    if (!gpu_sim.using_gpu())
        SKIP("no usable Vulkan compute device");

    nbody::Sim cpu_sim;
    setup_drifting(cpu_sim, false);

    gpu_sim.update(1.f);
    cpu_sim.update(1.f);

    const nbody::Vector g = gpu_sim.bodies()[0].pos;
    const nbody::Vector c = cpu_sim.bodies()[0].pos;

    REQUIRE(std::abs(g.x) <= 500.f);
    REQUIRE(std::abs(g.x - c.x) < 1e-3f);
    REQUIRE(std::abs(g.y - c.y) < 1e-3f);
    REQUIRE(std::abs(g.z - c.z) < 1e-3f);
}

TEST_CASE("gpu honors wrap being switched off", "[sim][wrap][gpu]")
{
    nbody::Sim sim;
    setup_drifting(sim, true);
    if (!sim.using_gpu())
        SKIP("no usable Vulkan compute device");

    sim.set_wrap(false);
    sim.update(1.f);

    REQUIRE(sim.bodies()[0].pos.x > 500.f);
}
