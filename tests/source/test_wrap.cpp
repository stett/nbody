#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include "nbody/sim.h"

// The GLSL integrate shader used to skip the toroidal wrap entirely while the CPU path
// applied it, so the two backends disagreed about where bodies ended up. These tests pin
// the shared behavior across every available variant, and cover `wrap` being switchable
// at run time.
//
// Each loop counts the variants it actually exercised and requires a floor. Skipping
// unavailable variants is legitimate, but without a floor a run where everything was
// unavailable would report a pass having tested nothing.
namespace
{
    void setup_drifting(nbody::Sim& sim)
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
    }

    // both CPU variants are always available
    constexpr size_t min_variants = 2;
}

TEST_CASE("every variant wraps a body that leaves the world box", "[sim][wrap]")
{
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);

        // guard against a silent fallback making this loop re-test the default
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        setup_drifting(sim);
        sim.update(1.f);   // unwrapped this would land at ~509

        const nbody::Vector p = sim.bodies()[0].pos;
        REQUIRE(std::abs(p.x) <= 500.f);
        REQUIRE(std::abs(p.y) <= 500.f);
        REQUIRE(std::abs(p.z) <= 500.f);
    }
    REQUIRE(tested >= min_variants);
}

TEST_CASE("every variant honors wrap being switched off", "[sim][wrap]")
{
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        setup_drifting(sim);
        sim.set_wrap(false);

        sim.update(1.f);

        REQUIRE(sim.bodies()[0].pos.x > 500.f);
    }
    REQUIRE(tested >= min_variants);
}

TEST_CASE("all variants agree on where a wrapped body lands", "[sim][wrap]")
{
    // Pin every available variant against the always-present CPU reference, so a
    // backend that wraps differently (as the GPU shader once did) is caught.
    nbody::Sim reference(nbody::Variant::CpuBarnesHut);
    setup_drifting(reference);
    reference.update(1.f);
    const nbody::Vector expected = reference.bodies()[0].pos;

    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        setup_drifting(sim);
        sim.update(1.f);

        const nbody::Vector p = sim.bodies()[0].pos;
        REQUIRE(std::abs(p.x - expected.x) < 1e-3f);
        REQUIRE(std::abs(p.y - expected.y) < 1e-3f);
        REQUIRE(std::abs(p.z - expected.z) < 1e-3f);
    }
    REQUIRE(tested >= min_variants);
}
