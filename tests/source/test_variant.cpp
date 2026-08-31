#include <cmath>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "nbody/sim.h"
#include "nbody/util.h"

namespace
{
    void seed_disk(nbody::Sim& sim, const size_t num)
    {
        sim.mutable_bodies().resize(num);
        nbody::util::disk(sim.mutable_bodies().begin(), sim.mutable_bodies().end(), { .outer_radius = 100.f });
    }
}

TEST_CASE("state survives a variant round trip", "[sim][variant]")
{
    nbody::Sim sim;
    REQUIRE(sim.variant() == nbody::Variant::CpuBarnesHut);

    seed_disk(sim, 512);
    for (int i = 0; i < 10; ++i)
        sim.update(1.f / 120.f);

    // shared_ptr<const State>: a const Sim must not hand out a mutable simulation,
    // or callers could change it without bumping the revision.
    const std::shared_ptr<const nbody::State> before = sim.state();
    const std::vector<nbody::Body> snapshot = before->bodies;

    REQUIRE(sim.set_variant(nbody::Variant::CpuBruteForce));

    // Pointer identity is the adopt() contract -- a solver must keep the State object
    // it was handed rather than substituting a fresh one, or Sim's handle and the
    // solver's would silently diverge after a switch.
    REQUIRE(sim.state() == before);

    // No step ran, so the carried-over bodies must match bit for bit.
    REQUIRE(sim.bodies().size() == snapshot.size());
    for (size_t i = 0; i < snapshot.size(); ++i)
    {
        REQUIRE(sim.bodies()[i].pos.x == snapshot[i].pos.x);
        REQUIRE(sim.bodies()[i].pos.y == snapshot[i].pos.y);
        REQUIRE(sim.bodies()[i].pos.z == snapshot[i].pos.z);
    }

    // brute force builds no acceleration structure, and callers must cope with that
    REQUIRE(sim.debug_node_count() == 0);

    REQUIRE(sim.set_variant(nbody::Variant::CpuBarnesHut));
    sim.accelerate();
    REQUIRE(sim.debug_node_count() > 1);
}

TEST_CASE("debug nodes are world space, with a full edge length", "[sim][debug]")
{
    // The one thing no other case pins, and the one a solver building its tree in some
    // other space will get wrong: DebugNode is world space and DebugNode::size is a full
    // edge, not a half extent. A solver that leaked its own normalized cube, or that
    // handed over a half extent, still produces a plausible-looking wireframe -- at the
    // origin, or at half scale -- so it has to be asserted rather than eyeballed.
    nbody::Sim sim;
    REQUIRE(sim.variant() == nbody::Variant::CpuBarnesHut);
    sim.set_size(1000.f);
    seed_disk(sim, 512);
    sim.accelerate();

    const size_t count = sim.debug_node_count();
    REQUIRE(count > 1);

    std::vector<nbody::DebugNode> nodes(count);
    REQUIRE(sim.write_debug_nodes(nodes) == count);

    // node 0 is the root, which spans the whole world: a half extent would give 500
    REQUIRE(nodes[0].size == 1000.f);
    REQUIRE(nodes[0].center.x == 0.f);
    REQUIRE(nodes[0].center.y == 0.f);
    REQUIRE(nodes[0].center.z == 0.f);

    // every node sits inside the root, which a unit-cube tree handed over unconverted
    // would not
    for (const nbody::DebugNode& node : nodes)
    {
        REQUIRE(node.size > 0.f);
        REQUIRE(node.size <= 1000.f);
        for (size_t axis = 0; axis < 3; ++axis)
        {
            REQUIRE(node.center[axis] - node.size * .5f >= -500.f);
            REQUIRE(node.center[axis] + node.size * .5f <= 500.f);
        }
    }
}

TEST_CASE("a short debug node span truncates rather than overflowing", "[sim][debug]")
{
    nbody::Sim sim;
    seed_disk(sim, 256);
    sim.accelerate();
    REQUIRE(sim.debug_node_count() > 4);

    // the contract is min(out.size(), debug_node_count()), so a caller that reuses one
    // buffer across variants cannot walk off the end of a smaller one
    std::vector<nbody::DebugNode> few(4);
    REQUIRE(sim.write_debug_nodes(few) == 4);

    // and an oversized buffer reports what was actually written, not its own size
    std::vector<nbody::DebugNode> many(sim.debug_node_count() + 16);
    REQUIRE(sim.write_debug_nodes(many) == sim.debug_node_count());

    // an empty span is legal and writes nothing
    REQUIRE(sim.write_debug_nodes({}) == 0);
}

TEST_CASE("every variant agrees on its own debug node count", "[sim][debug][variant]")
{
    // The count and the write have to stay in step for every variant, including the ones
    // that build nothing -- that pair is the whole contract, and it is the only thing a
    // new solver can get silently wrong.
    int tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        nbody::Sim sim(info.variant);
        seed_disk(sim, 256);
        sim.accelerate();

        std::vector<nbody::DebugNode> nodes(sim.debug_node_count());
        REQUIRE(sim.write_debug_nodes(nodes) == sim.debug_node_count());
        ++tested;
    }
    REQUIRE(tested >= 2);
}

TEST_CASE("settings carry across a variant switch", "[sim][variant]")
{
    nbody::Sim sim;
    sim.set_size(1234.f);
    sim.set_theta(0.25f);
    sim.set_gravity(2.f);
    sim.set_wrap(false);

    REQUIRE(sim.set_variant(nbody::Variant::CpuBruteForce));

    REQUIRE(sim.size() == 1234.f);
    REQUIRE(sim.theta() == 0.25f);
    REQUIRE(sim.gravity() == 2.f);
    REQUIRE(sim.wrap() == false);
}

TEST_CASE("switching to an unavailable variant is a no-op with a reason", "[sim][variant]")
{
    const nbody::Variant gpu = nbody::Variant::GpuBarnesHut;
    if (nbody::Sim::available(gpu))
        SKIP("gpu variant is available here");

    nbody::Sim sim;
    const nbody::Variant original = sim.variant();

    REQUIRE_FALSE(sim.set_variant(gpu));
    REQUIRE(sim.variant() == original);
    REQUIRE_FALSE(sim.last_error().empty());

    // the failed switch must leave a working simulation behind
    seed_disk(sim, 64);
    sim.update(1.f / 120.f);
    REQUIRE(sim.bodies().size() == 64);
}

TEST_CASE("every variant reports itself consistently", "[sim][variant]")
{
    const std::span<const nbody::VariantInfo> all = nbody::Sim::variants();
    REQUIRE(all.size() == size_t(nbody::Variant::Count));

    for (size_t i = 0; i < all.size(); ++i)
    {
        INFO("entry " << i << ": " << all[i].name);

        // the table must be indexed by the variant it describes
        REQUIRE(size_t(all[i].variant) == i);
        REQUIRE(std::string(all[i].name).length() > 0);

        // an unavailable variant must say why, or the UI has nothing to show
        if (!all[i].available)
            REQUIRE_FALSE(all[i].unavailable_reason.empty());

        // an available variant must actually construct and step
        if (all[i].available)
        {
            nbody::Sim sim(all[i].variant);
            REQUIRE(sim.variant() == all[i].variant);
            seed_disk(sim, 32);
            sim.update(1.f / 120.f);
        }
    }

    // an out-of-range variant must not be reported as usable
    REQUIRE_FALSE(nbody::Sim::available(nbody::Variant::Count));
    REQUIRE_FALSE(nbody::Sim::info(nbody::Variant::Count).available);
    {
        nbody::Sim sim;
        REQUIRE_FALSE(sim.set_variant(nbody::Variant::Count));
        REQUIRE(sim.variant() == nbody::Variant::CpuBarnesHut);
    }
}

TEST_CASE("every variant picks up bodies added after a step", "[sim][variant]")
{
    // The demo's spawn_galaxy pattern: resize the body array and fill the new tail
    // between steps, then check the new stars are actually in the solver's work set.
    //
    // NOTE: this does NOT exercise the ingest contract, despite appearances. It steps
    // via update(), and accelerate() uploads unconditionally, so the mutation reaches
    // the device whether or not ingest() ran -- gutting ingest() leaves this test
    // passing. The integrate()-only path in test_gpu.cpp is what covers that.
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        seed_disk(sim, 128);
        sim.update(1.f / 120.f);

        const size_t added = 64;
        const size_t first_added = sim.bodies().size();
        {
            std::vector<nbody::Body>& bodies = sim.mutable_bodies();
            bodies.resize(bodies.size() + added);
            nbody::util::disk(bodies.end() - added, bodies.end(),
                { .center = { 400.f, 0.f, 0.f }, .outer_radius = 50.f });
        }

        sim.update(1.f / 120.f);

        REQUIRE(sim.bodies().size() == first_added + added);

        // Skip the new group's own central mass at first_added; the stars around it
        // must have felt a force.
        size_t accelerated = 0;
        for (size_t i = first_added + 1; i < sim.bodies().size(); ++i)
            if (sim.bodies()[i].acc.size_sq() > 0.f)
                ++accelerated;

        REQUIRE(accelerated == added - 1);
    }
    REQUIRE(tested >= 2);
}

TEST_CASE("barnes-hut approximates brute force", "[sim][variant]")
{
    // Cross-algorithm, so this is an aggregate error budget rather than a per-body
    // bound: barnes-hut is an approximation by construction. It is still worth pinning,
    // because it is the only check that the tree summation and the exact summation are
    // computing the same physics at all.
    const size_t num = 512;

    nbody::Sim sim;
    seed_disk(sim, num);

    // Checked, not assumed: if this switch quietly failed the test would compare
    // Barnes-Hut against itself, get an error of exactly zero, and pass while verifying
    // nothing at all.
    REQUIRE(sim.set_variant(nbody::Variant::CpuBruteForce));
    sim.accelerate();
    const std::vector<nbody::Body> exact = sim.bodies();

    REQUIRE(sim.set_variant(nbody::Variant::CpuBarnesHut));
    sim.accelerate();
    const std::vector<nbody::Body> approx = sim.bodies();

    REQUIRE(exact.size() == approx.size());

    double total_error = 0;
    size_t counted = 0;
    for (size_t i = 0; i < exact.size(); ++i)
    {
        const float mag = std::sqrt(exact[i].acc.size_sq());
        if (mag <= 0.f)
            continue;
        const nbody::Vector delta = approx[i].acc - exact[i].acc;
        total_error += std::sqrt(delta.size_sq()) / mag;
        ++counted;
    }

    REQUIRE(counted > 0);
    const double mean_error = total_error / double(counted);
    INFO("mean relative acceleration error: " << mean_error);
    REQUIRE(mean_error < 0.05);
}

TEST_CASE("every variant tolerates bodies with no radius", "[sim][variant]")
{
    // Body::radius defaults to 0, and only brute force can skip self by index, so a body
    // meets itself at zero distance in every other variant. The force law must return
    // zero there rather than dividing by it and yielding NaN.
    //
    // Builds the bodies by hand: util::disk assigns a real radius via compute_radius, so
    // nothing seeded through it exercises the default.
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        {
            std::vector<nbody::Body>& bodies = sim.mutable_bodies();
            bodies.resize(3);
            bodies[0] = nbody::Body{ .pos = { 0.f, 0.f, 0.f }, .mass = 1000.f };
            bodies[1] = nbody::Body{ .pos = { 100.f, 0.f, 0.f }, .mass = 1.f };
            bodies[2] = nbody::Body{ .pos = { 0.f, 100.f, 0.f }, .mass = 1.f };
            for (const nbody::Body& body : bodies)
                REQUIRE(body.radius == 0.f);
        }

        sim.update(1.f / 120.f);

        for (const nbody::Body& body : sim.bodies())
        {
            REQUIRE(std::isfinite(body.acc.x));
            REQUIRE(std::isfinite(body.acc.y));
            REQUIRE(std::isfinite(body.acc.z));
            REQUIRE(std::isfinite(body.pos.x));
            REQUIRE(std::isfinite(body.pos.y));
            REQUIRE(std::isfinite(body.pos.z));
        }

        // and the physics still happened: the two orbiting stars are pulled inward
        REQUIRE(sim.bodies()[1].acc.x < 0.f);
        REQUIRE(sim.bodies()[2].acc.y < 0.f);
    }
    REQUIRE(tested >= 2);
}

TEST_CASE("every variant steps with no bodies", "[sim][variant]")
{
    // Stepping before anything is spawned is reachable -- the demo's reset path does it.
    // The GPU solvers must not dispatch then: an empty body array binds a null vk::Buffer
    // with range 0, which is invalid descriptor usage.
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        REQUIRE(sim.bodies().empty());
        sim.update(1.f / 120.f);
        REQUIRE(sim.bodies().empty());

        // and the solver is still usable afterwards
        seed_disk(sim, 32);
        sim.update(1.f / 120.f);
        REQUIRE(sim.bodies().size() == 32);
    }
    REQUIRE(tested >= 2);
}
