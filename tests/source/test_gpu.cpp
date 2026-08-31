#include <cmath>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "nbody/sim.h"
#include "nbody/util.h"

// The GPU solver is the first one that keeps a representation of its own, so it is the
// first real exercise of the State conversion protocol. Everything here skips cleanly
// when no compute device is present.
namespace
{
    bool skip_without_gpu(const nbody::Variant v)
    {
        if (nbody::Sim::available(v))
            return false;
        SKIP("no usable Vulkan compute device: " + nbody::Sim::info(v).unavailable_reason);
        return true;
    }

    void seed_disk(nbody::Sim& sim, const size_t num)
    {
        sim.mutable_bodies().resize(num);
        nbody::util::disk(sim.mutable_bodies().begin(), sim.mutable_bodies().end(), { .outer_radius = 100.f });
    }

    float max_relative_acc_error(const std::vector<nbody::Body>& a, const std::vector<nbody::Body>& b)
    {
        float worst = 0.f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const float mag = std::sqrt(a[i].acc.size_sq());
            if (mag <= 0.f)
                continue;
            const nbody::Vector delta = b[i].acc - a[i].acc;
            worst = std::max(worst, std::sqrt(delta.size_sq()) / mag);
        }
        return worst;
    }
}

TEST_CASE("integrating first thing binds valid node storage", "[sim][gpu]")
{
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    // integrate() stages no tree, so nothing had sized the node buffer it binds regardless.
    // Only detectable under validation, hence the coarse check: what is asserted here is
    // that the path runs at all, on a solver no accelerate() or update() has been through.
    nbody::Sim sim(v);
    seed_disk(sim, 256);

    const nbody::Vector vel = { 10.f, 0.f, 0.f };
    const nbody::Vector from = sim.bodies()[3].pos;
    sim.mutable_bodies()[3].vel = vel;
    sim.mutable_bodies()[3].acc = { 0.f, 0.f, 0.f };

    constexpr float dt = 0.1f;
    sim.integrate(dt);

    const nbody::Vector expected = from + vel * dt;
    const nbody::Vector p = sim.bodies()[3].pos;
    REQUIRE(std::sqrt((p - expected).size_sq()) < 1e-2f);
}

TEST_CASE("stepping blind agrees with stepping while reading", "[sim][gpu]")
{
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    // Readback is a copy and changes nothing the device computes, so how often a caller
    // reads must not change where the bodies end up. Long enough a run that the tree
    // outgrows its allocation partway through: that is what regressed, a device buffer
    // moving and forcing an upload of staging arrays no read had refreshed, which put the
    // first frame's velocities back over the device's own.
    constexpr float dt = 0.02f;
    constexpr size_t num = 8192;
    constexpr int steps = 64;

    nbody::Sim seed(v);
    seed_disk(seed, num);
    const std::vector<nbody::Body> initial = seed.bodies();

    // One at a time: the variants share a device, so a live second Sim would be stepping
    // over the first one's buffers.
    const auto run = [&](const bool read_every_step)
    {
        nbody::Sim sim(v);
        sim.mutable_bodies() = initial;
        for (int i = 0; i < steps; ++i)
        {
            sim.update(dt);
            if (read_every_step)
                REQUIRE(sim.bodies().size() == num);
        }
        return sim.bodies();
    };

    const std::vector<nbody::Body> blind = run(false);
    const std::vector<nbody::Body> reading = run(true);

    REQUIRE(blind.size() == reading.size());

    float worst = 0.f;
    for (size_t i = 0; i < blind.size(); ++i)
    {
        worst = std::max(worst, std::sqrt((blind[i].pos - reading[i].pos).size_sq()));
        worst = std::max(worst, std::sqrt((blind[i].vel - reading[i].vel).size_sq()));
    }

    INFO("worst divergence: " << worst);
    REQUIRE(worst < 1e-3f);
}

TEST_CASE("reading bodies materializes the device's work", "[sim][gpu]")
{
    // Both body layouts, so the conversion protocol is checked for each rather than for
    // whichever one happens to be wired to this variant today.
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    nbody::Sim sim(v);
    REQUIRE(sim.variant() == v);
    seed_disk(sim, 1024);

    // accelerate() alone leaves the new accelerations on the device. Read the bodies
    // with no intervening integrate() or explicit state() call: the read path itself
    // has to trigger the conversion, which is exactly what an implementation serving
    // reads from a cached State would get wrong.
    sim.accelerate();

    size_t moving = 0;
    for (size_t i = 1; i < sim.bodies().size(); ++i)
        if (sim.bodies()[i].acc.size_sq() > 0.f)
            ++moving;

    REQUIRE(moving > 0);
}

TEST_CASE("a mutation reaches the device on the integrate-only path", "[sim][gpu]")
{
    // Both body layouts, so the conversion protocol is checked for each rather than for
    // whichever one happens to be wired to this variant today.
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    nbody::Sim sim(v);
    seed_disk(sim, 256);
    sim.update(1.f / 120.f);

    // Move a body somewhere unmistakable and step WITHOUT accelerate(). This is the
    // case that regressed: the upload only happened inside accelerate(), so integrate()
    // alone stepped the device's pre-mutation copy and then downloaded the result over
    // the caller's write, destroying it. Reading through bodies() is not enough to
    // catch it -- the read path materializes nothing, so the write appears to survive
    // until a step actually runs.
    const nbody::Vector marker = { 4000.f, 0.f, 0.f };
    sim.mutable_bodies()[7].pos = marker;
    sim.mutable_bodies()[7].vel = { 0.f, 0.f, 0.f };
    sim.mutable_bodies()[7].acc = { 0.f, 0.f, 0.f };

    sim.integrate(1.f / 120.f);

    // Zero velocity and acceleration, so an integrator that saw the write leaves the
    // body at the marker. One that did not will report wherever the old body was.
    const nbody::Vector p = sim.bodies()[7].pos;
    INFO("pos after integrate: " << p.x << ", " << p.y << ", " << p.z);
    REQUIRE(std::abs(p.x - marker.x) < 1e-2f);
    REQUIRE(std::abs(p.y - marker.y) < 1e-2f);
    REQUIRE(std::abs(p.z - marker.z) < 1e-2f);
}

TEST_CASE("a mutation is visible through the read path", "[sim][gpu]")
{
    // Both body layouts, so the conversion protocol is checked for each rather than for
    // whichever one happens to be wired to this variant today.
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    nbody::Sim sim(v);
    seed_disk(sim, 256);
    sim.update(1.f / 120.f);

    const nbody::Vector marker = { 12.f, -34.f, 56.f };
    sim.mutable_bodies()[7].pos = marker;

    REQUIRE(sim.bodies()[7].pos.x == marker.x);
    REQUIRE(sim.bodies()[7].pos.y == marker.y);
    REQUIRE(sim.bodies()[7].pos.z == marker.z);
}

TEST_CASE("a mutation survives a variant switch", "[sim][gpu]")
{
    // Both body layouts, so the conversion protocol is checked for each rather than for
    // whichever one happens to be wired to this variant today.
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    nbody::Sim sim(v);
    seed_disk(sim, 256);
    sim.update(1.f / 120.f);

    const nbody::Vector marker = { 1.f, 2.f, 3.f };
    sim.mutable_bodies()[5].pos = marker;

    // The handoff must sync first, or the mutation is materialized over and lost.
    REQUIRE(sim.set_variant(nbody::Variant::CpuBarnesHut));

    REQUIRE(sim.bodies()[5].pos.x == marker.x);
    REQUIRE(sim.bodies()[5].pos.y == marker.y);
    REQUIRE(sim.bodies()[5].pos.z == marker.z);
}

TEST_CASE("gpu barnes-hut agrees with cpu barnes-hut", "[sim][gpu]")
{
    // Same algorithm on both backends, so differences are float ordering only. This is
    // a per-body bound, unlike the cross-algorithm comparison which can only be an
    // aggregate budget.
    const nbody::Variant gpu = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(gpu).name);
    if (skip_without_gpu(gpu))
        return;

    nbody::Sim sim;
    seed_disk(sim, 2048);

    sim.accelerate();
    const std::vector<nbody::Body> cpu = sim.bodies();

    REQUIRE(sim.set_variant(gpu));
    sim.accelerate();
    const std::vector<nbody::Body> dev = sim.bodies();

    // The two runs must actually have produced forces, or comparing them proves nothing.
    REQUIRE(cpu.size() == dev.size());
    REQUIRE(cpu[1].acc.size_sq() > 0.f);
    REQUIRE(dev[1].acc.size_sq() > 0.f);

    const float worst = max_relative_acc_error(cpu, dev);
    INFO("worst relative acceleration error: " << worst);
    REQUIRE(worst < 1e-2f);
}

TEST_CASE("gpu brute force agrees with cpu brute force", "[sim][gpu]")
{
    // Both are exact summations, so this is the strongest agreement check available.
    const nbody::Variant gpu = GENERATE(nbody::Variant::GpuBruteForce, nbody::Variant::GpuBruteForceSoA);
    INFO("variant: " << nbody::Sim::info(gpu).name);
    if (skip_without_gpu(gpu))
        return;

    nbody::Sim sim(nbody::Variant::CpuBruteForce);
    seed_disk(sim, 1024);

    sim.accelerate();
    const std::vector<nbody::Body> cpu = sim.bodies();

    REQUIRE(sim.set_variant(gpu));
    REQUIRE(sim.debug_node_count() == 0);   // the N^2 root-only tree is a binding placeholder
    sim.accelerate();
    const std::vector<nbody::Body> dev = sim.bodies();

    // The two runs must actually have produced forces, or comparing them proves nothing.
    REQUIRE(cpu.size() == dev.size());
    REQUIRE(cpu[1].acc.size_sq() > 0.f);
    REQUIRE(dev[1].acc.size_sq() > 0.f);

    const float worst = max_relative_acc_error(cpu, dev);
    INFO("worst relative acceleration error: " << worst);
    REQUIRE(worst < 1e-2f);
}

TEST_CASE("every variant honors the gravitational constant", "[sim][variant]")
{
    // The GPU push constant for G was left at its compile-time default, so set_gravity()
    // reached the CPU solvers but not the device. Scaling G must scale the resulting
    // accelerations on every backend.
    size_t tested = 0;
    for (const nbody::VariantInfo& info : nbody::Sim::variants())
    {
        if (!info.available)
            continue;

        INFO("variant: " << info.name);
        nbody::Sim sim(info.variant);
        REQUIRE(sim.variant() == info.variant);
        ++tested;

        seed_disk(sim, 256);
        sim.set_gravity(1.f);
        sim.accelerate();
        const float base = std::sqrt(sim.bodies()[1].acc.size_sq());
        REQUIRE(base > 0.f);

        sim.set_gravity(4.f);
        sim.accelerate();
        const float scaled = std::sqrt(sim.bodies()[1].acc.size_sq());

        INFO("base " << base << " scaled " << scaled);
        REQUIRE(std::abs(scaled / base - 4.f) < 0.01f);
    }
    REQUIRE(tested >= 2);
}

TEST_CASE("shrinking the body count does not corrupt memory", "[sim][gpu]")
{
    // Regression test: Buffer only ever grew, and read() copied the whole allocation
    // into the caller's vector, so shrinking the body count overran the heap. It was
    // guarded by an assert, which is compiled out in release.
    // Both body layouts, so the conversion protocol is checked for each rather than for
    // whichever one happens to be wired to this variant today.
    const nbody::Variant v = GENERATE(nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA);
    INFO("variant: " << nbody::Sim::info(v).name);
    if (skip_without_gpu(v))
        return;

    nbody::Sim sim(v);
    sim.mutable_bodies().resize(4096);
    nbody::util::cube(sim.mutable_bodies().begin(), sim.mutable_bodies().end());
    sim.update(1.f / 60.f);

    sim.mutable_bodies().resize(16);   // buffer stays sized for 4096
    sim.update(1.f / 60.f);

    REQUIRE(sim.bodies().size() == 16);
}

TEST_CASE("switching between gpu variants reuses one device", "[sim][gpu]")
{
    if (skip_without_gpu(nbody::Variant::GpuBarnesHut))
        return;
    if (!nbody::Sim::available(nbody::Variant::GpuBruteForce))
        SKIP("gpu brute force unavailable");

    nbody::Sim sim(nbody::Variant::GpuBarnesHut);
    seed_disk(sim, 512);
    sim.update(1.f / 120.f);

    // Mostly a smoke test that the cached device survives repeated switching; a device
    // rebuilt per switch would recompile shaders each time and be conspicuously slow.
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(sim.set_variant(nbody::Variant::GpuBruteForce));
        sim.update(1.f / 120.f);
        REQUIRE(sim.set_variant(nbody::Variant::GpuBarnesHut));
        sim.update(1.f / 120.f);
    }

    REQUIRE(sim.bodies().size() == 512);
}
