#include <memory>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "nbody/bhtree.h"
#include "nbody/sim.h"
#include "nbody/util.h"

// The two GPU body layouts, timed against each other. Hidden behind [.] so a normal test
// run does not pay for them:
//
//     nbody_tests "[benchmark]" --benchmark-samples 20
//
// Each pair runs the same simulation twice, once per layout. Two scenarios, because they
// answer different questions: "blind" steps without ever reading the bodies, which is what
// a headless run does and where lazy readback can pay off, and "reading" pulls the bodies
// back every step, which is what the demo does and where it cannot.
namespace
{
    constexpr float dt = 0.01f;

    std::unique_ptr<nbody::Sim> seeded(const nbody::Variant v, const size_t num)
    {
        auto sim = std::make_unique<nbody::Sim>();
        sim->mutable_bodies().resize(num);
        nbody::util::disk(sim->mutable_bodies().begin(), sim->mutable_bodies().end(), { .outer_radius = 100.f });
        REQUIRE(sim->set_variant(v));
        sim->update(dt);   // warm up: first step allocates and uploads
        return sim;
    }

    bool no_gpu()
    {
        if (nbody::Sim::available(nbody::Variant::GpuBarnesHut))
            return false;
        SKIP("no usable Vulkan compute device");
        return true;
    }
}

#define NBODY_BENCH_PAIR(label, aos, soa, num)                                          \
    BENCHMARK_ADVANCED(label " / interleaved / blind")(Catch::Benchmark::Chronometer m) \
    {                                                                                   \
        const auto sim = seeded(aos, num);                                              \
        m.measure([&](int) { sim->update(dt); });                                       \
    };                                                                                  \
    BENCHMARK_ADVANCED(label " / split / blind")(Catch::Benchmark::Chronometer m)       \
    {                                                                                   \
        const auto sim = seeded(soa, num);                                              \
        m.measure([&](int) { sim->update(dt); });                                       \
    };                                                                                  \
    BENCHMARK_ADVANCED(label " / interleaved / reading")(Catch::Benchmark::Chronometer m)\
    {                                                                                   \
        const auto sim = seeded(aos, num);                                              \
        m.measure([&](int) { sim->update(dt); return sim->bodies().size(); });          \
    };                                                                                  \
    BENCHMARK_ADVANCED(label " / split / reading")(Catch::Benchmark::Chronometer m)     \
    {                                                                                   \
        const auto sim = seeded(soa, num);                                              \
        m.measure([&](int) { sim->update(dt); return sim->bodies().size(); });          \
    }

TEST_CASE("gpu barnes-hut, 100k bodies", "[.][benchmark][gpu]")
{
    if (no_gpu()) return;
    NBODY_BENCH_PAIR("bh 100k", nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA, 100000);
}

TEST_CASE("gpu barnes-hut, 500k bodies", "[.][benchmark][gpu]")
{
    if (no_gpu()) return;
    NBODY_BENCH_PAIR("bh 500k", nbody::Variant::GpuBarnesHut, nbody::Variant::GpuBarnesHutSoA, 500000);
}

TEST_CASE("gpu brute force, 30k bodies", "[.][benchmark][gpu]")
{
    if (no_gpu()) return;
    NBODY_BENCH_PAIR("bf 30k", nbody::Variant::GpuBruteForce, nbody::Variant::GpuBruteForceSoA, 30000);
}

// The serial host-side tree build, which every barnes-hut step pays before the device is
// given anything to do. The ceiling on what any device-side or transfer-side change can win
// back in that mode, and the reason the two layouts look alike there.
TEST_CASE("host barnes-hut tree build", "[.][benchmark]")
{
    for (const size_t num : { size_t(100000), size_t(500000) })
    {
        std::vector<nbody::Body> bodies(num);
        nbody::util::disk(bodies.begin(), bodies.end(), { .outer_radius = 100.f });

        BENCHMARK_ADVANCED("tree build, " + std::to_string(num))(Catch::Benchmark::Chronometer m)
        {
            nbody::bh::Tree tree;
            m.measure([&](int)
            {
                tree.clear({ .size = 10000.f });
                tree.reserve(bodies.size() << 2);
                for (const nbody::Body& b : bodies)
                    tree.insert(b.pos, b.mass);
                return tree.nodes().size();
            });
        };
    }
}
