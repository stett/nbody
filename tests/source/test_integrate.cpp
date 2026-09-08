#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "detail/physics.h"
#include "nbody/sim.h"

// detail::simd::integrate_euler is the split-array (SoA) form of the integrator every
// other variant runs one Body at a time. Two things about how it is called make the tail
// of a run the interesting case, and both of them crashed the demo:
//
//  - the caller hands over a subrange of its arrays, one worker's block of a
//    parallel_blocks partition, so neither the start pointer nor the count is a multiple
//    of the SIMD vector width, and
//  - the last block's subrange ends where the allocation does, so a full-width store of
//    a partial vector writes off the end of the heap block.
//
// The tests below pin both, plus agreement with the scalar step.
namespace
{
    namespace hn = hwy::HWY_NAMESPACE;
    namespace detail = nbody::detail;

    size_t lanes()
    {
        const hn::ScalableTag<float> d;
        return hn::Lanes(d);
    }

    constexpr float world_size = 1000.f;
    constexpr float dt = 0.5f;

    // Deterministic bodies whose every coordinate is a multiple of 0.125 and smaller in
    // magnitude than 1024, so the whole step is exact in float and the scalar and SIMD
    // results can be compared without a tolerance for the arithmetic itself. The .125
    // offset also keeps positions clear of the wrap boundary at +/-500, where the two
    // wrap formulations (fmod against floor-division) could legitimately disagree by a
    // whole period.
    nbody::Body make_body(const size_t i)
    {
        // int64_t, not size_t: the subtraction that centers each range on zero has to be
        // signed, or it wraps around into ~1e19 and the "bodies" are nonsense.
        const auto spread = [i](const int64_t stride, const int64_t period, const int64_t k)
        {
            const int64_t v = static_cast<int64_t>(i) * stride + k * 311;
            return static_cast<float>((v % period) - (period / 2));
        };
        const auto coord = [&](const int64_t k) { return spread(137, 900, k) + .125f; };
        const auto vel = [&](const int64_t k) { return spread(61, 400, k); };
        const auto acc = [&](const int64_t k) { return spread(29, 20, k); };
        return nbody::Body{
            .pos = { coord(0), coord(1), coord(2) },
            .radius = 0.f,
            .vel = { vel(0), vel(1), vel(2) },
            .mass = 1.f,
            .acc = { acc(0), acc(1), acc(2) },
        };
    }

    // The split arrays, laid out with sentinel floats on both sides of the range the
    // integrator is allowed to touch.
    //
    // pad_front is odd, which is the point: operator new hands back at least
    // 16-byte-aligned storage, so a range starting an odd number of floats into it cannot
    // be vector-aligned for any vector of 16 bytes or more, and an aligned store to it
    // faults. pad_back is a whole vector wide even on AVX-512, so an overrun lands in the
    // sentinels and is reported rather than corrupting the heap and crashing elsewhere.
    struct SplitArrays
    {
        static constexpr size_t pad_front = 5;
        static constexpr size_t pad_back = 64;
        static constexpr float sentinel = -12345.f;

        explicit SplitArrays(const size_t count)
            : n(count)
        {
            for (std::vector<float>* a : all())
                a->assign(pad_front + count + pad_back, sentinel);
        }

        std::span<float> range(std::vector<float>& a) const { return std::span(a).subspan(pad_front, n); }

        void integrate(const bool do_wrap)
        {
            detail::simd::integrate_euler(
                range(x), range(y), range(z),
                range(vx), range(vy), range(vz),
                range(ax), range(ay), range(az),
                dt, world_size, do_wrap);
        }

        std::vector<std::vector<float>*> all()
        {
            return { &x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az };
        }

        size_t n;
        std::vector<float> x, y, z, vx, vy, vz, ax, ay, az;
    };

    static_assert(SplitArrays::pad_front % 2 == 1, "an odd offset is what guarantees the range is misaligned");

    SplitArrays scatter(const std::span<const nbody::Body> bodies)
    {
        SplitArrays a(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i)
        {
            const size_t j = SplitArrays::pad_front + i;
            a.x[j] = bodies[i].pos.x;  a.y[j] = bodies[i].pos.y;  a.z[j] = bodies[i].pos.z;
            a.vx[j] = bodies[i].vel.x; a.vy[j] = bodies[i].vel.y; a.vz[j] = bodies[i].vel.z;
            a.ax[j] = bodies[i].acc.x; a.ay[j] = bodies[i].acc.y; a.az[j] = bodies[i].acc.z;
        }
        return a;
    }
}

TEST_CASE("the SIMD integrator agrees with the scalar one", "[physics][simd]")
{
    // Counts either side of every plausible vector width (4, 8 and 16 floats), so at
    // least some of them leave a partial vector at the end.
    for (size_t n = 1; n <= 2 * 16 + 3; ++n)
    {
        for (const bool do_wrap : { false, true })
        {
            INFO("bodies: " << n << ", wrap: " << do_wrap << ", lanes: " << lanes());

            std::vector<nbody::Body> bodies(n);
            for (size_t i = 0; i < n; ++i)
                bodies[i] = make_body(i);

            SplitArrays a = scatter(bodies);
            a.integrate(do_wrap);

            for (size_t i = 0; i < n; ++i)
            {
                nbody::Body expected = bodies[i];
                detail::integrate_euler(expected, dt, world_size, do_wrap);

                INFO("body " << i);
                const size_t j = SplitArrays::pad_front + i;
                REQUIRE(a.vx[j] == Catch::Approx(expected.vel.x).margin(1e-3f));
                REQUIRE(a.vy[j] == Catch::Approx(expected.vel.y).margin(1e-3f));
                REQUIRE(a.vz[j] == Catch::Approx(expected.vel.z).margin(1e-3f));
                REQUIRE(a.x[j] == Catch::Approx(expected.pos.x).margin(1e-3f));
                REQUIRE(a.y[j] == Catch::Approx(expected.pos.y).margin(1e-3f));
                REQUIRE(a.z[j] == Catch::Approx(expected.pos.z).margin(1e-3f));
            }
        }
    }
}

TEST_CASE("the SIMD integrator writes nothing outside the range it was given", "[physics][simd]")
{
    // The crash this pins: with the count rounded up to a whole number of vectors, the
    // last store of a range whose length is not a multiple of the vector width lands up
    // to lanes()-1 floats past the end. Here that is caught by the trailing sentinels;
    // in the solver it was the end of the allocation, or another worker's bodies.
    for (size_t n = 1; n <= 2 * 16 + 3; ++n)
    {
        INFO("bodies: " << n << ", lanes: " << lanes());

        std::vector<nbody::Body> bodies(n);
        for (size_t i = 0; i < n; ++i)
            bodies[i] = make_body(i);

        SplitArrays a = scatter(bodies);
        a.integrate(true);

        for (const std::vector<float>* array : a.all())
        {
            for (size_t k = 0; k < SplitArrays::pad_front; ++k)
            {
                INFO("leading guard slot " << k);
                REQUIRE((*array)[k] == SplitArrays::sentinel);
            }
            for (size_t k = 0; k < SplitArrays::pad_back; ++k)
            {
                INFO("trailing guard slot " << k);
                REQUIRE((*array)[SplitArrays::pad_front + n + k] == SplitArrays::sentinel);
            }
        }
    }
}

TEST_CASE("the SIMD variant integrates a body count that is not a whole number of vectors", "[sim][simd]")
{
    // The same tail, reached the way the demo reaches it: through the solver, over a
    // parallel_blocks partition, so the block boundaries land wherever the worker count
    // puts them rather than on a vector boundary. With gravity switched off this is the
    // integrator alone, and it has to match the reference variant body for body.
    //
    // 1021 is prime, so neither the body count nor most of the partition of it is a whole
    // number of vectors of any width.
    constexpr size_t num_bodies = 1021;

    // A random cloud rather than anything regular: a lattice of positions puts enough
    // bodies on a line to drive the morton octree build into a malformed tree, which is
    // its own (pre-existing, variant-independent) problem and not what this is testing.
    std::vector<nbody::Body> initial(num_bodies);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> pos(-450.f, 450.f);
    std::uniform_real_distribution<float> vel(-60.f, 60.f);
    for (nbody::Body& body : initial)
        body = nbody::Body{
            .pos = { pos(rng), pos(rng), pos(rng) },
            .radius = 0.f,
            .vel = { vel(rng), vel(rng), vel(rng) },
            .mass = 1.f,
        };

    const auto run = [&initial](const nbody::Variant variant)
    {
        nbody::Sim sim(variant);
        REQUIRE(sim.variant() == variant);
        sim.set_size(world_size);
        sim.set_gravity(0.f);   // isolate the integrator
        sim.mutable_bodies() = initial;

        // enough steps that bodies cross the world boundary and wrap
        for (int step = 0; step < 8; ++step)
            sim.update(dt);

        return sim.bodies();
    };

    const std::vector<nbody::Body> expected = run(nbody::Variant::CpuBarnesHutMorton);
    const std::vector<nbody::Body> actual = run(nbody::Variant::SimdBarnesHutMorton);

    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        INFO("body " << i);
        REQUIRE(actual[i].pos.x == Catch::Approx(expected[i].pos.x).margin(1e-2f));
        REQUIRE(actual[i].pos.y == Catch::Approx(expected[i].pos.y).margin(1e-2f));
        REQUIRE(actual[i].pos.z == Catch::Approx(expected[i].pos.z).margin(1e-2f));
        REQUIRE(actual[i].vel.x == Catch::Approx(expected[i].vel.x).margin(1e-2f));
        REQUIRE(actual[i].vel.y == Catch::Approx(expected[i].vel.y).margin(1e-2f));
        REQUIRE(actual[i].vel.z == Catch::Approx(expected[i].vel.z).margin(1e-2f));
    }
}
