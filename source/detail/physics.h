#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include "hwy/highway.h"
#include "nbody/body.h"
#include "nbody/vector.h"

// The single host-side definition of the force law and the integrator, shared by every
// CPU solver and mirrored in GLSL (shaders/accelerate.comp and shaders/integrate.comp,
// over the declarations in shaders/include/common.glsl). Any change here must be made in
// both places or the CPU and GPU variants will silently disagree.
//
// The two are not bit-identical, and are not meant to be. The shader divides by an
// approximate reciprocal square root, which the hardware has an instruction for, where
// gravity() below uses an exact sqrt and divide. That difference turns out not to be what
// separates the two -- summing the same forces in a different order dominates it, and the
// cross-check in tests/source/test_gpu.cpp reports the same worst-case error either way.
// The agreement that check asserts is the real contract between them.
namespace nbody::detail
{
    // Gravitational acceleration on a body at `pos` due to mass `src_mass` at `src_pos`.
    inline Vector gravity(
        const Vector& pos,
        const float radius,
        const Vector& src_pos,
        const float src_mass,
        const float G)
    {
        const Vector delta = src_pos - pos;
        const float delta_sq = delta.size_sq();
        const float radii_sq = radius * radius;

        // If we're too close, don't apply a force.
        // NOTE: this condition no longer needed if we have collisions
        //
        // Must not be strict: radius defaults to 0, so a body at zero distance from
        // itself would fall through and divide by sqrt(0)*0, giving NaN. Only brute force
        // can skip self by index, so the zero case is handled here for every caller.
        if (delta_sq <= radii_sq)
            return { 0, 0, 0 };

        return G * src_mass * delta / (std::sqrt(delta_sq) * delta_sq);
    }

    // Wrap a coordinate into [-size/2, +size/2], making space a 3-torus.
    //
    // The double fmod is needed because std::fmod keeps the sign of the dividend, so a
    // single call can still return a negative value. GLSL's mod() is already
    // non-negative for a positive divisor, making the second application redundant
    // there, but the shader keeps the same form so the two read identically.
    inline float wrap(const float x, const float size)
    {
        if (size <= 0.f)
            return x;   // a zero/negative world size would otherwise produce NaN

        const float half = size * .5f;
        return std::fmod(std::fmod(x + half, size) + size, size) - half;
    }

    // Semi-implicit euler, which is well behaved for gravitational forces.
    inline void integrate_euler(Body& body, const float dt, const float size, const bool do_wrap)
    {
        body.vel += body.acc * dt;
        body.pos += body.vel * dt;

        if (!do_wrap)
            return;

        for (size_t i = 0; i < 3; ++i)
            body.pos[i] = wrap(body.pos[i], size);
    }

    namespace simd
    {
        namespace hn = hwy::HWY_NAMESPACE;
        using std::span;
        using std::vector;

        template <class D, class V = hn::Vec<D>>
        HWY_INLINE V wrap(D d, V x, V size)
        {
            const V half = hn::Mul(size, hn::Set(d, 0.5f));
            const V shifted = hn::Add(x, half);
            const V q = hn::Floor(hn::Div(shifted, size));
            const V wrapped = hn::NegMulAdd(q, size, shifted);  // shifted - q*size
            return hn::Sub(wrapped, half);
        }

        // Semi-implicit euler over the split body arrays, one SIMD vector of bodies at a
        // time. Same step as the scalar integrate_euler above, and the two have to agree:
        // a variant is chosen at run time, so the same scene must integrate the same way
        // whichever one is driving it.
        //
        // Every span is indexed by body, and the caller is free to hand over any subrange
        // of its arrays -- one worker's block of a parallel_blocks partition, typically.
        // So nothing here may assume the pointers are vector-aligned or that the count is
        // a whole number of vectors:
        //
        //  - the stores are StoreU. An aligned Store faults outright on a block whose
        //    first body is not vector-aligned, which is most of them: block boundaries
        //    come from dividing the body count by the worker count.
        //  - the final partial vector goes through LoadN/StoreN, which touch only the
        //    lanes that exist. A full-width store there would run past the end of the
        //    block, over bodies another worker has in flight, and past the end of the
        //    allocation itself for the last block -- corrupting the heap rather than
        //    faulting, so the crash lands somewhere else entirely, later.
        inline void integrate_euler(
            span<float> x, span<float> y, span<float> z,
            span<float> vx, span<float> vy, span<float> vz,
            span<float> ax, span<float> ay, span<float> az,
            const float dt, const float size, const bool do_wrap)
        {
            const size_t n = x.size();
            assert(y.size() == n && z.size() == n);
            assert(vx.size() == n && vy.size() == n && vz.size() == n);
            assert(ax.size() == n && ay.size() == n && az.size() == n);

            const hn::ScalableTag<float> d;
            const size_t num_lanes = hn::Lanes(d);

            const auto vec_dt = hn::Set(d, dt);
            const auto vec_size = hn::Set(d, size);

            // A world size of zero or less has no interior to wrap into, and dividing by
            // it would turn every position into a NaN. Same guard as scalar wrap().
            const bool wrap_positions = do_wrap && size > 0.f;

            // One vector of bodies at `i`. Templated on whether this is the tail rather
            // than branching on the count, so the full-width loop keeps plain loads and
            // stores; `count` is read only on the tail path.
            const auto step = [&]<bool tail>(const size_t i, const size_t count)
            {
                const auto load = [&](const span<float> v)
                {
                    if constexpr (tail)
                        return hn::LoadN(d, v.data() + i, count);
                    else
                        return hn::LoadU(d, v.data() + i);
                };

                const auto store = [&](const auto v, const span<float> out)
                {
                    if constexpr (tail)
                        hn::StoreN(v, d, out.data() + i, count);
                    else
                        hn::StoreU(v, d, out.data() + i);
                };

                // Semi-implicit: velocity first, and then position from the *new*
                // velocity. Stepping position on the old velocity instead would make this
                // explicit euler, which is a different (and less stable) integrator than
                // the one every other variant runs.
                const auto vec_vx = hn::MulAdd(load(ax), vec_dt, load(vx));
                const auto vec_vy = hn::MulAdd(load(ay), vec_dt, load(vy));
                const auto vec_vz = hn::MulAdd(load(az), vec_dt, load(vz));

                auto vec_x = hn::MulAdd(vec_vx, vec_dt, load(x));
                auto vec_y = hn::MulAdd(vec_vy, vec_dt, load(y));
                auto vec_z = hn::MulAdd(vec_vz, vec_dt, load(z));

                if (wrap_positions)
                {
                    vec_x = wrap(d, vec_x, vec_size);
                    vec_y = wrap(d, vec_y, vec_size);
                    vec_z = wrap(d, vec_z, vec_size);
                }

                store(vec_vx, vx);
                store(vec_vy, vy);
                store(vec_vz, vz);

                store(vec_x, x);
                store(vec_y, y);
                store(vec_z, z);
            };

            size_t i = 0;
            for (; i + num_lanes <= n; i += num_lanes)
                step.template operator()<false>(i, num_lanes);
            if (i < n)
                step.template operator()<true>(i, n - i);
        }
    }
}
