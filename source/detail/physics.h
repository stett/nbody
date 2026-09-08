#pragma once
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

        void integrate_euler(
            span<float> x, span<float> y, span<float> z,
            span<float> vx, span<float> vy, span<float> vz,
            span<float> ax, span<float> ay, span<float> az,
            const float dt, const float size, const bool do_wrap)
        {
            const hn::ScalableTag<float> d;
            const size_t num_lanes = hn::Lanes(d);

            size_t i = 0;
            for (; i < x.size(); i += num_lanes)
            {
                // load the simd vectors for position, velocity, and acceleration

                const auto vec_x = hn::LoadU(d, x.data() + i);
                const auto vec_y = hn::LoadU(d, y.data() + i);
                const auto vec_z = hn::LoadU(d, z.data() + i);

                const auto vec_vx = hn::LoadU(d, vx.data() + i);
                const auto vec_vy = hn::LoadU(d, vy.data() + i);
                const auto vec_vz = hn::LoadU(d, vz.data() + i);

                const auto vec_ax = hn::LoadU(d, ax.data() + i);
                const auto vec_ay = hn::LoadU(d, ay.data() + i);
                const auto vec_az = hn::LoadU(d, az.data() + i);

                // semi implicit step, update velocity first, then position

                hn::Store(hn::MulAdd(vec_ax, hn::Set(d, dt), vec_vx), d, vx.data() + i);
                hn::Store(hn::MulAdd(vec_ay, hn::Set(d, dt), vec_vy), d, vy.data() + i);
                hn::Store(hn::MulAdd(vec_az, hn::Set(d, dt), vec_vz), d, vz.data() + i);

                hn::Store(hn::MulAdd(vec_vx, hn::Set(d, dt), vec_x), d, x.data() + i);
                hn::Store(hn::MulAdd(vec_vy, hn::Set(d, dt), vec_y), d, y.data() + i);
                hn::Store(hn::MulAdd(vec_vz, hn::Set(d, dt), vec_z), d, z.data() + i);

                if (do_wrap)
                {
                    // toroidal wrap

                    const auto vec_size = hn::Set(d, size);
                    const auto vec_half = hn::Set(d, size * 0.5f);

                    wrap(d, vec_x, vec_size);
                    wrap(d, vec_y, vec_size);
                    wrap(d, vec_z, vec_size);
                }
            }
        }
    }
}
