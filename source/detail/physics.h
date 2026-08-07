#pragma once
#include <cmath>
#include <cstddef>
#include "nbody/body.h"
#include "nbody/vector.h"

// The single host-side definition of the force law and the integrator, shared by every
// CPU solver and mirrored in GLSL (shaders/accelerate.comp and shaders/integrate.comp,
// over the declarations in shaders/include/common.glsl). Any change here must be made in
// both places or the CPU and GPU variants will silently disagree.
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
}
