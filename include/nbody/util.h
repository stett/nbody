#pragma once
#include <vector>
#include "body.h"
#include "constants.h"

namespace nbody
{
    namespace util
    {
        // compute the radius of a sphere given its mass and density
        float compute_radius(const float mass, const float density);

        // generate a disk shaped distribution of stars with a massive body in the center
        void disk(std::vector<Body>& bodies,
            size_t num = 4096,
            const Vector& center = {0,0,0},
            const Vector& vel = {0,0,0},
            const Vector& axis = {0,0,1},
            float inner_radius = 25.f,
            float outer_radius = 100.f,
            float central_mass = sagittarius_mass,
            float star_mass = solar_mass);
    }
}
