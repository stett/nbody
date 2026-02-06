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

        struct DiskArgs
        {
            const Vector center = {0,0,0};
            const Vector vel = {0,0,0};
            const Vector axis = {0,0,1};
            float inner_radius = 0.f;
            float outer_radius = 250.f;
            float thickness = 1.f;
            float central_mass = sagittarius_mass;
            float star_mass = solar_mass;
        };

        // generate a disk shaped distribution of stars with a massive body in the center
        void disk(
            std::vector<Body>::iterator begin,
            std::vector<Body>::iterator end,
            DiskArgs args = DiskArgs());

        struct CubeArgs
        {
            const Vector center = {0,0,0};
            const float size = 500;
            const Vector vel = {0,0,0};
            float star_mass = solar_mass;
        };

        void cube(
            std::vector<Body>::iterator begin,
            std::vector<Body>::iterator end,
            CubeArgs args = CubeArgs());
    }
}
