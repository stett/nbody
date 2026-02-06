#include <cmath>
#include <random>
#include <algorithm>
#include "nbody/util.h"
#include "nbody/constants.h"
#include "nbody/bhtree.h"

float nbody::util::compute_radius(const float mass, const float density)
{
    return std::pow((3.f * pi * mass) / (4.f * density), 1.f/3.f);
}

void nbody::util::disk(
    std::vector<Body>::iterator begin,
    std::vector<Body>::iterator end,
    DiskArgs args)
{
    // temp tree for velocity calculation
    bh::Tree tree({.size=2.f * args.outer_radius, .center={ args.center.x, args.center.y, args.center.z }});

    // make sure we have space in the bodies array
    const size_t num = end - begin;
    if (num == 0) { return; }

    // create the central gravitational body
    const float center_radius = compute_radius(args.central_mass, star_density);
    auto body = begin;
    *(body++) = Body{ .mass=args.central_mass, .radius=center_radius, .pos=args.center };
    tree.insert({ args.center.x, args.center.y, args.center.z }, args.central_mass);

    // make sure none of the stars are spawned inside the center
    args.outer_radius = std::max(args.outer_radius, center_radius);
    args.inner_radius = std::max(args.inner_radius, center_radius);

    // create the random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform_distribution(0, 1);
    std::normal_distribution<float> gaussian_distribution(0, .5);

    // get the coordinate vectors of the axis
    // https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
    const Vector coord0 = {
        std::copysign(args.axis.z, args.axis.x),
        std::copysign(args.axis.z, args.axis.y),
        -std::copysign(args.axis.x, args.axis.z) - std::copysign(args.axis.y, args.axis.z)
    };
    const Vector coord1 = cross(args.axis, coord0);

    // add the stars
    for (size_t i = 1; i < num; ++i)
    {
        const float t = float(i) / float(num-1);
        const float angle = t * 2.f * pi;

        // distance from origin, perpendicular to axis
        const float dist = args.inner_radius + (sqrt(uniform_distribution(generator)) * (args.outer_radius - args.inner_radius));

        // displacement from plane along axis
        const float disp = ((args.outer_radius - dist) / args.outer_radius) * (args.thickness * center_radius) * gaussian_distribution(generator);

        // star velocity, just used for directionality
        const Vector star_coord0 = ((std::sin(angle) * coord0) + (std::cos(angle) * coord1)).norm();
        const Vector star_coord1 = (cross(args.axis, star_coord0)).norm();
        const Vector star_pos = args.center + (star_coord0 * dist) + (args.axis * disp);
        const Vector star_lin_vel = star_coord1;

        // given the mass and density, find the radius
        const float star_radius = compute_radius(args.star_mass, star_density);

        // add the star to the array
        *(body++) = { .mass=args.star_mass, .radius=star_radius, .pos=star_pos, .vel=star_lin_vel };

        // add the star to the temporary tree
        tree.insert({ star_pos.x, star_pos.y, star_pos.z }, args.star_mass);
    }

    // adjust velocites of bodies to have approximately circular orbits around central mass
    for (body = begin; body != end; ++body)
    {
        Vector com = {0,0,0};
        float mass = 0;
        tree.apply(body->pos, [&com, &mass](const bh::Node& node)
        {
            com = ((com * mass) + (node.com * node.mass)) / (mass + node.mass);
            mass += node.mass;
        });

        const float dist = (com - body->pos).size();
        const float speed = std::sqrt(G * mass / dist);
        body->vel = args.vel + (body->vel * speed);
    }
}

void nbody::util::cube(
    std::vector<Body>::iterator begin,
    std::vector<Body>::iterator end,
    CubeArgs args)
{
    // make sure we have space in the bodies array
    const size_t num = end - begin;
    if (num == 0) { return; }
    auto body = begin;

    // create the random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform_distribution(-.5 * args.size, .5 * args.size);

    // add the stars
    for (size_t i = 0; i < num; ++i)
    {
        const Vector star_pos =
        {
            uniform_distribution(generator),
            uniform_distribution(generator),
            uniform_distribution(generator)
        };
        const float star_radius = compute_radius(args.star_mass, star_density);
        *(body++) = { .mass=args.star_mass, .radius=star_radius, .pos=star_pos, .vel=args.vel };
    }
}
