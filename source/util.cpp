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
    const Vector& center,
    const Vector& vel,
    const Vector& axis,
    float inner_radius,
    float outer_radius,
    float central_mass,
    float star_mass)
{
    // temp tree for velocity calculation
    bh::Tree tree({.size=2.f * inner_radius, .center={ center.x, center.y, center.z }});

    // make sure we have space in the bodies array
    const size_t num = end - begin;
    if (num == 0) { return; }

    // create the central gravitational body
    const float center_radius = compute_radius(central_mass, star_density);
    auto body = begin;
    *(body++) = Body{ .mass=central_mass, .radius=center_radius, .pos=center, .vel=vel };
    tree.insert({ center.x, center.y, center.z }, central_mass);

    // make sure none of the stars are spawned inside the center
    outer_radius = std::max(outer_radius, center_radius);
    inner_radius = std::max(inner_radius, center_radius);

    // create the random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1);

    // get the coordinate vectors of the axis
    // https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
    const Vector coord0 = {
        std::copysign(axis.z, axis.x),
        std::copysign(axis.z, axis.y),
        -std::copysign(axis.x, axis.z) - std::copysign(axis.y, axis.z)
    };
    const Vector coord1 = cross(axis, coord0);

    // add the stars
    for (size_t i = 1; i < num; ++i)
    {
        const float t = float(i) / float(num-1);
        const float angle = t * 2.f * pi;
        const float rng = distribution(generator);
        const float dist = inner_radius + (sqrt(rng) * (outer_radius - inner_radius));
        const Vector star_coord0 = ((std::sin(angle) * coord0) + (std::cos(angle) * coord1)).norm();
        const Vector star_coord1 = (cross(axis, star_coord0)).norm();
        const Vector star_pos = center + star_coord0 * dist;
        const Vector star_lin_vel = star_coord1;
        const float star_radius = compute_radius(star_mass, star_density);
        *(body++) = { .mass=star_mass, .radius=star_radius, .pos=star_pos, .vel=star_lin_vel };
        tree.insert({ star_pos.x, star_pos.y, star_pos.z }, star_mass);
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
        body->vel = vel + (body->vel * speed);
    }

    /*
    // sort bodies by distance to sun
    //std::sort(bodies_begin, bodies.end(), [&center](Body& a, Body&b) { return (a.pos - center).size_sq() < (b.pos - center).size_sq(); });
    std::sort(begin, end, [&center](Body& a, Body&b) { return (a.pos - center).size_sq() < (b.pos - center).size_sq(); });

    // adjust velocities of each body so that they are
    float mass = 0;
    for (body = begin; body != end; ++body)
    {
        mass += body->mass;
        const float dist_sq = (body->pos - center).size_sq();
        if (dist_sq < std::numeric_limits<float>::epsilon())
            continue;
        const float speed = sqrt(G * mass / sqrt(dist_sq));
        body->vel = vel + (body->vel * speed);
    }
     */
}
