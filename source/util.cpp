#include <cmath>
#include <random>
#include <algorithm>
#include "util.h"
#include "constants.h"
#include "bhtree.h"

float nbody::util::compute_radius(const float mass, const float density)
{
    return std::pow((3.f * pi * mass) / (4.f * density), 1.f/3.f);
}

void nbody::util::disk(std::vector<Body>& bodies, size_t num,
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
    bodies.reserve(bodies.size() + num);
    if (num == 0) { return; }

    // get iterator to where our first new body will be
    auto bodies_begin = bodies.end();

    // create the central gravitational body
    const float center_radius = compute_radius(central_mass, star_density);
    bodies.emplace_back(Body{ .mass=central_mass, .radius=center_radius, .pos=center, .vel=vel });
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
        const Vector star_coord0 = (std::sin(angle) * coord0) + (std::cos(angle) * coord1);
        const Vector star_coord1 = cross(axis, star_coord0);
        const Vector star_pos = center + star_coord0 * dist;
        const Vector star_lin_vel = star_coord1;//vel + star_coord1 * std::sqrt(G * central_mass / (dist));
        const float star_radius = compute_radius(star_mass, star_density);
        bodies.push_back({ .mass=star_mass, .radius=star_radius, .pos=star_pos, .vel=star_lin_vel });
        tree.insert({ star_pos.x, star_pos.y, star_pos.z }, star_mass);
    }

    // adjust velocites of bodies to have approximately circular orbits around central mass
    for (auto body = bodies_begin; body != bodies.end(); ++body)
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
    std::sort(bodies_begin, bodies.end(), [&center](Body& a, Body&b) { return length2(a.pos - center) < length2(b.pos - center); });

    // adjust velocities of each body so that they are
    float mass = 0;
    for (auto body = bodies_begin; body != bodies.end(); ++body)
    {
        mass += body->mass;
        const float dist_sq = length2(body->pos - center);
        if (dist_sq < std::numeric_limits<float>::epsilon())
            continue;
        const float speed = sqrt(G * mass / sqrt(dist_sq));
        body->vel = vel + (body->vel * speed);
    }
    */
}
