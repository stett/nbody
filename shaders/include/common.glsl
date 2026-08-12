// Shared declarations for the compute shaders. Included after #version, so this file
// deliberately does not declare one. The CMake shader step rebuilds every stage when it
// changes.
//
// Everything here is independent of the body layout, so that the two being compared share
// one copy of the force law and the tree traversal. Define NBODY_NODE_BINDING before
// including to also get the node buffer and the traversal that reads it.
#ifndef NBODY_COMMON_GLSL
#define NBODY_COMMON_GLSL

layout(local_size_x = 256) in;

// must match nbody::bh::Node (include/nbody/bhtree.h) field for field
struct Node
{
    vec3 bounds_center;
    float bounds_size;
    vec3 com;
    float mass;
    uint next;
    uint children;
    uint __pad0;
    uint __pad1;
};

// must match nbody::PushConstants (source/gpu.h) field for field
layout(push_constant) uniform PushConstants {
    float dt;
    float theta;
    float G;
    int num_bodies;
    int num_nodes;
    int mode;
    float size;
    int wrap;
} pc;

const int N2 = 0;
const int NLogN = 1;

vec3 accelerate(vec3 body_pos, float body_radius, vec3 node_pos, float node_mass)
{
    const vec3 delta = node_pos - body_pos;
    const float delta_sq = dot(delta, delta);
    const float radii_sq = body_radius * body_radius;

    // if we're too close, don't apply a force
    // NOTE: this condition no longer needed if we have collisions
    //
    // Mirrors nbody::detail::gravity (source/detail/physics.h), including the non-strict
    // comparison: radius defaults to 0, and neither the n^2 loop nor the barnes-hut
    // traversal can skip this body by index, so a strict test would divide by sqrt(0)*0
    // and give NaN.
    if (delta_sq <= radii_sq)
        return vec3(0);

    // Compute force of gravity, as G * m * d / |d|^3.
    //
    // Deliberately not the sqrt-and-divide the host uses. Both of those are
    // correctly-rounded in GLSL and neither is a single instruction: each expands to a
    // hardware approximation followed by a refinement sequence, so the denominator alone
    // costs more than everything else here put together. inversesqrt is permitted to be
    // approximate, which is what lets it compile to the bare instruction.
    //
    // The approximation costs nothing measurable in agreement with the host: the worst
    // relative error the cross-check in tests/source/test_gpu.cpp reports is unchanged by
    // this (6.4e-6 before, 6.2e-6 after), because what separates the two paths is the order
    // they sum forces in, not how either one takes a square root.
    const float inv_dist = inversesqrt(delta_sq);
    const float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
    return (pc.G * node_mass * inv_dist_cubed) * delta;
}

#ifdef NBODY_NODE_BINDING

layout(std430, binding = NBODY_NODE_BINDING) buffer Nodes {
    Node nodes[];
};

// Reads nothing but the tree, so it is the same code for either body layout.
vec3 accelerate_nlogn(vec3 pos, float radius)
{
    const float theta_sq = pc.theta * pc.theta;
    vec3 acc = vec3(0);

    uint i = 0;
    do
    {
        // If the node is empty, skip it
        if (nodes[i].mass == 0)
        {
            i = nodes[i].next;
            continue;
        }

        // If the node has no children, apply function directly, and increment
        // i by one to indicate
        if (nodes[i].children == 0)
        {
            acc += accelerate(pos, radius, nodes[i].com, nodes[i].mass);
            i = nodes[i].next;
            continue;
        }

        // If the node is far enough away apply the node function
        const float node_size = nodes[i].bounds_size;
        const float node_size_sq = node_size * node_size;
        const vec3 delta = nodes[i].com - pos;
        const float dist_sq = dot(delta, delta);
        if (dist_sq > node_size_sq * theta_sq)
        {
            acc += accelerate(pos, radius, nodes[i].com, nodes[i].mass);
            i = nodes[i].next;
            continue;
        }

        // If we need to drill down, start looking at the node's children
        i = nodes[i].children;
    } while (0 < i && i < pc.num_nodes);

    return acc;
}

#endif // NBODY_NODE_BINDING

#endif // NBODY_COMMON_GLSL
