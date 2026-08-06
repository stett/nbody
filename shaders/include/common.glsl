// Shared declarations for the compute shaders. Included after #version, so this file
// deliberately does not declare one.
//
// This was previously the `glsl_common` string literal in source/shaders.h, concatenated
// onto each shader at run time. It is a real include now, so the CMake shader step rebuilds
// every stage when this file changes.
#ifndef NBODY_COMMON_GLSL
#define NBODY_COMMON_GLSL

layout(local_size_x = 256) in;

// must match nbody::Body (include/nbody/body.h) field for field
struct Body
{
    vec3 pos;
    float radius;
    vec3 vel;
    float mass;
    vec3 acc;
    float __pad;
};

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

#endif // NBODY_COMMON_GLSL
