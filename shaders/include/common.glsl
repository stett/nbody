// Shared declarations for the compute shaders. Included after #version, so this file
// deliberately does not declare one. The CMake shader step rebuilds every stage when it
// changes.
#ifndef NBODY_COMMON_GLSL
#define NBODY_COMMON_GLSL

layout(local_size_x = 256) in;

// The bodies, split into parallel arrays grouped by how often each field crosses the bus.
// Each struct must match the like-named one in source/gpu.h field for field, and the
// bindings must match make_descriptor_set_layout(). A stage may leave a binding
// undeclared -- integrate never reads the nodes at binding 3.
//
// pos and mass share an array because the n^2 inner loop wants exactly those two: a vec3
// occupies 16 bytes here whatever follows it, so pairing them costs nothing and saves the
// loop a second array to stream.
struct BodyPosMass
{
    vec3 pos;
    float mass;
};

struct BodyVelRadius
{
    vec3 vel;
    float radius;
};

struct BodyAcc
{
    vec3 acc;
    float __pad;
};

layout(std430, binding = 0) buffer PosMassBuffer {
    BodyPosMass pos_mass[];
};

layout(std430, binding = 1) buffer VelRadiusBuffer {
    BodyVelRadius vel_radius[];
};

layout(std430, binding = 2) buffer AccBuffer {
    BodyAcc accs[];
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
