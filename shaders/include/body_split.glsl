// Three parallel arrays grouped by how often each field crosses the bus. Must match the
// like-named structs in source/gpu.h field for field. pos and mass pair up because the n^2
// inner loop wants exactly those two, and a vec3 costs 16 bytes whatever follows it.
#ifndef NBODY_BODY_SPLIT_GLSL
#define NBODY_BODY_SPLIT_GLSL

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

#endif // NBODY_BODY_SPLIT_GLSL
