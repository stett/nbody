// One array of Body, exactly as the host holds them. The baseline body_split.glsl is
// measured against.
#ifndef NBODY_BODY_INTERLEAVED_GLSL
#define NBODY_BODY_INTERLEAVED_GLSL

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

layout(std430, binding = 0) buffer Bodies {
    Body bodies[];
};

#endif // NBODY_BODY_INTERLEAVED_GLSL
