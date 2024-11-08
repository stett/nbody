#include "vector.h"

namespace nbody
{
    Vector operator*(float s, Vector v)
    {
        return { s * v.x, s * v.y, s * v.z };
    }

    float dot(const Vector& a, const Vector& b)
    {
        return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
    }

    Vector cross(const Vector& a, const Vector& b)
    {
        return {
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x)
        };
    }
}
