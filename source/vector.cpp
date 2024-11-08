#include "vector.h"

namespace nbody
{
    Vector operator*(float s, Vector v)
    {
        return { s * v.x, s * v.y, s * v.z };
    }
}
