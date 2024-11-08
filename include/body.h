#pragma once
#include "vector.h"

namespace nbody
{
    struct Body
    {
        float mass = 0;
        float radius = 0;
        Vector pos = { 0,0,0 };
        Vector vel = { 0,0,0 };
        Vector acc = { 0,0,0 };
    };
}