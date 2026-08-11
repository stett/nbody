#pragma once
#include "vector.h"

namespace nbody
{
    struct Body
    {
        // statics
        Vector pos = { 0,0,0 };
        float radius = 0;

        // kinematics
        Vector vel = { 0,0,0 };
        float mass = 0;

        // dynamics
        Vector acc = { 0,0,0 };
        float __pad = 0;
    };
}