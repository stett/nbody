#pragma once
#include "vector.h"

namespace nbody
{
    struct Body
    {
        /*
        float mass = 0;
        float radius = 0;
        Vector pos = { 0,0,0 };
        Vector vel = { 0,0,0 };
        Vector acc = { 0,0,0 };
         */

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