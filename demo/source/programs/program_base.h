#pragma once
#include "cinder/gl/gl.h"

class ProgramBase
{
public:
    virtual ~ProgramBase() = default;
    virtual void setup() = 0;
    virtual void reset() = 0;
    virtual void evolve(float dt) = 0;
    virtual void buffer(bool particles, bool structures) = 0;
    virtual void draw(bool particles, bool structures) = 0;
    virtual float size() const = 0;
};
