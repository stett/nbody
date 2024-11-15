#pragma once
#include <vector>
#include "program_base.h"
#include "nbody/sim.h"

class ProgramNBodyCPU : public ProgramBase
{
public:
    ~ProgramNBodyCPU() override = default;
    void setup() override;
    void reset() override;
    void evolve(float dt) override;
    void buffer(bool particles, bool structures) override;
    void draw(bool particles, bool structures) override;
    virtual float size() const override;

private:

    // internal sim
    nbody::Sim sim;

    // shaders
    ci::gl::GlslProgRef particle_shader;
    ci::gl::GlslProgRef structure_shader;

    // buffers
    std::vector<float> particles_vbo_data;
    ci::gl::VboRef particles_vbo;
    std::vector<float> structures_vbo_data;
    ci::gl::VboRef structures_vbo;
};
