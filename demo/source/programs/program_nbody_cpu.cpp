#include "program_nbody_cpu.h"
#include "nbody/util.h"
#include "nbody/constants.h"
using namespace ci;

void ProgramNBodyCPU::setup()
{
    // create shader program for rendering stars
    particle_shader = gl::GlslProg::create(
#include "cpu_particles_vert.glsl"
        ,
#include "cpu_particles_frag.glsl"
        ,
#include "cpu_particles_geom.glsl"
    );

    // create shader program for rendering structures
    structure_shader = gl::GlslProg::create(
#include "cpu_structures_vert.glsl"
            ,
#include "cpu_structures_frag.glsl"
            ,
#include "cpu_structures_geom.glsl"
    );

    // create a vbos for getting stars onto gpu
    particles_vbo = gl::Vbo::create(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    structures_vbo = gl::Vbo::create(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    reset();
}

void ProgramNBodyCPU::reset()
{
    // create the default disk
    sim.bodies.clear();
    nbody::util::disk(sim.bodies);
    sim.accelerate();
}

void ProgramNBodyCPU::evolve(float dt)
{
    sim.update(dt);
}

void ProgramNBodyCPU::buffer(bool particles, bool structures)
{
    if (particles) {
        // update intermediate buffer of particle data
        particles_vbo_data.resize(sim.bodies.size() * 8);
        for (size_t i = 0; i < sim.bodies.size(); i++) {
            nbody::Body &body = sim.bodies[i];
            particles_vbo_data[(i * 4) + 0] = (body.pos.x);
            particles_vbo_data[(i * 4) + 1] = (body.pos.y);
            particles_vbo_data[(i * 4) + 2] = (body.pos.z);
            particles_vbo_data[(i * 4) + 3] = (body.radius);
        }

        // update gpu particle data buffer
        particles_vbo->bufferData(particles_vbo_data.size() * sizeof(float), particles_vbo_data.data(), GL_DYNAMIC_DRAW);
    }

    if (structures)
    {
        // Update the CPU buffer for tree data
        // Create and populate VBO containing bounds data
        const size_t num_nodes = sim.acc_tree.nodes().size();
        structures_vbo_data.clear();
        structures_vbo_data.reserve(7 * num_nodes);
        float max_potential = 0;
        float avg_potential = 0;
        const float num_nodes_inv = 1.f / float(num_nodes);
        for (const nbody::bh::Node& node : sim.acc_tree.nodes())
        {
            const vec3 half = vec3(node.bounds.size * .5f);
            const vec3 bounds_center = vec3(node.bounds.center.x, node.bounds.center.y, node.bounds.center.z);
            for (size_t i = 0; i < 3; ++i)
                structures_vbo_data.emplace_back(bounds_center[i] - half[i]);
            for (size_t i = 0; i < 3; ++i)
                structures_vbo_data.emplace_back(bounds_center[i] + half[i]);

            // get gravitational potential at the center of this node and store it in GPU data
            const nbody::Vector& center = node.bounds.center;
            float potential = 0;
            sim.acc_tree.apply(center, [&potential, &center](const nbody::bh::Node& node) {
                const nbody::Vector delta = node.com - center;
                const float delta_sq = delta.size_sq();
                if (delta_sq > std::numeric_limits<float>::epsilon())
                    potential += node.mass / delta_sq;
            });
            max_potential = std::max(max_potential, potential);
            avg_potential += potential * num_nodes_inv;
            structures_vbo_data.emplace_back(potential);
        }
        const float max_potential_inv = max_potential > std::numeric_limits<float>::epsilon() ? 1.f / max_potential : 0;
        const float avg_potential_inv = avg_potential > std::numeric_limits<float>::epsilon() ? 1.f / avg_potential : 0;
        for (size_t i = 0; i < structures_vbo_data.size(); i += 7)
        {
            float& potential = structures_vbo_data[i+6];
            potential = std::min(1.f, potential * avg_potential_inv);

        }
        structures_vbo->bufferData(structures_vbo_data.size() * sizeof(float), structures_vbo_data.data(), GL_DYNAMIC_DRAW);

    }
}

void ProgramNBodyCPU::draw(bool particles, bool structures)
{
    if (particles) {
        gl::ScopedGlslProg glsl_scope(particle_shader);
        particles_vbo->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        gl::vertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (3 * sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei) sim.bodies.size());
        particles_vbo->unbind();
        gl::setDefaultShaderVars();
    }

    if (structures)
    {
        gl::ScopedGlslProg glsl_scope(structure_shader);
        gl::ScopedDepth depth_scope(false);
        structures_vbo->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::enableVertexAttribArray(2);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(0*sizeof(float)));
        gl::vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(3*sizeof(float)));
        gl::vertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei)sim.acc_tree.nodes().size());
        structures_vbo->unbind();
        gl::setDefaultShaderVars();
    }
}

float ProgramNBodyCPU::size() const
{
    return sim.size;
}
