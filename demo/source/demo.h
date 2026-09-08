#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#include <string>
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#include "nbody/sim.h"
#include "nbody/util.h"

using glm::vec2;
using glm::vec3;
using glm::ivec2;
using glm::ivec3;
using glm::mat4;
using glm::quat;
using ci::app::MouseEvent;
using ci::CameraPersp;
using ci::gl::GlslProgRef;
using ci::gl::VboRef;

namespace nbody {
    class Demo : public ci::app::App {
    public:
        void setup() override;
        void resize() override;
        void update() override;
        void draw() override;
        void mouseMove(MouseEvent event) override;
        void mouseDrag(MouseEvent event) override;
        void mouseWheel(MouseEvent event) override;
        void mouseDown(MouseEvent event) override;
        void mouseUp(MouseEvent event) override;

    private:

        void spawn_galaxy(uint32_t num, nbody::util::DiskArgs args);
        void spawn_cube(uint32_t num, nbody::util::CubeArgs args);
        void setup_sim_data();
        void update_gpu_data();

        // helpers
        vec3 homogeneous_to_world(const vec3& homo) const;
        void mouse_ray(vec3& out_ray_origin, vec3& out_ray_direction) const;
        vec3 mouse_plane_pos(const vec3& plane_point, const vec3& plane_axis) const;
        vec3 mouse_world_pos(const float dist_from_eye = 500) const;

        // nbody sim;
        nbody::Sim sim;

        // time
        double time = 0;
        float delta_time = 0;

        // Frames and seconds since the displayed rate was last recomputed. The instantaneous
        // 1/delta_time is unreadable -- it changes every frame and swings with any hitch.
        float hz_accum_time = 0;
        int hz_accum_frames = 0;
        float hz_display = 0;

        float sim_dt = 1.f / 60.f;
        float sim_dt_accum = 0;
        float sim_dt_scale = 1.f;

        // shaders
        GlslProgRef bounds_shader;
        GlslProgRef particle_shader;

        // gpu data caches. the per-element float counts must match the vertex attribute
        // layouts set up in setup(), and are what the draw calls derive their vertex
        // counts from -- deriving those from the sim instead would draw bodies that have
        // not been uploaded yet, since bodies can be spawned from an input handler.
        static constexpr size_t floats_per_particle = 4;   // pos.xyz, radius
        static constexpr size_t floats_per_bound = 7;      // min.xyz, max.xyz, weight
        std::vector<float> gpu_particle_data;
        VboRef vbo_particles;
        std::vector<float> gpu_bounds_data;
        VboRef vbo_bounds;

        // Scratch for Sim::write_debug_nodes(). Solvers do not own their debug nodes -- they
        // advertise a count and fill a span -- so one buffer here serves every variant the
        // sim is switched through, and re-fills rather than re-allocates each frame.
        std::vector<nbody::DebugNode> debug_nodes;

        // settings
        bool setup_complete = false;
        bool run_simulation = false;
        bool draw_bh_bounds = false;
        bool draw_axes = false;
        size_t target_num_elems = 1 << 18;

        // why the last variant switch failed; empty when it succeeded
        std::string variant_error;

        // camera
        CameraPersp camera;
        vec3 cam_focus = glm::vec3(0);
        vec3 cam_focus_target = glm::vec3(0);
        vec2 cam_angles = glm::vec2(M_PI * .5, 0);
        vec2 cam_target_angles = glm::vec2(M_PI * .5, 0);
        float cam_dist = 500;
        float cam_target_dist = 500;

        // mouse
        ivec2 mouse_pos;
        ivec2 mouse_delta;
        bool mouse_drag = false;
        vec3 mouse_world_drag_origin;
    };
}
