#pragma once
#include <memory>
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#include "programs/program_base.h"

using namespace ci;
using namespace ci::app;

class NBodyDemoApp : public App
{
public:
    ~NBodyDemoApp() override = default;
    void setup() override;
    void update() override;
    void draw() override;
    void resize() override;
    void mouseMove(MouseEvent event) override;
    void mouseDrag(MouseEvent event) override;
    void mouseWheel(MouseEvent event) override;

private:

    void update_time();
    void update_camera();
    void update_ui();
    void update_program();

private:

    // time
    double time = 0;
    float delta_time = 0;
    float sim_duration = 0;
    uint32_t sim_hertz = 60;
    float sim_accum = 0;

    // mouse
    glm::ivec2 mouse_pos;
    glm::ivec2 mouse_delta;

    // camera
    CameraPersp	cam;
    glm::vec3 cam_focus = glm::vec3(0);
    glm::vec3 cam_focus_target = glm::vec3(0);
    glm::vec2 cam_angles = glm::vec2(M_PI * .5, 0);
    glm::vec2 cam_target_angles = glm::vec2(M_PI * .5, 0);
    float cam_dist = 50;
    float cam_target_dist = 50;

    // gui options
    bool run_program = false;
    bool show_stars = true;
    bool show_structures = false;
    bool show_axes = false;

    // simulation
    std::unique_ptr<ProgramBase> program;
};
