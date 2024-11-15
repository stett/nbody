#include "cinder/Log.h"
#include "cinder/CinderImGui.h"
#include "cinder/Display.h"
#include "demo_app.h"
#include "draw_utils.h"
#include "programs/program_nbody_cpu.h"

void NBodyDemoApp::setup()
{
    ImGui::Initialize();

    // setup window
    setWindowSize(1024, 1024);

    // setup clock
    time = getElapsedSeconds();

    // setup program to run
    program = std::make_unique<ProgramNBodyCPU>();
    program->setup();
    program->buffer(show_stars, show_structures);

    // setup camera
    cam_dist = program->size();
    cam_target_dist = cam_dist;
}

void NBodyDemoApp::update()
{
    update_time();
    update_camera();
    update_ui();
    update_program();
}

void NBodyDemoApp::update_time()
{
    // Update dt
    const double new_time = getElapsedSeconds();
    delta_time = float(new_time - time);
    time = new_time;
}

void NBodyDemoApp::update_camera()
{
    // TODO: focus on selected element if there is one
    cam_focus_target = vec3(0);
    const float snap = 2.f;
    cam_focus += (cam_focus_target - cam_focus) * std::min(delta_time * snap, 1.f);
    cam_angles += (cam_target_angles - cam_angles) * std::min(delta_time * snap, 1.f);
    cam_dist += (cam_target_dist - cam_dist) * std::min(delta_time * snap, 1.f);
    const glm::mat4 m0 = glm::rotate(cam_angles[0], glm::vec3(0, 1, 0));
    const glm::mat4 m1 = glm::rotate(cam_angles[1], glm::vec3(0, 0, 1));
    const glm::mat4 m2 = glm::translate(vec3(cam_dist, 0, 0));
    // NOTE: The cam_focus bit is really not right
    const glm::vec3 cam_pos = m0 * m1 * m2 * glm::translate(cam_focus) * vec4(0,0,0, 1);
    cam.lookAt(cam_pos, cam_focus, vec3(0,1,0));
}

void NBodyDemoApp::update_ui()
{
    // For high-dpi displays
    const float content_scale = getDisplay()->getContentScale();
    ImGuiIO &io = ImGui::GetIO();
    io.DisplayFramebufferScale.x = content_scale;
    io.DisplayFramebufferScale.y = content_scale;

    ImGui::Begin("settings");
    if (ImGui::Checkbox("run program", &run_program)) { }
    if (ImGui::Button("reset program")) { program->reset(); program->buffer(show_stars, show_structures); }
    if (ImGui::Checkbox("show stars", &show_stars)) { program->buffer(show_stars, show_structures); }
    if (ImGui::Checkbox("show structures", &show_structures)) { program->buffer(show_stars, show_structures); }
    if (ImGui::Checkbox("show axes", &show_axes)) { }
    ImGui::End();
}

void NBodyDemoApp::update_program()
{
    if (!run_program)
        return;

    const float sim_delta_time = 1.f / float(sim_hertz);
    if (sim_duration > sim_delta_time)
    {
        sim_duration = 0;
    }

    sim_accum += delta_time;
    while (sim_accum > sim_delta_time && sim_duration <= sim_delta_time)
    {
        sim_accum -= sim_delta_time;
        const double pre_sim_time = getElapsedSeconds();
        program->evolve(sim_delta_time);
        const double post_sim_time = getElapsedSeconds();
        sim_duration = float(post_sim_time - pre_sim_time);
    }
    program->buffer(show_stars, show_structures);
}

void NBodyDemoApp::draw()
{
    gl::clear(ColorA(0, 0, 0, 1), true);
    gl::setMatrices(cam);

    if (show_axes)
    {
        util::draw_axes(program->size());
    }

    program->draw(show_stars, show_structures);
}

void NBodyDemoApp::resize()
{
    cam.setPerspective(60, getWindowAspectRatio(), 1, 1e5);
    gl::setMatrices(cam);
}

void NBodyDemoApp::mouseMove(MouseEvent event)
{
    const glm::ivec2& new_mouse_pos = event.getPos();
    mouse_pos = new_mouse_pos;
}

void NBodyDemoApp::mouseDrag(MouseEvent event)
{
    const glm::ivec2& new_mouse_pos = event.getPos();
    mouse_delta = new_mouse_pos - mouse_pos;
    mouse_pos = new_mouse_pos;

    cam_target_angles[0] -= mouse_delta.x * .01f;
    cam_target_angles[1] += mouse_delta.y * .01f;

    if (cam_target_angles[1] < -M_PI * .4f) { cam_target_angles[1] = -M_PI * .4f; }
    if (cam_target_angles[1] > M_PI * .4f) { cam_target_angles[1] = M_PI * .4f; }
}

void NBodyDemoApp::mouseWheel(MouseEvent event)
{
    cam_target_dist -= event.getWheelIncrement() * 5.f;
    if (cam_target_dist < 1.f) { cam_target_dist = 1.f; }
}