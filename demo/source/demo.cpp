#include "demo.h"
#include <cmath>
#include <random>
#include "cinder/CinderImGui.h"
#include "nbody/profile.h"

using namespace ci;
using namespace ci::gl;

namespace
{
    // imgui runs entirely in framebuffer pixels here: CinderImGui feeds it toPixels()'d
    // display and mouse coordinates and DisplayFramebufferScale stays at 1 (see setup()).
    // So the default metrics -- a 13px font, 8px padding -- are physical pixels, and the
    // whole UI comes out half-size on a 2x display. Scale the style and the font by the
    // window's content scale to get point-sized widgets back. A conventional display
    // reports a scale of 1, where this is a no-op.
    //
    // Re-checked every frame rather than once at startup, so dragging the window onto a
    // display of a different density rescales the UI, and so we never have to ask for the
    // content scale during setup(), where some backends have yet to compute it.
    // Returns true when the scale changed, so the caller can re-fit windows that are
    // holding a size measured at the old one.
    bool sync_imgui_content_scale(float content_scale)
    {
        // Captured before anything has scaled it: the first call is after ImGui::Initialize().
        static const ImGuiStyle unscaled_style = ImGui::GetStyle();
        static float applied_scale = 1.f;

        if (! std::isfinite(content_scale) || content_scale <= 0.f || content_scale == applied_scale)
            return false;

        // ScaleAllSizes() scales the style in place and truncates as it goes, so rescale the
        // pristine copy rather than compounding a correction onto the current values.
        ImGuiStyle& style = ImGui::GetStyle();
        style = unscaled_style;
        style.ScaleAllSizes(content_scale);

        // Fonts are baked on demand in imgui 1.92 and the GL backend advertises
        // ImGuiBackendFlags_RendererHasTextures, so this re-rasterizes the atlas at the
        // larger size instead of filtering a 13px bitmap up.
        style.FontScaleDpi = content_scale;

        applied_scale = content_scale;
        return true;
    }
}

void nbody::Demo::setup()
{
    NBODY_PROFILE_THREAD("main");
    NBODY_PROFILE_ZONE();
    // Cinder already reports io.DisplaySize in pixels via toPixels(), so DisplayFramebufferScale has
    // to stay at 1. Scaling it by the display's content scale made imgui's GL backend size its
    // framebuffer at DisplaySize * 2, and it derives scissor rects from that height -- so every clip
    // rect landed above the real 1024px framebuffer and the whole UI was scissored away.
    ImGui::Initialize();

    // Prefer the GPU when one is usable. Best effort: a false return just leaves the
    // sim on its default CPU variant, and the combo shows why.
    sim.set_variant(nbody::Variant::CpuBarnesHutMorton);

    setWindowSize(1024, 1024);

    //rng.seed(42);

    time = getElapsedSeconds();

    {
        static const std::string shader_vert(
                "#version 410\n"
                "layout (location=0) in vec3 v_min_pos;"
                "layout (location=1) in vec3 v_max_pos;"
                "layout (location=2) in float v_potential;"
                "layout (location=0) out vec3 g_min_pos;"
                "layout (location=1) out vec3 g_max_pos;"
                "layout (location=2) out float g_potential;"
                "void main(void)"
                "{"
                "    g_min_pos = v_min_pos;"
                "    g_max_pos = v_max_pos;"
                "    g_potential = v_potential;"
                "}");
        static const std::string shader_geom(
                "#version 410\n"
                "uniform mat4	ciModelViewProjection;"
                "layout (points) in;"
                "layout (line_strip, max_vertices = 18) out;"
                "layout (location=0) in vec3 g_min_pos[];"
                "layout (location=1) in vec3 g_max_pos[];"
                "layout (location=2) in float g_potential[];"
                "layout (location=0) out float f_potential;"
                "void main(void)"
                "{"
                "    f_potential = g_potential[0];"
                "    vec3 aa = g_min_pos[0];"
                "    vec3 bb = g_max_pos[0];"
                "    vec4 v[8] = vec4[8]("
                "       ciModelViewProjection * vec4(aa.x, aa.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, aa.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, bb.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, bb.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, aa.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, aa.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, bb.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, bb.y, bb.z, 1));"
                ""
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[0+4]; EmitVertex();"
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[1]; EmitVertex();"
                "    gl_Position = v[1+4]; EmitVertex();"
                "    gl_Position = v[1]; EmitVertex();"
                "    gl_Position = v[2]; EmitVertex();"
                "    gl_Position = v[2+4]; EmitVertex();"
                "    gl_Position = v[2]; EmitVertex();"
                "    gl_Position = v[3]; EmitVertex();"
                "    gl_Position = v[3+4]; EmitVertex();"
                "    gl_Position = v[3]; EmitVertex();"
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[4]; EmitVertex();"
                "    gl_Position = v[5]; EmitVertex();"
                "    gl_Position = v[6]; EmitVertex();"
                "    gl_Position = v[7]; EmitVertex();"
                "    gl_Position = v[4]; EmitVertex();"
                "    EndPrimitive();"
                "}"
        );
        static const std::string shader_frag(
                "#version 410\n"
                "out vec4 		oColor;"
                "layout (location=0) in float f_potential;"
                "void main(void)"
                "{"
                "    float percent = f_potential;"
                "    float a = percent;"
                "    vec4 potential_color = vec4(1-(a*.5), a, 0, .1 + .15 * a);"
                "    oColor = potential_color;"
                "}");
        bounds_shader = gl::GlslProg::create(shader_vert, shader_frag, shader_geom);
    }

    {
        static const std::string shader_vert(
                "#version 410\n"
                "uniform mat4 ciModelViewProjection;"
                "layout (location=0) in vec3 v_pos;"
                "layout (location=1) in float v_rad;"
                //"layout (location=2) in vec4 v_rot;"
                "layout (location=0) out float g_rad;"
                "void main(void)"
                "{"
                "    gl_Position = ciModelViewProjection * vec4(v_pos, 1);"
                "    g_rad = v_rad;"
                "}");
        static const std::string shader_geom(
                "#version 410\n"
                "#define pi 3.1415926535897932384626433832795\n"
                "uniform mat4 ciProjectionMatrix;"
                //"uniform float radius;"
                "layout (points) in;"
                "layout (triangle_strip, max_vertices = 38) out;"
                "layout (location=0) in float g_rad[];"
                "void main(void)"
                "{"
                "    float aspect = ciProjectionMatrix[1][1] / ciProjectionMatrix[0][0];"
                //"    vec2 scale = vec2(1, aspect);"
                "    vec2 scale = vec2(1, aspect) * ciProjectionMatrix[0][0];"
                "    float r = g_rad[0];"
                //"    float r = radius;"
                "    float n = 20;"
                "    gl_Position = gl_in[0].gl_Position + vec4(r*scale.x,0, 0, 0);"
                "    EmitVertex();"
                "    for (float i = 1; i < n; ++i) {"
                "        float t = (i / (n-1)) * 2 * pi;"
                "        vec2 xy = r * vec2(cos(t), sin(t));"
                "        gl_Position = gl_in[0].gl_Position + vec4(xy.x*scale.x, xy.y*scale.y, 0, 0);"
                "        EmitVertex();"
                "        gl_Position = gl_in[0].gl_Position;"
                "        EmitVertex();"
                "    }"
                "    EndPrimitive();"
                "}");
        static const std::string shader_frag(
                "#version 150\n"
                "out vec4 oColor;"
                "void main(void)"
                "{"
                "    oColor = vec4(1,1,1,1);"
                "}");
        particle_shader = gl::GlslProg::create(shader_vert, shader_frag, shader_geom);
    }

    setup_sim_data();

    // Create and populate VBOs containing particle and bounds data
    vbo_particles = gl::Vbo::create(GL_ARRAY_BUFFER,
        sim.bodies().size() * floats_per_particle * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    vbo_bounds = gl::Vbo::create(GL_ARRAY_BUFFER,
        sim.debug_node_count() * floats_per_bound * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    update_gpu_data();

    gl::enableDepthWrite();
    gl::enableDepthRead();

    setup_complete = true;
}

void nbody::Demo::spawn_galaxy(uint32_t num, nbody::util::DiskArgs args)
{
    NBODY_PROFILE_ZONE();
    std::vector<nbody::Body>& bodies = sim.mutable_bodies();
    bodies.resize(bodies.size() + num);
    nbody::util::disk(bodies.end() - num, bodies.end(), args);
}

void nbody::Demo::spawn_cube(uint32_t num, nbody::util::CubeArgs args)
{
    NBODY_PROFILE_ZONE();
    std::vector<nbody::Body>& bodies = sim.mutable_bodies();
    bodies.resize(bodies.size() + num);
    nbody::util::cube(bodies.end() - num, bodies.end(), args);
}

void nbody::Demo::setup_sim_data()
{
    NBODY_PROFILE_ZONE();

    // remove all bodies from the sim
    sim.mutable_bodies().clear();

    // add a disk galaxy at the origin
    // designators must follow DiskArgs' member order (center, vel, axis)
    //spawn_galaxy(target_num_elems, { .center={0,0,0}, .vel={0,0,0}, .axis={0,0,1} });

    // add two colliding disk galaxies
    spawn_galaxy(target_num_elems * .5f, { .center={-250,0,0}, .vel={0,40,0},  .axis={0, 1.f/sqrtf(2.f),  1.f/sqrtf(2.f)} });
    spawn_galaxy(target_num_elems * .5f, { .center={250,0,0},  .vel={0,-40,0}, .axis={0, -1.f/sqrtf(2.f), 1.f/sqrtf(2.f)} });

    // this forces an update to the acceleration structure, which is
    // needed if we want to update the structure rendering
    sim.accelerate();
}

void nbody::Demo::update_gpu_data()
{
    // Runs every frame whether or not the simulation advanced.
    NBODY_PROFILE_ZONE();

    // Update the CPU buffer for particle data. Read-only: bind const so this per-frame
    // loop never takes mutable access.
    const std::vector<nbody::Body>& bodies = sim.bodies();
    gpu_particle_data.resize(bodies.size() * floats_per_particle);
    for (size_t i = 0; i < bodies.size(); i++)
    {
        const nbody::Body& body = bodies[i];
        gpu_particle_data[(i * floats_per_particle) + 0] = (body.pos.x);
        gpu_particle_data[(i * floats_per_particle) + 1] = (body.pos.y);
        gpu_particle_data[(i * floats_per_particle) + 2] = (body.pos.z);
        gpu_particle_data[(i * floats_per_particle) + 3] = (body.radius);
    }

    // Update the GPU buffer
    vbo_particles->bufferData(gpu_particle_data.size() * sizeof(float), gpu_particle_data.data(), GL_DYNAMIC_DRAW);

    // Not every variant builds an acceleration structure, so tolerate there being none --
    // and only ask for one when it is going to be drawn. Nothing is cached solver-side, so
    // filling this costs a barnes-hut traversal per node every time it is asked for.
    size_t num_debug_nodes = 0;
    if (draw_bh_bounds)
    {
        // resize() on a member, so this allocates on the first frame and re-fills after
        debug_nodes.resize(sim.debug_node_count());
        num_debug_nodes = sim.write_debug_nodes(debug_nodes);
    }

    if (num_debug_nodes > 0)
    {
        // Update the CPU buffer for tree data
        // Create and populate VBO containing bounds data
        gpu_bounds_data.clear();
        gpu_bounds_data.reserve(floats_per_bound * num_debug_nodes);
        float avg_weight = 0;
        const float num_nodes_inv = 1.f / float(num_debug_nodes);
        for (size_t i_node = 0; i_node < num_debug_nodes; ++i_node)
        {
            const nbody::DebugNode& node = debug_nodes[i_node];

            // DebugNode::size is a full edge length, matching nbody::Bounds -- see
            // nbody/debug.h, where the opposite convention next door is spelled out
            const vec3 half = vec3(node.size * .5f);
            const vec3 bounds_center = vec3(node.center.x, node.center.y, node.center.z);
            for (size_t i = 0; i < 3; ++i)
                gpu_bounds_data.emplace_back(bounds_center[i] - half[i]);
            for (size_t i = 0; i < 3; ++i)
                gpu_bounds_data.emplace_back(bounds_center[i] + half[i]);

            // The solver hands over a raw per-node scalar; normalizing it into a colour is
            // this side's policy, so the average is accumulated here and folded in below.
            avg_weight += node.weight * num_nodes_inv;
            gpu_bounds_data.emplace_back(node.weight);
        }
        const float avg_weight_inv = avg_weight > std::numeric_limits<float>::epsilon() ? 1.f / avg_weight : 0;
        for (size_t i = 0; i < gpu_bounds_data.size(); i += floats_per_bound)
        {
            float& weight = gpu_bounds_data[i+6];
            weight = std::min(1.f, weight * avg_weight_inv);
        }
        vbo_bounds->bufferData(gpu_bounds_data.size() * sizeof(float), gpu_bounds_data.data(), GL_DYNAMIC_DRAW);
    }
    else
    {
        // draw() derives its vertex count from this buffer, so it has to be emptied rather
        // than left holding the last tree a variant happened to build.
        gpu_bounds_data.clear();
    }
}

void nbody::Demo::resize()
{
#if ! defined(CINDER_MSW)
    // Cinder derives the viewport from glfwGetFramebufferSize() in
    // RendererImplGlfwGl::defaultResize(). During startup on macOS that can report the
    // window as still retina-backed, before GLFW settles the NSView for a non-high-density
    // app, so a 1024pt window bakes in a 2048px viewport and nothing re-runs the query
    // afterwards, leaving the scene off-centre until the first manual resize. Set the
    // viewport from the window size ourselves.
    //
    // Not on MSW, where that renderer already sets it correctly and calling toPixels()
    // here is actively harmful: setWindowSize() in setup() dispatches resize()
    // synchronously, and WindowImplMsw has its size before its display, so getSize() is
    // correct while getContentScale() is still uninitialised. Their product is nonsense,
    // and the same toPixels() path feeds imgui's DisplaySize, so the next NewFrame()
    // fails "Invalid DisplaySize value!" and aborts.
    const ivec2 size_px = ci::app::toPixels(getWindowSize());
    gl::viewport(0, 0, size_px.x, size_px.y);
#endif

    camera.setPerspective(60, getWindowAspectRatio(), 1, 1e5 );
    gl::setMatrices(camera );
}

void nbody::Demo::update()
{
    NBODY_PROFILE_ZONE();

    // Which solver the frame ran on, so a capture can be told apart from one taken with a
    // different variant selected.
    NBODY_PROFILE_ZONE_TEXT(nbody::Sim::info(sim.variant()).name);

    // setWindowSize() in setup() dispatches a resize synchronously on some backends,
    // which drives update()/draw() before the shaders and VBOs below exist.
    if (! setup_complete)
        return;

    bool one_tick = false;

    // Update gui
    {
        NBODY_PROFILE_ZONE_NAMED("imgui");

        // Before the first Begin(), so the window's initial auto-fit uses scaled metrics.
        // A zero size asks imgui to re-fit: any size restored from imgui.ini, or left over
        // from another display, was measured against a different font size and would clip.
        if (sync_imgui_content_scale(getWindowContentScale()))
            ImGui::SetNextWindowSize(ImVec2(0, 0));

        ImGui::Begin("Settings");
        ImGui::Text("framerate: %dhz", int(hz_display + .5f));
        // Polled every frame, wireframe or not, so it has to stay off write_debug_nodes()
        // and its per-node traversal. The percentage this used to show was size/capacity
        // against build_tree()'s reserve(count << 2) -- a bh::Tree tuning number that no
        // longer has a home in a neutral interface, and one the Tracy plot in
        // detail/tree.h already reports.
        if (const size_t num_nodes = sim.debug_node_count())
            ImGui::Text("tree nodes: %d", (int)num_nodes);
        else
            ImGui::Text("tree nodes: n/a");
        ImGui::Checkbox("run simulation", &run_simulation);
        int sim_hz = int(ceil(1.f / sim_dt));
        if (ImGui::SliderInt("sim hz", &sim_hz, 1.f, 144.f)) { sim_dt = 1.f / float(sim_hz); }
        if (ImGui::SliderFloat("sim t-scale", &sim_dt_scale, .0f, 1.f)) { }
        if (ImGui::Button("tick simulation")) { one_tick = true; }
        if (ImGui::Button("reset simulation")) { setup_sim_data(); }

        // simulation variant
        {
            const nbody::VariantInfo& current = nbody::Sim::info(sim.variant());
            if (ImGui::BeginCombo("variant", current.name))
            {
                for (const nbody::VariantInfo& info : nbody::Sim::variants())
                {
                    // Latch this before the push. `info` refers into the live variant
                    // table, and a failed switch below marks that same entry
                    // unavailable, so re-reading info.available for the pop would
                    // underflow the style stack.
                    const bool greyed = !info.available;

                    // Grey out unavailable entries by hand rather than with
                    // BeginDisabled, which the ImGui bundled with Cinder predates. They
                    // stay clickable on purpose: set_variant refuses safely and reports
                    // why, so clicking a greyed entry explains itself.
                    if (greyed)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1.f));

                    if (ImGui::Selectable(info.name, info.variant == sim.variant()))
                    {
                        if (sim.set_variant(info.variant))
                        {
                            variant_error.clear();

                            // The incoming solver adopts the bodies but not an acceleration
                            // structure, so build one now. Without this the tree wireframe
                            // and the node-capacity readout stay empty until the next step,
                            // which never comes while the simulation is paused.
                            sim.accelerate();
                        }
                        else
                        {
                            variant_error = sim.last_error();
                        }
                    }

                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", greyed ? info.unavailable_reason.c_str() : info.description);

                    if (greyed)
                        ImGui::PopStyleColor();
                }
                ImGui::EndCombo();
            }
            if (!variant_error.empty())
                ImGui::TextColored(ImVec4(1.f, .4f, .4f, 1.f), "%s", variant_error.c_str());
        }

        bool wrap_space = sim.wrap();
        if (ImGui::Checkbox("wrap space", &wrap_space)) { sim.set_wrap(wrap_space); }

        if (ImGui::Checkbox("show gravity tree", &draw_bh_bounds)) { }
        if (ImGui::Checkbox("show coordinate axes", &draw_axes)) { }

        ImGui::Text("# stars: %d", int(target_num_elems));
        int log_num_stars = std::log2(target_num_elems);
        if (ImGui::SliderInt("log2(# stars)", &log_num_stars, 0, 20))
        {
            target_num_elems = std::pow(2, log_num_stars);
            setup_sim_data();
        }
        ImGui::End();
    }

    // Update dt
    double new_time = getElapsedSeconds();
    delta_time = float(new_time - time);
    time = new_time;

    // Roll the displayed framerate up over a second at a time. Averaged over the window
    // rather than sampled at the end of it, so one slow frame cannot stand in for all of them.
    hz_accum_time += delta_time;
    ++hz_accum_frames;
    if (hz_accum_time >= 1.f)
    {
        hz_display = float(hz_accum_frames) / hz_accum_time;
        hz_accum_time = 0;
        hz_accum_frames = 0;
    }
    else if (hz_display == 0 && hz_accum_time > 0)
    {
        // Nothing to roll yet: show the partial window rather than 0hz for the first second.
        hz_display = float(hz_accum_frames) / hz_accum_time;
    }

    // Update camera
    {
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
        camera.lookAt(cam_pos, cam_focus, vec3(0, 1, 0));
    }

    // if the last delta tick was too big, stop running the sim
    if (run_simulation && delta_time > .5)
        run_simulation = false;

    if (one_tick)
    {
        NBODY_PROFILE_ZONE_NAMED("sim step");
        one_tick = false;
        sim.update(sim_dt);
    }
    else {
        // if running simulation, tick it
        if (run_simulation) {
            NBODY_PROFILE_ZONE_NAMED("sim step");
            sim_dt_accum += delta_time;
            size_t sim_steps = 1;
            while (sim_dt_accum > sim_dt && sim_steps-- > 0) {
                sim_dt_accum -= sim_dt;
                sim.update(sim_dt * sim_dt_scale);
            }
        }
    }

    // Update GPU data
    update_gpu_data();
}

void nbody::Demo::mouseMove(MouseEvent event)
{
    mouse_pos = event.getPos();
}

void nbody::Demo::mouseDrag(MouseEvent event)
{
    const glm::ivec2 new_mouse_pos = event.getPos();
    mouse_delta = new_mouse_pos - mouse_pos;
    mouse_pos = new_mouse_pos;

    if (!mouse_drag)
    {
        cam_target_angles[0] -= mouse_delta.x * .01f;
        cam_target_angles[1] += mouse_delta.y * .01f;

        if (cam_target_angles[1] < -M_PI * .4f) { cam_target_angles[1] = -M_PI * .4f; }
        if (cam_target_angles[1] > M_PI * .4f) { cam_target_angles[1] = M_PI * .4f; }
    }
}

void nbody::Demo::mouseWheel(MouseEvent event)
{
    cam_target_dist -= event.getWheelIncrement() * 5.f;
    if (cam_target_dist < 1.f) { cam_target_dist = 1.f; }
}

void nbody::Demo::mouseDown(MouseEvent event)
{
    // shift click spawns a new galaxy
    if (event.isShiftDown())
    {
        mouse_world_drag_origin = mouse_world_pos();
        mouse_drag = true;
    }
}

void nbody::Demo::mouseUp(MouseEvent event)
{
    if (mouse_drag)
    {
        mouse_drag = false;
        const vec3 pos0 = mouse_world_drag_origin;
        const vec3 pos1 = mouse_world_pos();
        const vec3 diff = pos1 - mouse_world_drag_origin;
        const vec3 n = normalize(diff);
        const nbody::Vector galaxy_axis = {n.x, n.y, n.z};
        const nbody::Vector galaxy_vel = galaxy_axis * length(diff) * .00001f;
        const nbody::Vector galaxy_pos = {pos0.x, pos0.y, pos0.z};
        spawn_galaxy(target_num_elems, {.center=galaxy_pos, .vel=galaxy_vel, .axis=galaxy_axis });
    }
}

void nbody::Demo::draw()
{
    NBODY_PROFILE_ZONE();

    if (! setup_complete)
        return;

    gl::clear(ColorA(0, 0, 0, 1), true);

    gl::setMatrices(camera);

    if (mouse_drag)
    {
        const vec3 pos0 = mouse_world_drag_origin;
        const vec3 pos1 = mouse_world_pos();
        gl::color(1, .2, .2, .9);
        gl::drawLine(vec3(0), vec3(pos1.x, 0, 0));
        gl::color(.2, 1, .2, .9);
        gl::drawLine(vec3(pos1.x, 0, 0), vec3(pos1.x, pos1.y, 0));
        gl::color(.4, .4, 1, .9);
        gl::drawLine(vec3(pos1.x, pos1.y, 0), pos1);

        gl::color(1,1,0,1);
        gl::drawLine(pos0, pos1);
    }


    if (draw_bh_bounds)
    {
        NBODY_PROFILE_ZONE_NAMED("draw bh bounds");
        gl::ScopedGlslProg glsl_scope(bounds_shader);
        gl::ScopedDepth depth_scope(false);
        vbo_bounds->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::enableVertexAttribArray(2);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, floats_per_bound*sizeof(float), (void*)(0*sizeof(float)));
        gl::vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, floats_per_bound*sizeof(float), (void*)(3*sizeof(float)));
        gl::vertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, floats_per_bound*sizeof(float), (void*)(6*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei)(gpu_bounds_data.size() / floats_per_bound));
        vbo_bounds->unbind();
        gl::setDefaultShaderVars();
    }

    if (draw_axes)
    {
        gl::color(1, .2, .2, .5);
        gl::drawLine(vec3(-sim.size(), 0, 0), vec3(sim.size(), 0, 0));
        gl::color(.2, 1, .2, .5);
        gl::drawLine(vec3(0, -sim.size(), 0), vec3(0, sim.size(), 0));
        gl::color(.2, .2, 1, .5);
        gl::drawLine(vec3(0, 0, -sim.size()), vec3(0, 0, sim.size()));
    }

    {
        NBODY_PROFILE_ZONE_NAMED("draw particles");
        gl::ScopedGlslProg glsl_scope(particle_shader);
        vbo_particles->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, floats_per_particle*sizeof(float), nullptr);
        gl::vertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, floats_per_particle*sizeof(float), (void*)(3*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei)(gpu_particle_data.size() / floats_per_particle));
        vbo_particles->unbind();
        gl::setDefaultShaderVars();
    }

    // Last statement in the last thing cinder calls, so this is the frame boundary.
    NBODY_PROFILE_FRAME();
}

vec3 nbody::Demo::homogeneous_to_world(const vec3& homo) const
{
    const mat4 view = camera.getViewMatrix();
    const mat4 proj = camera.getProjectionMatrix();
    const vec4 world = glm::inverse(proj * view) * vec4(homo, 1.f);
    return vec3(world) / world.w;
}

void nbody::Demo::mouse_ray(vec3& out_ray_origin, vec3& out_ray_direction) const
{
    const vec2 mouse_homo = vec2(
            2.0f * (float)mouse_pos.x / (float)getWindowWidth() - 1.0f,
            1.0f - 2.0f * (float)mouse_pos.y / (float)getWindowHeight());
    out_ray_origin = camera.getEyePoint();
    out_ray_direction = normalize(homogeneous_to_world(vec3(mouse_homo, 0)) - out_ray_origin);

    //out_ray_origin = homogeneous_to_world(vec3(mouse_homo, 0));
    //const vec3 ray_end = homogeneous_to_world(vec3(mouse_homo, 1));
    //out_ray_direction = glm::normalize(ray_end - out_ray_origin);
}

vec3 nbody::Demo::mouse_plane_pos(const vec3& plane_point, const vec3& plane_axis) const
{
    vec3 ray_origin;
    vec3 ray_direction;
    mouse_ray(ray_origin, ray_direction);

    const vec3 diff = ray_origin - plane_point;
    float numer = dot(diff, plane_axis);
    float denom = dot(ray_direction, plane_axis);
    if (std::numeric_limits<float>::epsilon() > denom && denom > -std::numeric_limits<float>::epsilon()) {
        return plane_point;
    }
    const float t = numer / denom;
    const vec3 proj = ray_origin - (ray_direction * t);
    return proj;
}

vec3 nbody::Demo::mouse_world_pos(const float dist_from_eye) const
{
    vec3 ray_origin;
    vec3 ray_direction;
    mouse_ray(ray_origin, ray_direction);
    const vec3 plane_pos = camera.getEyePoint() + (ray_direction * dist_from_eye);
    const vec3 plane_axis = -ray_direction;
    return mouse_plane_pos(plane_pos, plane_axis);
}
