#include <GLFW/glfw3.h>
#include <cassert>
#include <iostream>
#include "vk/vk_rhi.h"
#include "win/win_platform.h"

int main()
{
    // Create error function callback
    const auto glfw_error_callback = [](int error, const char *description) {
        fprintf(stderr, "GLFW error: %s\n", description);
    };
    glfwSetErrorCallback(glfw_error_callback);

    // Initialize GLFW
    assert(glfwInit());

    // Create GLFW window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "hardrare", nullptr, nullptr);

    // Set keyboard input so that we can close the window with "esc"
    const auto glfw_key_callback = [](GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
            if (key == GLFW_KEY_ESCAPE)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
    };
    glfwSetKeyCallback(window, glfw_key_callback);

    {
        // Create platform accessor
        hrr::PlatformConfig platform_config;
        std::unique_ptr<hrr::Platform> platform = std::make_unique<hrr::WinPlatform>(platform_config);

        // Create vulkan instance
        hrr::RHIConfig rhi_config;
        std::unique_ptr<hrr::RHI> rhi = std::make_unique<hrr::VulkanRHI>(rhi_config, window);

        // Use the RHI to load shader program binaries
        std::vector<uint8_t> vert_bytes = platform->get_resource_bytes("shaders/triangle.vert.glsl.spv");
        std::vector<uint8_t> frag_bytes = platform->get_resource_bytes("shaders/triangle.frag.glsl.spv");
        hrr::RHIProgram* triangle_vert = rhi->create_shader(hrr::RHIProgram::Type::Vertex, "triangle:vert", vert_bytes);
        hrr::RHIProgram* triangle_frag = rhi->create_shader(hrr::RHIProgram::Type::Fragment, "triangle:frag", frag_bytes);

        // Create a render pipeline
        std::vector<hrr::RHIProgram*> programs { triangle_vert, triangle_frag };
        hrr::RHIPipeline* triangle_pipeline = rhi->create_pipeline(programs);

        // Loop until GLFW asks to close the window
        while (glfwWindowShouldClose(window) == false)
        {
            glfwPollEvents();

            // render the frame
            rhi->dispatch(triangle_pipeline);
        }
    }

    // Terminate window
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}