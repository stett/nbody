#pragma once
#include <iostream>
#include "GLFW/glfw3.h"

namespace hrr
{
    struct RHIConfig
    {
        bool raytracing = true;
        bool debug = true;
        std::ostream &log = std::cout;
        std::ostream &err = std::cout;
    };

    struct RHIProgram
    {
        enum Type
        {
            None,
            Vertex,
            Geometry,
            Compute,
            Fragment
        };

        RHIProgram(const std::string& name) : name(name) { }

        virtual ~RHIProgram() = default;

        std::string name = "none";
    };

    struct RHIPipeline
    {
        virtual ~RHIPipeline() = default;
    };

    class RHI
    {
    public:
        explicit RHI(RHIConfig &config, GLFWwindow* window) : log(config.log), err(config.err), window(window) { }

        virtual ~RHI() = default;

        virtual RHIProgram* create_shader(RHIProgram::Type type, const std::string& name, const std::vector<uint8_t>& bytes) = 0;

        virtual RHIPipeline* create_pipeline(const std::vector<RHIProgram*>& stages) = 0;

        // Execute active pipelines ("render 1 frame")
        virtual void dispatch(RHIPipeline* pipeline) = 0;

    protected:
        std::ostream &log;
        std::ostream &err;
        GLFWwindow* window;
    };
}
