cmake_minimum_required(VERSION 3.24)
project(nbody)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add dependencies
add_subdirectory(external)

# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_library(nbody STATIC ${sources})

# link external libraries
target_include_directories(nbody PUBLIC "include" "source")
target_link_libraries(nbody PUBLIC BS_thread_pool)

# GPU support is a build-time dependency but a *run-time* capability: the Vulkan backend
# is always compiled, and whether a usable compute device actually exists is discovered at
# run time (see GPU::init / Sim::init_gpu). There is deliberately no build option here —
# a missing or unusable device must degrade to the CPU path, not fail the build.
message(STATUS "VULKAN_SDK: $ENV{VULKAN_SDK}")
find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
# PRIVATE: no public header includes vulkan (gpu.h lives under source/), so consumers must
# not inherit the include dirs. nbody is STATIC, so these still propagate for linking.
target_link_libraries(nbody PRIVATE Vulkan::Vulkan Vulkan::shaderc_combined)
target_link_libraries(nbody PUBLIC Vulkan::Headers)
message(STATUS "Vulkan_VERSION: ${Vulkan_VERSION}")

# set up compile options
target_compile_features(nbody PRIVATE cxx_std_20)
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# add nbody test suite
add_subdirectory(tests)

# compile shaders
#[[
if (nbody_gpu)

    # Make sure we have the glslc program for compiling shaders
    find_program(exec_glslc NAMES glslc HINTS Vulkan::glslc)
    find_program(exec_spirv_opt NAMES spirv-opt HINTS Vulkan::SPIRV-Tools)
    find_program(exec_glslang_validator NAMES glslangValidator HINTS Vulkan::glslangValidator)

    message("${exec_glslc}")
    message("${exec_spirv_opt}")
    message("${exec_glslang_validator}")

    # find shader files to compile to spir-v
    file(GLOB_RECURSE shader_sources "shader/*.*")

    # create shader output directory
    set(SHADER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shader")
    message("SHADER_SPV_DIR ${SHADER_SPV_DIR}")
    add_custom_command(
        TARGET nbody
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_SPV_DIR}
        COMMENT "Creating ${SHADER_SPV_DIR}")

    # loop over each glsl shader
    foreach(shader_source IN LISTS shader_sources)

        # extract filename & add the .spv extension
        get_filename_component(filename ${shader_source} NAME)
        set(filename_spv "${filename}.spv")
        set(shader_spv "${SHADER_SPV_DIR}/${filename_spv}")
        message("    ${filename} -> ${filename_spv}")

        # generate the compile command
        add_custom_command(
            TARGET nbody
            MAIN_DEPENDENCY "${shader_source}"
            DEPENDS "${source} ${shader_includes} ${SHADER_SPV_DIR}"
            COMMAND "${exec_glslang_validator}" "${shader_source}" -V --target-env vulkan1.4 -I"${CMAKE_CURRENT_SOURCE_DIR}/shaders/include" -o "${shader_spv}"
            #COMMAND "${exec_spirv_opt}" ${shader_spv} -O --strip-debug -o "${shader_spv}"
            COMMENT "Compiling shader: ${filename}"
        )

    endforeach()

endif()
#]]
