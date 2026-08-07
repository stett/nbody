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
# run time, which decides whether the GPU simulation variants are offered. There is
# deliberately no build option here — a missing or unusable device must degrade to a CPU
# variant, not fail the build.
message(STATUS "VULKAN_SDK: $ENV{VULKAN_SDK}")
find_package(Vulkan REQUIRED)
# PRIVATE: no public header includes vulkan (gpu.h lives under source/), so consumers must
# not inherit the include dirs. nbody is STATIC, so these still propagate for linking.
target_link_libraries(nbody PRIVATE Vulkan::Vulkan)
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
#
# Not guarded by a GPU build option: the Vulkan backend is always compiled (see above), so
# gpu.cpp always #includes the generated shader headers and they must always be generated.

# the SPIR-V toolchain ships with the Vulkan SDK, which is already a hard dependency
find_program(exec_spirv_opt NAMES spirv-opt HINTS Vulkan::SPIRV-Tools)
find_program(exec_glslang_validator NAMES glslangValidator HINTS Vulkan::glslangValidator)
if (NOT exec_glslang_validator OR NOT exec_spirv_opt)
    message(FATAL_ERROR
        "glslangValidator and spirv-opt are required to build the shaders. "
        "Both ship with the Vulkan SDK; ensure its Bin directory is on the PATH. "
        "(VULKAN_SDK: $ENV{VULKAN_SDK})")
endif()
message(STATUS "glslangValidator: ${exec_glslang_validator}")
message(STATUS "spirv-opt: ${exec_spirv_opt}")

# find shader stage files to compile to spir-v. only real stage files are
# listed here; anything under shaders/include is a dependency, not a
# translation unit of its own.
file(GLOB_RECURSE shader_sources CONFIGURE_DEPENDS
    "shaders/*.comp" "shaders/*.vert" "shaders/*.frag"
    "shaders/*.geom" "shaders/*.tesc" "shaders/*.tese")

# every shader is rebuilt when any shared include changes
file(GLOB_RECURSE shader_includes CONFIGURE_DEPENDS "shaders/include/*")

# shader output directories. created at configure time so no build step
# has to depend on their existence.
#
# Generated headers go in the build tree, not the source tree. In the source tree they
# would be shared by every build directory configured against this checkout, so two build
# trees would clobber each other's shaders; they would also make the source tree
# unbuildable when read-only, and require a .gitignore entry to keep generated output
# untracked. Here each build tree gets its own copy.
set(shader_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(shader_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(shader_h_dir "${shader_generated_dir}/shaders")
message(STATUS "Shader binary directory: ${shader_spv_dir}")
message(STATUS "Shader header directory: ${shader_h_dir}")
file(MAKE_DIRECTORY "${shader_spv_dir}" "${shader_h_dir}")

# PRIVATE: only source/gpu.cpp includes these, as "shaders/<stage>.h". They are an
# implementation detail of the Vulkan backend, so consumers must not inherit the path --
# same reasoning as Vulkan::Vulkan above.
target_include_directories(nbody PRIVATE "${shader_generated_dir}")

# loop over each glsl shader
set(shader_headers "")
foreach(shader_source IN LISTS shader_sources)

    # extract filename & add the .spv extension.
    get_filename_component(filename ${shader_source} NAME_WLE)
    set(filename_spv "${filename}.spv")
    set(filename_h "${filename}.h")
    set(shader_spv "${shader_spv_dir}/${filename_spv}")
    set(shader_h "${shader_h_dir}/${filename_h}")
    set(shader_c_arr "spv_${filename}")
    message(STATUS "Shader compilation step: ${filename} -> ${filename_spv} -> ${filename_h} : ${shader_c_arr}")

    # glsl -> spv
    add_custom_command(
        OUTPUT "${shader_spv}"
        MAIN_DEPENDENCY "${shader_source}"
        DEPENDS ${shader_includes}
        # the include path is one argument with no embedded quotes: under VERBATIM cmake
        # escapes quotes inside an argument, so -I"${dir}" would reach glslang with the
        # quote characters still in the path.
        COMMAND "${exec_glslang_validator}" "${shader_source}" -V --target-env vulkan1.4 "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders/include" -o "${shader_spv}"
        COMMAND "${exec_spirv_opt}" "${shader_spv}" -O --strip-debug -o "${shader_spv}"
        COMMENT "Compiling shader spv: ${filename} -> ${filename_spv}"
        VERBATIM)

    # spv -> c header, via a cmake script so the build needs no tooling beyond the
    # Vulkan SDK and cmake itself.
    add_custom_command(
        OUTPUT "${shader_h}"
        DEPENDS "${shader_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake"
        COMMAND "${CMAKE_COMMAND}"
            -Dspv=${shader_spv} -Dheader=${shader_h} -Dname=${shader_c_arr}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake"
        COMMENT "Embedding shader header ${filename_spv} -> ${filename_h}"
        VERBATIM)

    list(APPEND shader_headers "${shader_h}")

endforeach()

# gpu.cpp #includes the generated headers, so they must exist before any of
# nbody's objects compile. a target-level dependency gives that ordering;
# attaching these steps to nbody as POST_BUILD would deadlock, since the
# headers would only be generated after the target that needs them links.
add_custom_target(nbody_shaders DEPENDS ${shader_headers})
add_dependencies(nbody nbody_shaders)
