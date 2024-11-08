#project(nbody_demo cxx)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Stop windows from stomping min and max functions with macros
if (WIN32)
  add_compile_definitions(NOMINMAX)
endif()

# Set up Volk
if (WIN32)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
    #elseif()
    #...
endif()
add_subdirectory(external)

# Find shaders
file(GLOB_RECURSE shader_sources "shaders/source/*.glsl")
file(GLOB_RECURSE shader_includes "shaders/include/*.glsl")

# Make demo target
file(GLOB_RECURSE demo_sources "source/*")
add_executable(nbody_demo ${demo_sources} ${shader_sources})
target_compile_features(nbody_demo PRIVATE cxx_std_20)
set_target_properties(nbody_demo PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(nbody_demo PUBLIC "source")
target_link_libraries(nbody_demo
    volk
    #ktx
    #stb
    #${ASTC_TARGET}
    #imgui
    #tinygltf
    #glm
    #glslang
    #SPIRV
    #vma
    #spirv-cross-glsl
    #glslang-default-resource-limits
    #spdlog
    #ctpl
    #CLI11::CLI11
    glfw)


# Make sure we have the glslc program for compiling shaders
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

# Create shader output directory
set(SHADER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
message("SHADER_SPV_DIR ${SHADER_SPV_DIR}")
add_custom_command(
        TARGET nbody_demo
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_SPV_DIR}
        COMMENT "Creating ${SHADER_SPV_DIR}")

foreach(shader_source IN LISTS shader_sources)

    # Extract filename & add the .spv extension
    get_filename_component(filename ${shader_source} NAME)
    set(filename_spv "${filename}.spv")
    set(shader_spv "${SHADER_SPV_DIR}/${filename_spv}")
    message("    ${filename} -> ${filename_spv}")

    # Generate the compile command
    add_custom_command(
            TARGET nbody_demo
            MAIN_DEPENDENCY "${shader_source}"
            DEPENDS "${source} ${shader_includes} ${SHADER_SPV_DIR}"
            COMMAND "$ENV{VULKAN_SDK}/bin/glslangValidator" "${shader_source}" -V --target-env vulkan1.2 -I"${CMAKE_CURRENT_SOURCE_DIR}/shaders/include" -o "${shader_spv}"
            COMMAND "$ENV{VULKAN_SDK}/bin/spirv-opt" ${shader_spv} -O --strip-debug -o "${shader_spv}"
            COMMENT "Compiling shader: ${filename}"
    )

endforeach()
