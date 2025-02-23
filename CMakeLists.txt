cmake_minimum_required(VERSION 3.5)
project(nbody)
set(CMAKE_CXX_STANDARD 20)

# options
option(nbody_gpu "include gpu acceleration" ON)

# add dependencies
add_subdirectory(external)

# preprocessor defines
add_compile_definitions(NBODY_GPU=1)

# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_library(nbody STATIC ${sources})

# link external libraries
set(CMAKE_CXX_STANDARD 20)
target_include_directories(nbody PUBLIC "include" "source")
target_link_libraries(nbody PUBLIC BS_thread_pool)
if (nbody_gpu)
    message("VULKAN_SDK: $ENV{VULKAN_SDK}")
    find_package(Vulkan REQUIRED)
    target_link_libraries(nbody PUBLIC Vulkan::Vulkan Vulkan::Headers)
    message("Vulkan_VERSION: ${Vulkan_VERSION}")
    message("Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
    message("Vulkan_LIBRARY: ${Vulkan_LIBRARY}")
    message("Vulkan_INCLUDE_DIR: ${Vulkan_INCLUDE_DIR}")
endif()

# set up compile options
target_compile_features(nbody PRIVATE cxx_std_20)
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# add nbody test suite
add_subdirectory(tests)

# compile shaders
if (nbody_gpu)

    # Make sure we have the glslc program for compiling shaders
    find_program(exec_glslc NAMES glslc HINTS Vulkan::glslc)
    find_program(exec_spirv_opt NAMES spirv-opt HINTS Vulkan::SPIRV-Tools)
    find_program(exec_glslang_validator NAMES glslangValidator HINTS Vulkan::glslangValidator)

    message("${exec_glslc}")
    message("${exec_spirv_opt}")
    message("${exec_glslang_validator}")

    # find shader files to compile to spir-v
    file(GLOB_RECURSE shader_sources "shaders/*.*")

    # create shader output directory
    #set(shader_spv_dir "${CMAKE_CURRENT_BINARY_DIR}")
    set(shader_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    message("Shader binary directory: ${shader_spv_dir}")
    add_custom_command(
        TARGET nbody
        COMMAND ${CMAKE_COMMAND} -E make_directory ${shader_spv_dir}
        COMMENT "Creating shader binary directory: ${shader_spv_dir}")

    # create shader spv-header directory
    set(shader_h_dir "${CMAKE_CURRENT_SOURCE_DIR}/include/shaders")
    message("Shader header directory: ${shader_h_dir}")
    add_custom_command(
        TARGET nbody
        COMMAND ${CMAKE_COMMAND} -E make_directory ${shader_h_dir}
        COMMENT "Creating shader header directory: ${shader_h_dir}")

    # loop over each glsl shader
    foreach(shader_source IN LISTS shader_sources)

        # extract filename & add the .spv extension.
        get_filename_component(filename ${shader_source} NAME_WLE)
        set(filename_spv "${filename}.spv")
        set(filename_h "${filename}.h")
        set(shader_spv "${shader_spv_dir}/${filename_spv}")
        set(shader_h "${shader_h_dir}/${filename_h}")
        set(shader_c_arr "spv_${filename}")
        message("Shader compilation step: ${filename} -> ${filename_spv} -> ${filename_h} : ${shader_c_arr}")

        # generate the compile command
        add_custom_command(
            TARGET nbody
            MAIN_DEPENDENCY "${shader_source}"
            DEPENDS "${source} ${shader_includes} ${shader_spv_dir}"
            COMMAND "${exec_glslang_validator}" "${shader_source}" -V --target-env vulkan1.4 -I"${CMAKE_CURRENT_SOURCE_DIR}/shaders/include" -o "${shader_spv}"
            COMMAND "${exec_spirv_opt}" ${shader_spv} -O --strip-debug -o "${shader_spv}"
            COMMENT "Compiling shader spv: ${filename} -> ${filename_spv}")

        # generate command to
        add_custom_command(
            TARGET nbody
            OUTPUT ${SHADER_HEADER}
            COMMAND xxd -i -n ${shader_c_arr} ${shader_spv} > ${shader_h}
            DEPENDS ${SHADER_SPV}
            COMMENT "Compiling shader header ${filename_spv} -> ${filename_h}")

    endforeach()

endif()
