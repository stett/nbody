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

