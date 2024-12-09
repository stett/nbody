cmake_minimum_required(VERSION 3.5)
project(nbody)
set(CMAKE_CXX_STANDARD 20)

# add dependencies
add_subdirectory(external)

# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_library(nbody STATIC ${sources})
set(CMAKE_CXX_STANDARD 20)
target_include_directories(nbody PUBLIC "include")
target_link_libraries(nbody PUBLIC BS_thread_pool)
target_compile_features(nbody PRIVATE cxx_std_20)
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# add nbody test suite
add_subdirectory(tests)

