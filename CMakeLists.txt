cmake_minimum_required(VERSION 3.5)
project(nbody)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_library(nbody STATIC ${sources})
target_compile_features(nbody PRIVATE cxx_std_20)
target_include_directories(nbody PUBLIC "include")

# add nbody test suite
add_subdirectory(tests)

# add demo project
add_subdirectory(demo)
