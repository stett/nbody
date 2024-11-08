
# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_library(nbody STATIC ${sources})
set(CMAKE_CXX_STANDARD 20)
target_include_directories(nbody PUBLIC "include")

# add nbody test suite
add_subdirectory(tests)

