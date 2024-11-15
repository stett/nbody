#project(nbody_demo cxx)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Stop windows from stomping min and max functions with macros
if (WIN32)
  add_compile_definitions(NOMINMAX)
endif()

add_subdirectory(external)

# Make demo target
file(GLOB_RECURSE demo_sources "source/*.cpp")
file(GLOB_RECURSE demo_resources "resources/*")
add_executable(nbody_demo ${demo_sources})
target_compile_features(nbody_demo PRIVATE cxx_std_20)
set_target_properties(nbody_demo PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(nbody_demo PUBLIC "source" "resources")
target_link_libraries(nbody_demo PUBLIC nbody)
target_link_libraries(nbody_demo PUBLIC cinder)
set_target_properties(${demo} PROPERTIES WIN32_EXECUTABLE TRUE)
