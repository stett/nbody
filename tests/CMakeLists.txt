add_subdirectory(Catch2)
file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_executable(nbody_tests ${test_sources})
target_link_libraries(nbody_tests PUBLIC nbody)
target_link_libraries(nbody_tests PRIVATE Catch2::Catch2WithMain)

#set(nbody_gpu ON)
add_compile_definitions(NBODY_GPU=1)
