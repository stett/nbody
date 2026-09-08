add_subdirectory(Catch2)
file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_executable(nbody_tests ${test_sources})
target_link_libraries(nbody_tests PUBLIC nbody)
target_link_libraries(nbody_tests PRIVATE Catch2::Catch2WithMain)

# The SIMD kernels in detail/physics.h are header-inline, so the vector width a test
# exercises is whatever the *test* translation unit was compiled for, not what the library
# ships. Without these the tests would run 128-bit SSE2 (MSVC's x86-64 baseline) against a
# library built for 256-bit AVX2 and would not cover the width that actually runs -- which
# is how a partial-vector store bug got through. Same flags, same reasoning, as nbody's own
# (see the root CMakeLists.txt).
if (MSVC)
    target_compile_options(nbody_tests PRIVATE $<$<NOT:$<CONFIG:Debug>>:/arch:AVX2 /fp:contract>)
endif()
