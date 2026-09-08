add_subdirectory(external)

file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_executable(nbody_demo ${sources})
target_include_directories(nbody_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source")
target_link_libraries(nbody_demo PRIVATE nbody cinder)
set_target_properties(nbody_demo PROPERTIES WIN32_EXECUTABLE TRUE LINKER_LANGUAGE CXX)
target_compile_features(nbody_demo PRIVATE cxx_std_20)
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")

    # Matches the flags nbody builds itself with -- see the note there for what they do and
    # why they are not the default. Repeated here rather than inherited because Vector and
    # friends are defined in nbody's public headers, so they are compiled into this target
    # too; building the same inline functions two different ways would leave the linker to
    # pick one, and it may not pick the fast one.
    target_compile_options(nbody_demo PRIVATE $<$<NOT:$<CONFIG:Debug>>:/arch:AVX2 /fp:contract>)
endif()
