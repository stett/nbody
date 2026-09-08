
# add thread pool lib
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include)

# add google highway, which supplies the portable SIMD vocabulary: one source written
# against hwy's ops compiles to SSE4/AVX2/AVX-512 on x86 and to NEON/SVE on arm64, with
# the target chosen per translation unit (static dispatch) or per call (dynamic dispatch).
# Unlike tracy this is not optional, so the missing-checkout case has to fail here with
# something better than a header-not-found error from the first translation unit.
#
# add_subdirectory rather than an interface target: highway is not header-only -- libhwy
# carries the runtime target detection and the aligned allocator -- so the `hwy` target
# has to actually be built.
#
# The include root is the repository root, making the include form <hwy/highway.h>.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/highway/hwy/highway.h")
    message(FATAL_ERROR
        "external/highway is empty. "
        "Fetch it with: git submodule update --init --recursive")
endif()

# Everything highway builds beyond the library itself, turned off. These are plain cache
# entries in highway's own cmake rather than option()s, so setting them here first is what
# makes its `set(... CACHE ...)` calls no-ops. FORCE because a build directory configured
# against an earlier revision would otherwise keep the defaults it cached. Tests matter
# most: HWY_ENABLE_TESTS also fetches googletest over the network while configuring.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(highway)

# hwy_test (the test-only support library) and hwy_list_targets (a tool that prints the
# compiled-in targets to stderr from a POST_BUILD step) are created unconditionally by
# highway's cmake, HWY_ENABLE_TESTS notwithstanding. Nothing here links either, so keep
# them out of ordinary builds; both remain buildable by name.
set_target_properties(hwy_test hwy_list_targets PROPERTIES EXCLUDE_FROM_ALL ON)

# SYSTEM, because these headers are not ours to keep warning-clean -- the same -external:I
# treatment Vulkan's headers already get. add_subdirectory(highway SYSTEM) would say this
# directly but needs cmake 3.25; promoting the interface include dirs by hand is the
# equivalent for the 3.24 this tree requires.
get_target_property(hwy_include_dirs hwy INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(hwy PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${hwy_include_dirs}")

# either half of tracy needs the submodule actually checked out
if (NBODY_TRACY OR NBODY_TRACY_PROFILER)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tracy/CMakeLists.txt")
        message(FATAL_ERROR
            "NBODY_TRACY or NBODY_TRACY_PROFILER is ON but external/tracy is empty. "
            "Fetch it with: git submodule update --init --recursive")
    endif()
endif()

# tracy profiler client, only when profiling was asked for
if (NBODY_TRACY)
    # TRACY_ENABLE defaults OFF in tracy and would compile away everything instrumented
    # here. ON_DEMAND waits for a profiler to connect; STATIC avoids shipping a client dll.
    # Plain variables, not cache: CMP0077 is NEW in tracy's scope so its option() defers.
    set(TRACY_ENABLE ON)
    set(TRACY_ON_DEMAND ON)
    set(TRACY_STATIC ON)

    add_subdirectory(tracy)
endif()

# The `tracy-profiler` target. Only entered on request: its cmake fetches imgui, freetype,
# capstone and the rest over the network while configuring.
if (NBODY_TRACY_PROFILER)
    # CPM's default cache sits under the build tree, where the paths outgrow the 260
    # character windows limit and the clone fails outright. Keep it near the source root.
    if (NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
        set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cpm-cache"
            CACHE PATH "Directory CPM caches fetched dependencies in")
    endif()

    # EXCLUDE_FROM_ALL: still buildable by name, but ordinary builds do not relink it.
    add_subdirectory(tracy/profiler EXCLUDE_FROM_ALL)
endif()