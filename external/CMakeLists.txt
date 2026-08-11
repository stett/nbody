
# add thread pool lib
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include)

# tracy profiler client, only when profiling was asked for
if (NBODY_TRACY)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tracy/CMakeLists.txt")
        message(FATAL_ERROR
            "NBODY_TRACY is ON but external/tracy is empty. "
            "Fetch it with: git submodule update --init --recursive")
    endif()

    # Plain variables rather than cache entries: tracy declares cmake 3.13, so CMP0077 is
    # NEW in its scope and its option() calls defer to these without a cache entry being
    # left behind to confuse a later reconfigure.
    #
    # TRACY_ENABLE is the one that matters -- tracy defaults it OFF, and without it every
    # macro in the client headers expands to nothing however thoroughly the code is
    # instrumented. ON_DEMAND then holds the recording back until a profiler actually
    # connects, rather than accumulating from process start. STATIC avoids a client dll
    # that would have to be shipped alongside the executable.
    set(TRACY_ENABLE ON)
    set(TRACY_ON_DEMAND ON)
    set(TRACY_STATIC ON)

    add_subdirectory(tracy)
endif()