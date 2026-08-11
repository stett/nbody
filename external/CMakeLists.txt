
# add thread pool lib
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include)

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