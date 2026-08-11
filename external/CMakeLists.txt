
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

# The profiler application, as the `tracy-profiler` target.
#
# Its cmake fetches imgui, freetype, capstone and the rest through CPM while configuring, so
# this directory is only entered when asked for -- otherwise every configure of every build
# directory would want the network. It declares its own project() and keeps its compile
# options to its own scope, so nothing it does reaches our targets.
if (NBODY_TRACY_PROFILER)
    # CPM caches what it fetches inside the build tree by default. That path is already
    # several directories down -- build root, external, nbody, external, tracy, profiler --
    # and adding a package name, a forty character commit hash and the deepest file in the
    # package runs past the 260 characters windows still imposes on most of its file API. A
    # vendored wayland protocol file inside the file-dialog dependency is long enough on its
    # own to fail the clone, so the default is not merely untidy here, it does not work.
    #
    # Anchoring the cache near the source root keeps the total well inside the limit, and
    # incidentally shares it between build directories rather than refetching per configure.
    # Deferred to an existing setting if there is one, since a machine-wide cache is a
    # perfectly good answer too.
    if (NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
        set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cpm-cache"
            CACHE PATH "Directory CPM caches fetched dependencies in")
    endif()

    # EXCLUDE_FROM_ALL so that having the profiler available does not mean rebuilding it on
    # every ordinary build. It stays a named target -- build `tracy-profiler` when you want
    # it, which in practice is once.
    add_subdirectory(tracy/profiler EXCLUDE_FROM_ALL)
endif()