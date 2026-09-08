if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cinder/CMakeLists.txt")
    message(FATAL_ERROR
        "NBODY_DEMO is ON but demo/external/Cinder is empty. "
        "Fetch it with: git submodule update --init --recursive")
endif()
add_subdirectory(Cinder)
