
# add thread pool lib
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include)

# add collision library
set(BUILD_PYTHON_INTERFACE false)
add_subdirectory(coal)