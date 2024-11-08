project(nbody_demo_external)


#
# vulkan
#

add_library(vulkan INTERFACE)
set(VULKAN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include)
target_sources(vulkan INTERFACE ${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h)
target_include_directories(vulkan SYSTEM INTERFACE ${VULKAN_INCLUDE_DIR})

target_compile_definitions(vulkan INTERFACE VK_NO_PROTOTYPES)

if(VKB_ENABLE_PORTABILITY)
    # When portability is enabled, must enable Vulkan beta extensions for access to VK_KHR_portability_subset
    target_compile_definitions(vulkan INTERFACE VK_ENABLE_BETA_EXTENSIONS)
endif()

if(ANDROID)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_ANDROID_KHR)
elseif(WIN32)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_METAL_EXT)
elseif(UNIX)
    # Choose WSI based on VKB_WSI_SELECTION
    if (VKB_WSI_SELECTION STREQUAL XCB OR VKB_WSI_SELECTION STREQUAL XLIB OR VKB_WSI_SELECTION STREQUAL WAYLAND)
        find_package(PkgConfig REQUIRED)
    endif()
    if (VKB_WSI_SELECTION STREQUAL XCB)
        pkg_check_modules(XCB xcb REQUIRED)
        if (XCB_FOUND)
            target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_XCB_KHR)
        endif()
    elseif (VKB_WSI_SELECTION STREQUAL XLIB)
        pkg_check_modules(X11 x11 REQUIRED)
        if (X11_FOUND)
            target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_XLIB_KHR)
        endif()
    elseif (VKB_WSI_SELECTION STREQUAL WAYLAND)
        pkg_check_modules(WAYLAND wayland-client REQUIRED)
        if (WAYLAND_FOUND)
            target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_WAYLAND_KHR)
        endif()
    elseif (VKB_WSI_SELECTION STREQUAL D2D)
        set(DIRECT_TO_DISPLAY TRUE)
        set(DIRECT_TO_DISPLAY TRUE PARENT_SCOPE)
        target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_DISPLAY_KHR)
    else()
        message(FATAL_ERROR "Unknown WSI")
    endif()
endif()

#
# volk
#

set(VOLK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/volk")
set(VOLK_FILES
    "${VOLK_DIR}/volk.c"
    "${VOLK_DIR}/volk.h")

add_library(volk STATIC ${VOLK_FILES})
set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(volk PUBLIC vulkan)

target_include_directories(volk SYSTEM PUBLIC ${VOLK_DIR})
if (VKB_WSI_SELECTION STREQUAL XCB)
    target_include_directories(volk SYSTEM PUBLIC ${XCB_INCLUDE_DIRS})
elseif (VKB_WSI_SELECTION STREQUAL XLIB)
    target_include_directories(volk SYSTEM PUBLIC ${X11_INCLUDE_DIRS})
elseif (VKB_WSI_SELECTION STREQUAL WAYLAND)
    target_include_directories(volk SYSTEM PUBLIC ${WAYLAND_INCLUDE_DIRS})
endif()

set_property(TARGET volk PROPERTY FOLDER "external")

#
# GLFW
#

if(NOT ANDROID AND NOT IOS)
    if (NOT DIRECT_TO_DISPLAY)
        # GLFW
        option(GLFW_BUILD_DOCS OFF)
        option(GLFW_BUILD_TESTS OFF)
        option(GLFW_BUILD_EXAMPLES OFF)
        option(GLFW_BUILD_TESTS OFF)
        option(GLFW_INSTALL OFF)
        option(BUILD_SHARED_LIBS ON)
        if (VKB_WSI_SELECTION STREQUAL WAYLAND)
            set(GLFW_USE_WAYLAND ON)
        endif()

        add_subdirectory(glfw)

        if (TARGET update_mappings)
            set_target_properties(update_mappings PROPERTIES FOLDER "external")
        endif()
        set_property(TARGET glfw PROPERTY FOLDER "external")
    endif()
endif()
