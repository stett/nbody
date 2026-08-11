# Embed a SPIR-V blob in a C header as a byte array.
#
# Run as a build step via `cmake -P`, so the build needs no tooling beyond the Vulkan SDK
# and CMake itself.
#
# Usage:
#   cmake -Dspv=<in.spv> -Dheader=<out.h> -Dname=<identifier> -P spv_to_header.cmake

foreach(required spv header name)
    if (NOT ${required})
        message(FATAL_ERROR "spv_to_header.cmake requires -D${required}=")
    endif()
endforeach()

if (NOT EXISTS "${spv}")
    message(FATAL_ERROR "spv_to_header.cmake: no such file: ${spv}")
endif()

file(READ "${spv}" hex HEX)
string(REGEX MATCHALL "[0-9a-f][0-9a-f]" bytes "${hex}")
list(LENGTH bytes byte_count)

# SPIR-V is a stream of 32-bit words, and vk::ShaderModuleCreateInfo takes the size in
# bytes. A blob that is not a whole number of words means a truncated or non-SPIR-V input;
# Vulkan would reject it later with a far less obvious error, so fail here instead.
if (byte_count EQUAL 0)
    message(FATAL_ERROR "spv_to_header.cmake: ${spv} is empty")
endif()
math(EXPR remainder "${byte_count} % 4")
if (NOT remainder EQUAL 0)
    message(FATAL_ERROR
        "spv_to_header.cmake: ${spv} is ${byte_count} bytes, "
        "not a whole number of 32-bit SPIR-V words")
endif()

# format as 12 bytes per line, purely for legibility of the generated file
set(lines "")
set(line "")
set(column 0)
foreach(byte IN LISTS bytes)
    string(APPEND line "0x${byte}, ")
    math(EXPR column "${column} + 1")
    if (column EQUAL 12)
        string(STRIP "${line}" line)
        list(APPEND lines "    ${line}")
        set(line "")
        set(column 0)
    endif()
endforeach()
if (NOT line STREQUAL "")
    string(STRIP "${line}" line)
    list(APPEND lines "    ${line}")
endif()
string(JOIN "\n" body ${lines})

get_filename_component(spv_name "${spv}" NAME)

# `static`: an external array would collide at link time if two translation units ever
# included the same header.
file(WRITE "${header}"
"// Generated from ${spv_name} by cmake/spv_to_header.cmake. Do not edit.\n"
"#pragma once\n"
"\n"
"static const unsigned char ${name}[${byte_count}] = {\n"
"${body}\n"
"};\n")
