# ####################################################################################################################################################
# probes/find_package_consumer/CMakeLists.txt
#
# Tiny downstream consumer used by Test C's live phase to verify `find_package(AMIO REQUIRED)` resolves cleanly against an installed AMIO prefix
# (R13.1) without any host-level Rust/Cargo/Go toolchain (R13.2).
#
# The consumer compiles a single C99 translation unit that:
#
# * #includes only <amio/amio.h> (the public header, R13.6),
# * links against AMIO::amio_core via the imported target supplied by AMIOConfig.cmake.
#
# It does not need to *run* anything -- a successful configure + build is the signal that imported targets are wired up correctly.
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)
project(amio_find_package_consumer LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT DEFINED AMIO_INSTALL_PREFIX)
    message(FATAL_ERROR "AMIO_INSTALL_PREFIX must be set on the command line")
endif()

# Require AMIO via its installed package config.  The PATHS hint points at the temp install prefix the test driver populated.
find_package(AMIO REQUIRED PATHS "${AMIO_INSTALL_PREFIX}" NO_DEFAULT_PATH)

if(NOT TARGET AMIO::amio_core)
    message(FATAL_ERROR "find_package(AMIO) succeeded but AMIO::amio_core target was not exported")
endif()
if(NOT TARGET AMIO::public_headers)
    message(FATAL_ERROR "find_package(AMIO) succeeded but AMIO::public_headers target was not exported")
endif()

add_executable(amio_consumer_check consumer.c)
target_link_libraries(amio_consumer_check PRIVATE AMIO::amio_core)

# Final positive signal the test driver greps for.
message(STATUS "CONSUMER_RESOLVED_AMIO=OK")
