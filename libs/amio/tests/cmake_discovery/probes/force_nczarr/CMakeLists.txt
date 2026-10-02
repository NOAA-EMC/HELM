# ####################################################################################################################################################
# probes/force_nczarr/CMakeLists.txt
#
# Test probe for: AMIO_FORCE_NCZARR=ON skips TensorStore discovery and yields AMIO_HAS_TENSORSTORE=OFF.
#
# Mermaid path verified (design.md "TensorStore discovery and NCZarr fallback (configure-time)"):
#
# A[CMake configure] -> B{AMIO_FORCE_NCZARR?} B -- ON --> F[Set AMIO_HAS_TENSORSTORE=OFF; STATUS: TensorStore discovery short-circuited]
#
# Driven by tests/cmake_discovery/CMakeLists.txt; receives AMIO_SOURCE_DIR (the AMIO repo root) on the command line so we can pull in the
# FindTensorStore.cmake module under test from the unmodified source tree.
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)
project(amio_probe_force_nczarr LANGUAGES NONE)

if(NOT DEFINED AMIO_SOURCE_DIR)
    message(FATAL_ERROR "AMIO_SOURCE_DIR must be defined by the caller")
endif()

# Pull in the find module under test from the AMIO repository.  We deliberately do NOT add the rest of AMIO's CMake plumbing -- this probe isolates
# exactly the discovery decision, nothing else.
list(APPEND CMAKE_MODULE_PATH "${AMIO_SOURCE_DIR}/cmake")

# Honor the input forcing flag (the test harness passes ON).
if(NOT DEFINED AMIO_FORCE_NCZARR)
    set(AMIO_FORCE_NCZARR OFF)
endif()

find_package(TensorStore MODULE QUIET)

# Emit a single-line marker block CTest's PASS_REGULAR_EXPRESSION anchors against.  Using a single message() call keeps the values adjacent on one
# logical output line, which makes the test regex robust against interleaving from CMake's own status messages.
message(STATUS "PROBE_AMIO_HAS_TENSORSTORE=${AMIO_HAS_TENSORSTORE} " "PROBE_TENSORSTORE_FOUND=${TensorStore_FOUND} "
               "PROBE_TENSORSTORE_REASON=${TensorStore_REASON}")
