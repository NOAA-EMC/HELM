# ####################################################################################################################################################
# probes/fallback_warning/CMakeLists.txt
#
# Test probe for: TensorStore unavailable -> NCZarr fallback WARNING is emitted, naming the fallback mode and the reason.
#
# Mermaid path verified (design.md "TensorStore discovery and NCZarr fallback (configure-time)"):
#
# A -> B{AMIO_FORCE_NCZARR?} -- OFF --> C[find_package(TensorStore CONFIG)] -> D{TensorStore_FOUND?} -- NO --> F (fallback message)
#
# Setup: * AMIO_FORCE_NCZARR=OFF (TensorStore discovery is allowed to run). * CMAKE_PREFIX_PATH is emptied by the caller so any TensorStore
# installation on the host cannot accidentally be located -- the probe must observe the "not found" branch deterministically.
#
# Validation: * AMIO_NETCDF_HAS_NCZARR is forced to TRUE in the probe so the selection function reaches the WARNING branch (the FATAL_ERROR branch in
# AMIOZarrBackend.cmake fires only when both TensorStore and NCZarr-capable netCDF-c are missing -- that branch is exercised by Test C's negative-path
# inspection instead, which keeps this probe deterministic on bare hosts).
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)
project(amio_probe_fallback_warning LANGUAGES NONE)

if(NOT DEFINED AMIO_SOURCE_DIR)
    message(FATAL_ERROR "AMIO_SOURCE_DIR must be defined by the caller")
endif()

list(APPEND CMAKE_MODULE_PATH "${AMIO_SOURCE_DIR}/cmake")

if(NOT DEFINED AMIO_FORCE_NCZARR)
    set(AMIO_FORCE_NCZARR OFF)
endif()

# Run the actual discovery.  With AMIO_FORCE_NCZARR=OFF and the test harness's empty CMAKE_PREFIX_PATH, this find_package() resolves
# TensorStore_FOUND=FALSE on any host, including air-gapped ones.
find_package(TensorStore MODULE QUIET)

# Force the NCZarr-capable-netCDF probe result so we exercise the WARNING branch of amio_select_zarr_backend() rather than the FATAL_ERROR branch.
# This is correct for an isolated unit test of R13.4's messaging contract -- the netCDF-discovery half is verified independently in CMakeLists.txt's
# check_c_source_compiles probe.
set(AMIO_NETCDF_HAS_NCZARR TRUE)

include(AMIOZarrBackend)
amio_select_zarr_backend()

# Emit machine-checkable confirmation that we did go through the expected branch (TensorStore not found, AMIO_HAS_TENSORSTORE=OFF). The
# PASS_REGULAR_EXPRESSION on the test does not depend on this line, but it is useful for diagnostic output if the test fails.
message(STATUS "PROBE_AMIO_HAS_TENSORSTORE=${AMIO_HAS_TENSORSTORE} " "PROBE_TENSORSTORE_FOUND=${TensorStore_FOUND} "
               "PROBE_BACKEND_DESC=${AMIO_ZARR_BACKEND_DESC} " "PROBE_BACKEND_REASON=${AMIO_ZARR_BACKEND_REASON}")
