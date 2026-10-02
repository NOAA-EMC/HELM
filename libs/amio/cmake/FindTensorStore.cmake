# ####################################################################################################################################################
# FindTensorStore.cmake
#
# Locates Google TensorStore for AMIO's Zarr v3 backend.  Encapsulates the configure-time discovery flow described in design.md -> "TensorStore
# discovery and NCZarr fallback (configure-time)" so that the top-level CMakeLists.txt only consumes the result.
#
# Flow (mirrors the mermaid diagram in design.md):
#
# CMake configure | v AMIO_FORCE_NCZARR ? |       \ ON \       \-- OFF |             \ |              find_package(TensorStore CONFIG) | | |
# TensorStore_FOUND ? |                   |        \ |                  YES        NO |                   |          \ v v           v [skip
# discovery]    [tensorstore]   [fallback] |                   |               |
# +------- driver selection ----------+
#
# Outputs (cache + parent-scope): TensorStore_FOUND         - TRUE iff a usable TensorStore was located TensorStore_REASON        - human-readable
# reason (only if NOT found or when AMIO_FORCE_NCZARR=ON) AMIO_HAS_TENSORSTORE      - ON/OFF mirror of TensorStore_FOUND, used by the rest of the
# build (R13.3, R13.4)
#
# Imported target (when found): TensorStore::tensorstore  - linked PRIVATELY into driver_zarr
#
# This module deliberately does NOT search for `nceplibs-g2c`, `netCDF`, or any other backend dependency -- those discoveries are orchestrated
# separately by the top-level CMakeLists.txt so the fallback decision (NCZarr vs FATAL_ERROR) lives in one place.
#
# Requirements: R13.3, R13.4 (warning/status emission), R13.5 (fallback selection), R8.6 (NCZarr fallback path activation).
# ####################################################################################################################################################

include_guard(GLOBAL)
include(FindPackageHandleStandardArgs)

# AMIO_FORCE_NCZARR is declared at the top level (option()) so that the Spack `~tensorstore` variant can wire it via `-DAMIO_FORCE_NCZARR=ON`. We
# tolerate it being undefined here -- treat that as OFF.
if(NOT DEFINED AMIO_FORCE_NCZARR)
    set(AMIO_FORCE_NCZARR OFF)
endif()

# Reset prior outputs so re-configure cleanly reflects current state.
set(TensorStore_FOUND FALSE)
set(TensorStore_REASON "")

if(AMIO_FORCE_NCZARR)
    set(TensorStore_REASON "AMIO_FORCE_NCZARR=ON: TensorStore discovery skipped by user request")
    message(STATUS "TensorStore discovery short-circuited (AMIO_FORCE_NCZARR=ON)")
else()
    # Defer to upstream's CONFIG-mode package file when present.  The design doc allows either FindTensorStore.cmake (this file) or an
    # upstream-provided tensorstoreConfig.cmake; we try CONFIG first because it is the supported integration path.
    find_package(TensorStore CONFIG QUIET)

    if(TensorStore_FOUND AND TARGET TensorStore::tensorstore)
        # Nothing further to do; CONFIG file exposed the imported target.
    else()
        set(TensorStore_FOUND FALSE)
        set(TensorStore_REASON "TensorStore not located via find_package(TensorStore CONFIG)")
    endif()
endif()

# Mirror into the AMIO-level feature flag.  Use CACHE INTERNAL so it survives across configure passes but is not user-tweakable (the correct knob is
# AMIO_FORCE_NCZARR, not the derived flag).
if(TensorStore_FOUND)
    set(AMIO_HAS_TENSORSTORE ON CACHE INTERNAL "TensorStore availability" FORCE)
else()
    set(AMIO_HAS_TENSORSTORE OFF CACHE INTERNAL "TensorStore availability" FORCE)
endif()

# Standard find-module reporting.  We pass FOUND_VAR explicitly so the QUIET / REQUIRED protocol wires through correctly when callers do
# `find_package(TensorStore REQUIRED)`.
find_package_handle_standard_args(TensorStore FOUND_VAR TensorStore_FOUND REQUIRED_VARS TensorStore_FOUND REASON_FAILURE_MESSAGE
                                                                                        "${TensorStore_REASON}")
