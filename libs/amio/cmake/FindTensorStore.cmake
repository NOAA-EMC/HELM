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
    # Prefer an installed upstream package. The opt-in FetchContent path is intended for reproducible backend qualification in environments where
    # TensorStore is not packaged.
    find_package(TensorStore CONFIG QUIET)

    if(TensorStore_FOUND AND TARGET TensorStore::tensorstore)
        # Nothing further to do; CONFIG file exposed the imported target.
    elseif(AMIO_FETCH_TENSORSTORE)
        include(FetchContent)
        FetchContent_Declare(
            tensorstore
            GIT_REPOSITORY https://github.com/google/tensorstore.git
            GIT_TAG v0.1.86
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(tensorstore)
        # cece-dev uses strict C99 mode, which hides glibc's POSIX barrier declarations in the bundled Blosc C source unless a feature level is
        # requested explicitly.
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET Blosc_blosc)
            target_compile_definitions(Blosc_blosc PRIVATE _GNU_SOURCE)
        endif()
        if(TARGET tensorstore::tensorstore)
            get_target_property(_amio_tensorstore_target tensorstore::tensorstore ALIASED_TARGET)
            if(NOT _amio_tensorstore_target)
                set(_amio_tensorstore_target tensorstore::tensorstore)
            endif()
            add_library(TensorStore::tensorstore ALIAS ${_amio_tensorstore_target})
            set(TensorStore_FOUND TRUE)
            # TensorStore v0.1.86 and its bundled Abseil are compiled as C++17. AMIO itself is C++20, and Abseil's default option (2) changes
            # absl::SourceLocation's ABI based on the language mode. Provide a build-tree options.h with the stable implementation for AMIO
            # translation units that consume the fetched TensorStore headers.
            set(_amio_absl_options "${absl_SOURCE_DIR}/absl/base/options.h")
            if(EXISTS "${_amio_absl_options}")
                file(READ "${_amio_absl_options}" _amio_absl_options_text)
                string(REPLACE "#define ABSL_OPTION_USE_STD_SOURCE_LOCATION 2" "#define ABSL_OPTION_USE_STD_SOURCE_LOCATION 0"
                               _amio_absl_options_text "${_amio_absl_options_text}")
                set(_amio_absl_abi_include "${CMAKE_CURRENT_BINARY_DIR}/amio-tensorstore-abi")
                set(AMIO_TENSORSTORE_ABI_INCLUDE_DIR "${_amio_absl_abi_include}")
                file(MAKE_DIRECTORY "${_amio_absl_abi_include}/absl/base")
                file(WRITE "${_amio_absl_abi_include}/absl/base/options.h" "${_amio_absl_options_text}")
                add_library(amio_tensorstore_abi INTERFACE)
                target_include_directories(amio_tensorstore_abi BEFORE INTERFACE "${_amio_absl_abi_include}")
                # The TensorStore C++ convenience target omits optional driver registrations; force-link the Zarr v3 driver implementation.
                if(TARGET tensorstore::driver_zarr3)
                    target_link_libraries(amio_tensorstore_abi INTERFACE tensorstore::driver_zarr3)
                endif()
                if(TARGET tensorstore::kvstore_file)
                    target_link_libraries(amio_tensorstore_abi INTERFACE tensorstore::kvstore_file)
                endif()
                # AMIO's legacy TensorStore driver uses dynamic dimension expressions; the Bazel-to-CMake convenience target omits their operation
                # libraries from its public link interface.
                foreach(_amio_ts_index_target IN ITEMS tensorstore::index_space_dimension_selection tensorstore::index_space_interval_slice_op
                                                       tensorstore::index_space_translate_op)
                    if(TARGET ${_amio_ts_index_target})
                        target_link_libraries(amio_tensorstore_abi INTERFACE ${_amio_ts_index_target})
                    endif()
                endforeach()
            endif()
        else()
            set(TensorStore_REASON "Fetched TensorStore v0.1.86 did not define tensorstore::tensorstore")
        endif()
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
