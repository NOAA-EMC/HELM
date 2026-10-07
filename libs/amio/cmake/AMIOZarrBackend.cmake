# ####################################################################################################################################################
# AMIOZarrBackend.cmake
#
# Encapsulates AMIO's Zarr-backend selection / messaging block so the top-level CMakeLists.txt and the build-time verification tests in
# tests/cmake_discovery/ both consume the same logic.
#
# Inputs (read from caller scope):
#
# AMIO_FORCE_NCZARR        - ON/OFF (option set in CMakeLists.txt) AMIO_HAS_TENSORSTORE     - ON/OFF (set by FindTensorStore.cmake) TensorStore_REASON
# - human-readable reason string (set by FindTensorStore.cmake when not found or when discovery was skipped) AMIO_NETCDF_HAS_NCZARR   - TRUE/FALSE
# (set by netcdf_meta.h / netcdf.h symbol probe in CMakeLists.txt)
#
# Outputs (set in caller scope via PARENT_SCOPE):
#
# AMIO_ZARR_BACKEND_DESC   - human-readable backend name; appears verbatim in the configure summary line "AMIO Zarr backend: <DESC>"
# AMIO_ZARR_BACKEND_REASON - reason text emitted alongside the fallback summary; empty when the TensorStore branch is selected
#
# Side effects (the verifiable contract):
#
# * AMIO_HAS_TENSORSTORE=ON message(STATUS) -- "AMIO Zarr backend: TensorStore (...)"
#
# * AMIO_HAS_TENSORSTORE=OFF, AMIO_NETCDF_HAS_NCZARR=TRUE, AMIO_FORCE_NCZARR=ON message(STATUS) -- "AMIO Zarr backend: NCZarr fallback ..." followed
#   by "   reason: <REASON>" (User explicitly requested fallback; no severity escalation.)
#
# * AMIO_HAS_TENSORSTORE=OFF, AMIO_NETCDF_HAS_NCZARR=TRUE, AMIO_FORCE_NCZARR=OFF message(WARNING) -- "AMIO Zarr backend: NCZarr fallback ... reason:
#   <REASON>" (single multi-line warning; matches R13.4: WARNING that names the fallback and the reason).
#
# * AMIO_HAS_TENSORSTORE=OFF, AMIO_NETCDF_HAS_NCZARR=FALSE message(FATAL_ERROR) -- "AMIO: no Zarr backend available." (R13.5: hard configuration
#   error.)
#
# Requirements implemented: R13.3, R13.4, R13.5, R8.6.
# ####################################################################################################################################################

include_guard(GLOBAL)

function(amio_select_zarr_backend)
    # The function reads variables from its parent (caller) scope. CMake's variable-lookup rules give us read access automatically via the dynamic
    # scope; explicit `set(... PARENT_SCOPE)` is used only for the named outputs.

    if(AMIO_HAS_TENSORSTORE)
        set(_desc "TensorStore (Zarr v3, sharding, cloud KvStore)")
        set(_reason "")

        # STATUS-level summary; the configure-summary block in CMakeLists.txt re-emits this line so Spack logs are immediately greppable for the
        # active Zarr path.
        message(STATUS "AMIO Zarr backend: ${_desc}")

    elseif(AMIO_NETCDF_HAS_NCZARR)
        set(_desc "NCZarr fallback (sharding unavailable)")

        if(AMIO_FORCE_NCZARR)
            set(_reason "AMIO_FORCE_NCZARR=ON; TensorStore discovery skipped")
            # User explicitly requested fallback -- STATUS is the appropriate severity (not a surprise to the operator).
            message(STATUS "AMIO Zarr backend: ${_desc}")
            message(STATUS "   reason: ${_reason}")
        else()
            set(_reason "${TensorStore_REASON}")
            # Genuine fallback because TensorStore could not be located -- WARNING severity per R13.4 ("emit a CMake status or warning message ...
            # that names the fallback mode and identifies the reason TensorStore was not selected").  We use WARNING so the message is visible even in
            # non-verbose configure runs.
            message(WARNING "AMIO Zarr backend: ${_desc}\n" "   reason: ${_reason}")
        endif()

    else()
        # Neither TensorStore nor NCZarr-capable netCDF-c is available -- the Zarr_Driver cannot be built.  R13.5 hard configuration error.
        message(
            FATAL_ERROR
                "AMIO: no Zarr backend available.\n"
                "  TensorStore: not found (${TensorStore_REASON})\n"
                "  netCDF-c NCZarr support: not detected "
                "(neither NC_HAS_NCZARR nor NC_FORMATX_NCZARR found in "
                "netcdf.h, or netCDF not found)\n"
                "Either install Google TensorStore (and re-run with "
                "find_package(TensorStore) succeeding), or install "
                "netCDF-c built with --enable-nczarr "
                "(Spack: netcdf-c+nczarr+blosc+zstd).")
    endif()

    set(AMIO_ZARR_BACKEND_DESC "${_desc}" PARENT_SCOPE)
    set(AMIO_ZARR_BACKEND_REASON "${_reason}" PARENT_SCOPE)
endfunction()
