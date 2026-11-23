# relocate_openmp.cmake — POST_BUILD helper for the axis_py extension on macOS.
#
# Purpose (general, environment-agnostic):
#   Read the libomp dependency that the linker ACTUALLY recorded in the built
#   extension and rewrite any absolute reference to "@rpath/libomp.dylib". When
#   shared Kokkos has an absolute OpenMP dependency, use that exact dependency
#   instead: loading a second compiler-provided runtime can crash on first use.
#   For static Kokkos builds, RPATH relocation lets dyld resolve libomp against
#   the target's RPATHs instead of a fixed build-host path. Runtime conflicts
#   from other libraries in the Python process still require compatible builds.
#
#   Unlike enumerating known Homebrew / /usr/local paths, this inspects the
#   binary itself, so it works regardless of where libomp lives.
#
# Invoked via:  cmake -DTARGET_FILE=... -DINSTALL_NAME_TOOL=... -DOTOOL=... -P
#
# Required -D variables:
#   TARGET_FILE        — path to the built extension (.so / .dylib)
#   INSTALL_NAME_TOOL  — path to install_name_tool
#   OTOOL              — path to otool
# Optional -D variable:
#   KOKKOS_LIBRARY     — shared Kokkos core library whose runtime to reuse

if(NOT EXISTS "${TARGET_FILE}")
    message(WARNING "relocate_openmp: TARGET_FILE does not exist: ${TARGET_FILE}; skipping")
    return()
endif()

execute_process(
    COMMAND "${OTOOL}" -L "${TARGET_FILE}"
    OUTPUT_VARIABLE _deps
    RESULT_VARIABLE _otool_rc
)
if(NOT _otool_rc EQUAL 0)
    message(WARNING "relocate_openmp: otool failed on ${TARGET_FILE}; skipping")
    return()
endif()

string(REPLACE "\n" ";" _dep_lines "${_deps}")
set(_kokkos_omp_path "")
if(DEFINED KOKKOS_LIBRARY AND EXISTS "${KOKKOS_LIBRARY}")
    execute_process(
        COMMAND "${OTOOL}" -L "${KOKKOS_LIBRARY}"
        OUTPUT_VARIABLE _kokkos_deps
        RESULT_VARIABLE _kokkos_otool_rc
    )
    if(_kokkos_otool_rc EQUAL 0)
        string(REPLACE "\n" ";" _kokkos_dep_lines "${_kokkos_deps}")
        foreach(_line IN LISTS _kokkos_dep_lines)
            string(STRIP "${_line}" _line)
            if(_line MATCHES "^/.*(libomp|libgomp|libiomp5)[^ ]*\\.dylib")
                string(REGEX REPLACE " \\(compatibility.*$" "" _kokkos_omp_path "${_line}")
                break()
            endif()
        endforeach()
    endif()
endif()
foreach(_line IN LISTS _dep_lines)
    string(STRIP "${_line}" _line)
    # Match any OpenMP runtime flavor: libomp, libgomp, libiomp5.
    if(_line MATCHES "(libomp|libgomp|libiomp5)[^ ]*\\.dylib")
        # Strip the trailing " (compatibility version ..., current version ...)".
        string(REGEX REPLACE " \\(compatibility.*$" "" _dep_path "${_line}")
        string(STRIP "${_dep_path}" _dep_path)
        get_filename_component(_dep_name "${_dep_path}" NAME)
        set(_replacement "@rpath/${_dep_name}")
        if(_kokkos_omp_path)
            get_filename_component(_kokkos_omp_name "${_kokkos_omp_path}" NAME)
            if(_kokkos_omp_name STREQUAL _dep_name)
                set(_replacement "${_kokkos_omp_path}")
            endif()
        endif()
        if(_dep_path STREQUAL _replacement)
            message(STATUS "relocate_openmp: '${_dep_path}' already uses the selected runtime; no change")
        else()
            message(STATUS "relocate_openmp: rewriting '${_dep_path}' -> '${_replacement}'")
            execute_process(
                COMMAND "${INSTALL_NAME_TOOL}" -change "${_dep_path}" "${_replacement}" "${TARGET_FILE}"
                RESULT_VARIABLE _rc
            )
            if(NOT _rc EQUAL 0)
                message(WARNING "relocate_openmp: install_name_tool -change failed for '${_dep_path}'")
            endif()
        endif()
    endif()
endforeach()
