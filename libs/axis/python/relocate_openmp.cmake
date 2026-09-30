# relocate_openmp.cmake — POST_BUILD helper for the axis_py extension on macOS.
#
# Purpose (general, environment-agnostic):
#   Read the libomp dependency that the linker ACTUALLY recorded in the built
#   extension and rewrite any absolute reference to "@rpath/libomp.dylib". This
#   lets dyld resolve libomp against the target's RPATHs at runtime instead of a
#   build-host-specific absolute path. It fixes two problems at once:
#     1. "libomp.dylib not found" when the build-host path is absent on the
#        machine that later imports the module.
#     2. The duplicate OpenMP runtime crash (OMP: Error #15) when the host
#        process (NumPy / SciPy / PyTorch) has already loaded its own libomp.
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
foreach(_line IN LISTS _dep_lines)
    string(STRIP "${_line}" _line)
    # Match any OpenMP runtime flavor: libomp, libgomp, libiomp5.
    if(_line MATCHES "(libomp|libgomp|libiomp5)[^ ]*\\.dylib")
        # Strip the trailing " (compatibility version ..., current version ...)".
        string(REGEX REPLACE " \\(compatibility.*$" "" _dep_path "${_line}")
        string(STRIP "${_dep_path}" _dep_path)
        get_filename_component(_dep_name "${_dep_path}" NAME)
        if(_dep_path MATCHES "^@rpath/")
            message(STATUS "relocate_openmp: '${_dep_path}' already uses @rpath; no change")
        else()
            message(STATUS "relocate_openmp: rewriting '${_dep_path}' -> '@rpath/${_dep_name}'")
            execute_process(
                COMMAND "${INSTALL_NAME_TOOL}" -change "${_dep_path}" "@rpath/${_dep_name}" "${TARGET_FILE}"
                RESULT_VARIABLE _rc
            )
            if(NOT _rc EQUAL 0)
                message(WARNING "relocate_openmp: install_name_tool -change failed for '${_dep_path}'")
            endif()
        endif()
    endif()
endforeach()
