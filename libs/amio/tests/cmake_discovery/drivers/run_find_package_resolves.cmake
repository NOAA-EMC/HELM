# ####################################################################################################################################################
# run_find_package_resolves.cmake
#
# Driver script for the `amio_discovery_find_package_resolves_without_rust_cargo_go` CTest entry.
#
# Verifies that `find_package(AMIO)` resolves cleanly on a host without Rust, Cargo, or Go (R13.1, R13.2, R13.3).
#
# The test runs in two phases:
#
# Phase 1 -- always-on, hermetic checks: 1a. Rust/Cargo/Go absent from the host PATH (precondition for the live half of R13.2). 1b. AMIO's CMake
# configuration files contain no find_package / find_program / Find<Rust|Cargo|Go> references that could pull a prohibited toolchain into the
# resolution path. This is a structural guard against regressions.
#
# Phase 2 -- live consumer configure (best-effort): 2a. Configure AMIO with -DAMIO_FORCE_NCZARR=ON. 2b. Install AMIO into a temporary prefix. 2c.
# Configure a tiny consumer project that calls find_package(AMIO REQUIRED PATHS <prefix>) and links AMIO::amio_core. If any step in phase 2 fails
# because of a missing AMIO runtime dependency (eckit, netCDF, MPI), the driver emits "[SKIP] ..." rather than failing -- such hosts cannot complete a
# full AMIO configure regardless of the find_package machinery.
#
# Output markers: AMIO_FIND_PACKAGE_RESOLVES=OK   -- all enabled checks passed AMIO_FIND_PACKAGE_RESOLVES=FAIL -- any enabled check failed [SKIP] ...
# -- live half could not run
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)

foreach(_var IN ITEMS AMIO_TEST_CMAKE AMIO_TEST_GENERATOR AMIO_SOURCE_DIR AMIO_TEST_WORK_DIR AMIO_TEST_CONSUMER_SRC)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "${_var} must be set on the cmake -P command line")
    endif()
endforeach()

set(_failures "")
set(_skipped FALSE)
set(_skip_reason "")

# ####################################################################################################################################################
# Phase 1a: host has no rust / cargo / go on PATH.
# ####################################################################################################################################################
foreach(_tool IN ITEMS rustc cargo go)
    find_program(_tool_path_${_tool} ${_tool})
    if(_tool_path_${_tool})
        list(APPEND _failures "${_tool} is present on PATH at ${_tool_path_${_tool}} -- " "test precondition violated")
    endif()
    unset(_tool_path_${_tool} CACHE)
endforeach()

# ####################################################################################################################################################
# Phase 1b: AMIO's build configuration must not reference Rust / Cargo / Go in any form that could produce a toolchain dependency.
#
# Files inspected: * Top-level CMakeLists.txt * cmake/AMIOConfig.cmake.in * cmake/FindTensorStore.cmake * cmake/AMIOZarrBackend.cmake
#
# We look for case-insensitive matches of find_package\s*\(\s*(rust|cargo|go) find_program\s*\(\s*[^)]*\b(rustc|cargo|go)\b FindRust|FindCargo|FindGo
# (cmake module names) in those files.
#
# (R13.2's negative-existence claim must hold structurally, not just when the host happens to lack the toolchains.)
# ####################################################################################################################################################
set(_inspect_files "${AMIO_SOURCE_DIR}/CMakeLists.txt" "${AMIO_SOURCE_DIR}/cmake/AMIOConfig.cmake.in"
                   "${AMIO_SOURCE_DIR}/cmake/FindTensorStore.cmake" "${AMIO_SOURCE_DIR}/cmake/AMIOZarrBackend.cmake")

set(_forbidden_patterns "find_package[ \t]*\\([ \t]*(Rust|Cargo|Go)[ \t)]" "find_program[ \t]*\\([^)]*[ \t]+(rustc|cargo|go)[ \t)]"
                        "FindRust\\.cmake|FindCargo\\.cmake|FindGo\\.cmake")

foreach(_file IN LISTS _inspect_files)
    if(NOT EXISTS "${_file}")
        # Only AMIOZarrBackend.cmake might be optional during early bootstrap; missing files are non-fatal.
        continue()
    endif()
    file(READ "${_file}" _content)
    foreach(_pat IN LISTS _forbidden_patterns)
        string(REGEX MATCH "${_pat}" _hit "${_content}")
        if(_hit)
            list(APPEND _failures "${_file} matches forbidden pattern '${_pat}': '${_hit}'")
            if(_pat MATCHES "Rust|rustc")
                list(APPEND _failures "rust toolchain leaked into AMIO build")
            elseif(_pat MATCHES "Cargo|cargo")
                list(APPEND _failures "cargo toolchain leaked into AMIO build")
            elseif(_pat MATCHES "Go|FindGo|[ \t]go[ \t]")
                list(APPEND _failures "go toolchain leaked into AMIO build")
            endif()
        endif()
    endforeach()
endforeach()

# ####################################################################################################################################################
# Phase 2: live consumer configure.
#
# Best-effort -- a host without eckit / netCDF / MPI cannot run AMIO configure to completion, but that does not invalidate R13.1's claim about the
# find_package interface.  We emit [SKIP] in that case and let CTest record SKIPPED.
# ####################################################################################################################################################
file(REMOVE_RECURSE "${AMIO_TEST_WORK_DIR}")
set(_amio_build "${AMIO_TEST_WORK_DIR}/amio-build")
set(_amio_install "${AMIO_TEST_WORK_DIR}/amio-install")
set(_consumer_build "${AMIO_TEST_WORK_DIR}/consumer-build")
file(MAKE_DIRECTORY "${_amio_build}")
file(MAKE_DIRECTORY "${_amio_install}")
file(MAKE_DIRECTORY "${_consumer_build}")

execute_process(
    COMMAND "${AMIO_TEST_CMAKE}" -S "${AMIO_SOURCE_DIR}" -B "${_amio_build}" -G "${AMIO_TEST_GENERATOR}" -DAMIO_FORCE_NCZARR=ON
            -DAMIO_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${_amio_install} -DBUILD_SHARED_LIBS=ON
    OUTPUT_VARIABLE _amio_cfg_stdout
    ERROR_VARIABLE _amio_cfg_stderr
    RESULT_VARIABLE _amio_cfg_rc)

set(_amio_cfg_combined "${_amio_cfg_stdout}\n${_amio_cfg_stderr}")
message(STATUS "AMIO configure rc=${_amio_cfg_rc}")
message(STATUS "AMIO configure output:\n${_amio_cfg_combined}")

if(NOT _amio_cfg_rc EQUAL 0)
    # Inspect the failure reason.  If it is the documented "no Zarr backend available" FATAL_ERROR (no eckit / no netCDF / no NCZarr capability), this
    # host genuinely cannot run a live consumer test -- skip it.  Any other failure mode counts as a genuine test failure.
    set(_skip_markers "no Zarr backend available" "Could NOT find eckit" "Could NOT find netCDF" "Could NOT find MPI" "Could NOT find mdspan")
    set(_skip_matched FALSE)
    foreach(_marker IN LISTS _skip_markers)
        string(FIND "${_amio_cfg_combined}" "${_marker}" _h)
        if(NOT _h EQUAL -1)
            set(_skip_matched TRUE)
            set(_skip_reason "AMIO live configure cannot complete on this host: ${_marker}")
            break()
        endif()
    endforeach()
    if(_skip_matched)
        set(_skipped TRUE)
    else()
        list(APPEND _failures "AMIO live configure failed (rc=${_amio_cfg_rc}) with no recognized skip reason")
    endif()
else()
    # Build + install AMIO.
    execute_process(
        COMMAND "${AMIO_TEST_CMAKE}" --build "${_amio_build}" --target install
        OUTPUT_VARIABLE _build_stdout
        ERROR_VARIABLE _build_stderr
        RESULT_VARIABLE _build_rc)
    if(NOT _build_rc EQUAL 0)
        list(APPEND _failures "AMIO build/install failed (rc=${_build_rc})")
        message(STATUS "AMIO build output:\n${_build_stdout}\n${_build_stderr}")
    else()
        # Live consumer configure -- the actual R13.1 acceptance criterion: find_package(AMIO REQUIRED) succeeds with no additional include / link /
        # flag plumbing on the consumer side.
        execute_process(
            COMMAND "${AMIO_TEST_CMAKE}" -S "${AMIO_TEST_CONSUMER_SRC}" -B "${_consumer_build}" -G "${AMIO_TEST_GENERATOR}"
                    -DAMIO_INSTALL_PREFIX=${_amio_install}
            OUTPUT_VARIABLE _cons_stdout
            ERROR_VARIABLE _cons_stderr
            RESULT_VARIABLE _cons_rc)
        message(STATUS "Consumer configure rc=${_cons_rc}")
        message(STATUS "Consumer configure output:\n${_cons_stdout}\n${_cons_stderr}")
        if(NOT _cons_rc EQUAL 0)
            list(APPEND _failures "consumer find_package(AMIO REQUIRED) failed (rc=${_cons_rc})")
        else()
            # The consumer's configure-time emit of "CONSUMER_RESOLVED_AMIO=OK" is our final positive signal.
            string(FIND "${_cons_stdout}\n${_cons_stderr}" "CONSUMER_RESOLVED_AMIO=OK" _hit_consumer)
            if(_hit_consumer EQUAL -1)
                list(APPEND _failures "consumer did not emit CONSUMER_RESOLVED_AMIO=OK")
            endif()
        endif()
    endif()
endif()

# ####################################################################################################################################################
# Final marker emission.
# ####################################################################################################################################################
if(_failures)
    foreach(_msg IN LISTS _failures)
        message(STATUS "FAIL: ${_msg}")
    endforeach()
    message(FATAL_ERROR "AMIO_FIND_PACKAGE_RESOLVES=FAIL")
endif()

if(_skipped)
    message(STATUS "[SKIP] ${_skip_reason}")
    # Phase 1's hermetic checks all passed; record them as the OK signal so the test reports as SKIPPED with phase-1 evidence.
    return()
endif()

message(STATUS "AMIO_FIND_PACKAGE_RESOLVES=OK")
