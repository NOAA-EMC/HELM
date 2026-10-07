# ####################################################################################################################################################
# tests/cmake_discovery/CMakeLists.txt
#
# CTest entries that verify the configure-time TensorStore discovery and NCZarr fallback flow described in design.md (mermaid diagram "TensorStore
# discovery and NCZarr fallback (configure-time)").
#
# Validates Requirements R13.3, R13.4, R13.5:
#
# * R13.3 -- TensorStore discovery failing -> NCZarr fallback mode.
# * R13.4 -- Fallback configure emits a STATUS or WARNING message naming the fallback mode and the reason TensorStore was not selected.
# * R13.5 -- AMIO_FORCE_NCZARR=ON skips TensorStore discovery and selects the NCZarr code path (the build-time half of R13.5; the runtime Round_Trip
#   half is exercised by P1 in tasks 13.x).
#
# Each test runs a `cmake -P` driver script that:
#
# 1. invokes a sub-CMake configure of a small probe project,
# 2. captures both stdout and stderr,
# 3. uses string(FIND ...) / string(REGEX MATCH ...) to verify every documented observable from the mermaid flow,
# 4. emits a single-line marker that CTest's PASS_REGULAR_EXPRESSION anchors against.
#
# Driver-script-based verification is preferred over CTest's native PASS_REGULAR_EXPRESSION because (a) message(WARNING) output spans multiple lines
# and CMake's `.` does not match newlines, (b) the driver script can ensure all markers are present (AND semantics) instead of CTest's native OR
# semantics.
#
# Probes are deliberately self-contained: they do not require netCDF, MPI, or any other AMIO runtime dependency to be installed. This makes the tests
# runnable on developer workstations and the minimal CI image that exercises only the configure-time flow.
# ####################################################################################################################################################

set(_amio_probe_src ${CMAKE_CURRENT_SOURCE_DIR}/probes)
set(_amio_probe_bin ${CMAKE_CURRENT_BINARY_DIR}/probes)
set(_amio_probe_drv ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
file(MAKE_DIRECTORY ${_amio_probe_bin})

# Resolve the AMIO source root.  CMAKE_CURRENT_SOURCE_DIR points at tests/cmake_discovery/ regardless of how this file is included, so walking up two
# levels gives us the AMIO repo root.  This is more robust than ${PROJECT_SOURCE_DIR} because it works correctly when tests/CMakeLists.txt is added by
# a wrapper project (developer sanity harness) as well as by AMIO's own CMakeLists.txt.
get_filename_component(_amio_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# ####################################################################################################################################################
# Test A (R13.5): AMIO_FORCE_NCZARR=ON skips TensorStore discovery and yields AMIO_HAS_TENSORSTORE=OFF.
#
# Mermaid path verified: A[CMake configure] -> B{AMIO_FORCE_NCZARR?} B -- ON --> F[Set AMIO_HAS_TENSORSTORE=OFF; STATUS: TensorStore discovery
# short-circuited]
# ####################################################################################################################################################

add_test(
    NAME amio_discovery_force_nczarr_disables_tensorstore
    COMMAND
        ${CMAKE_COMMAND} -DAMIO_TEST_CMAKE=${CMAKE_COMMAND} -DAMIO_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DAMIO_TEST_PROBE_SRC=${_amio_probe_src}/force_nczarr -DAMIO_TEST_PROBE_BIN=${_amio_probe_bin}/force_nczarr
        -DAMIO_SOURCE_DIR=${_amio_repo_root} -P ${_amio_probe_drv}/run_force_nczarr.cmake)

set_tests_properties(
    amio_discovery_force_nczarr_disables_tensorstore PROPERTIES PASS_REGULAR_EXPRESSION "AMIO_DISCOVERY_FORCE_NCZARR=OK" FAIL_REGULAR_EXPRESSION
                                                                "AMIO_DISCOVERY_FORCE_NCZARR=FAIL" LABELS "cmake;discovery;force_nczarr")

# ####################################################################################################################################################
# Test B (R13.3 + R13.4): TensorStore unavailable -> NCZarr fallback WARNING is emitted, naming the fallback mode and the reason.
#
# Mermaid path verified: A -> B{AMIO_FORCE_NCZARR?} -- OFF --> C[find_package(TensorStore CONFIG)] -> D{TensorStore_FOUND?} -- NO --> F (fallback
# warning)
# ####################################################################################################################################################

add_test(
    NAME amio_discovery_tensorstore_absent_emits_fallback_warning
    COMMAND
        ${CMAKE_COMMAND} -DAMIO_TEST_CMAKE=${CMAKE_COMMAND} -DAMIO_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DAMIO_TEST_PROBE_SRC=${_amio_probe_src}/fallback_warning -DAMIO_TEST_PROBE_BIN=${_amio_probe_bin}/fallback_warning
        -DAMIO_SOURCE_DIR=${_amio_repo_root} -P ${_amio_probe_drv}/run_fallback_warning.cmake)

set_tests_properties(
    amio_discovery_tensorstore_absent_emits_fallback_warning
    PROPERTIES PASS_REGULAR_EXPRESSION "AMIO_DISCOVERY_FALLBACK_WARNING=OK" FAIL_REGULAR_EXPRESSION "AMIO_DISCOVERY_FALLBACK_WARNING=FAIL" LABELS
               "cmake;discovery;fallback_warning")

# ####################################################################################################################################################
# Test C (R13.1 + R13.2 + R13.3): find_package(AMIO) resolves on a host without Rust, Cargo, or Go.
#
# This driver script performs:
#
# Phase 1 (always-on, hermetic): - Static guard against any rust/cargo/go references in cmake/AMIOConfig.cmake.in or in CMakeLists.txt's
# find_package() / find_program() / depends_on() calls.  A failure here means a future change could pull a prohibited toolchain into the AMIOConfig
# resolution path. - Verify rustc, cargo, go are absent from the test host's PATH so the precondition "host without Rust/Cargo/Go installed" is
# genuinely satisfied (R13.2 host-side precondition).
#
# Phase 2 (live, conditional): - Build AMIO with -DAMIO_FORCE_NCZARR=ON into a temp build tree, install into a temp prefix, then configure a tiny
# downstream consumer project with `find_package(AMIO REQUIRED PATHS <prefix>)` and build/link its C99 executable. Success proves installed targets
# work end-to-end without any prohibited toolchain. - If AMIO's own configure cannot complete on this host (for example, no netCDF-c with NCZarr
# support is installed), the driver emits a "[SKIP] ..." line; CTest's SKIP_REGULAR_EXPRESSION turns that into a SKIPPED outcome rather than a FAIL.
# ####################################################################################################################################################

# This live test starts fresh CMake processes: FetchContent targets and package redirects in the parent build are not installed dependencies. CI may
# prepare a real prefix after configure and before CTest; do not require it to exist yet.
set(AMIO_DISCOVERY_DEPENDENCY_PREFIX_PATH "" CACHE STRING "Installed dependency prefixes for the live AMIO package-discovery test")
set(_amio_find_package_cache "${CMAKE_CURRENT_BINARY_DIR}/find-package-cache.cmake")
file(WRITE "${_amio_find_package_cache}" "# Generated test-only settings. No parent dependency provider or package redirects.\n")

# Use an initial-cache file instead of expanding list-valued -D arguments through add_test/execute_process. In particular, preserve spaces and
# semicolons in a multi-prefix CMAKE_PREFIX_PATH. Adapt the bracket delimiter for unusual paths.
function(_amio_discovery_cache_entry name value)
    set(_equals "=")
    while("${value}" MATCHES "\\]${_equals}\\]")
        string(APPEND _equals "=")
    endwhile()
    file(APPEND "${_amio_find_package_cache}" "set(${name} [${_equals}[${value}]${_equals}] CACHE STRING \"Inherited test setting\" FORCE)\n")
endfunction()
set(_amio_discovery_prefixes "${AMIO_DISCOVERY_DEPENDENCY_PREFIX_PATH}")
list(APPEND _amio_discovery_prefixes ${CMAKE_PREFIX_PATH})
list(REMOVE_ITEM _amio_discovery_prefixes "")
_amio_discovery_cache_entry(CMAKE_PREFIX_PATH "${_amio_discovery_prefixes}")
foreach(
    _setting IN
    ITEMS CMAKE_TOOLCHAIN_FILE
          CMAKE_MAKE_PROGRAM
          CMAKE_BUILD_TYPE
          CMAKE_FIND_ROOT_PATH
          CMAKE_C_COMPILER
          CMAKE_CXX_COMPILER
          CMAKE_Fortran_COMPILER
          CMAKE_DISABLE_FIND_PACKAGE_netCDFCxx)
    if(DEFINED ${_setting})
        _amio_discovery_cache_entry("${_setting}" "${${_setting}}")
    endif()
endforeach()
foreach(_language IN ITEMS C CXX Fortran)
    foreach(_suffix IN ITEMS "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
        set(_setting "CMAKE_${_language}_FLAGS${_suffix}")
        if(DEFINED ${_setting})
            _amio_discovery_cache_entry("${_setting}" "${${_setting}}")
        endif()
    endforeach()
endforeach()

add_test(
    NAME amio_discovery_find_package_resolves_without_rust_cargo_go
    COMMAND
        ${CMAKE_COMMAND} -DAMIO_TEST_CMAKE=${CMAKE_COMMAND} -DAMIO_TEST_GENERATOR=${CMAKE_GENERATOR} -DAMIO_SOURCE_DIR=${_amio_repo_root}
        -DAMIO_TEST_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/find_package -DAMIO_TEST_CONSUMER_SRC=${_amio_probe_src}/find_package_consumer
        -DAMIO_TEST_CACHE_FILE=${_amio_find_package_cache} -DAMIO_TEST_CONFIG=$<CONFIG> -DAMIO_TEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
        -DAMIO_TEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET} -P ${_amio_probe_drv}/run_find_package_resolves.cmake)

set_tests_properties(
    amio_discovery_find_package_resolves_without_rust_cargo_go
    PROPERTIES PASS_REGULAR_EXPRESSION
               "AMIO_FIND_PACKAGE_RESOLVES=OK"
               FAIL_REGULAR_EXPRESSION
               "AMIO_FIND_PACKAGE_RESOLVES=FAIL"
               SKIP_REGULAR_EXPRESSION
               "\\[SKIP\\]"
               LABELS
               "cmake;discovery;find_package;no_rust_no_cargo_no_go")
