# ####################################################################################################################################################
# tests/header_isolation/CMakeLists.txt
#
# Implements task 1.4: the four header-isolation CI checks that run on every PR and gate merge.  Each check is registered as a CTest target executing
# a small, self-contained CMake `-P` script so that the tests are portable across CI agents and produce actionable error messages without an external
# test runner.
#
# 1. amio_header_c99_isolation     - R10.1, R10.2, R13.6
# 2. amio_header_cxx_inclusion     - R10.3
# 3. amio_symbol_mangling          - R10.5, R10.8
# 4. amio_spack_no_rust_cargo_go   - R13.2
#
# All four tests share the `header_isolation` CTest label, so CI can scope to them with:   ctest -L header_isolation --output-on-failure
# ####################################################################################################################################################

set(_amio_root "${CMAKE_SOURCE_DIR}")
set(_amio_public_header "${CMAKE_SOURCE_DIR}/include/amio/amio.h")
set(_amio_scripts "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
set(_amio_test_workdir "${CMAKE_CURRENT_BINARY_DIR}")

# ####################################################################################################################################################
# 1. C99 isolation compile (R10.1, R10.2, R13.6)
#
# Run a clean C99 compile of include/amio/amio.h with `-pedantic -Werror` and a deliberately empty include-path set (the project's own include/ tree
# is supplied via -I; nothing else).  Any token or header that is not in {<stdint.h>, <stddef.h>, <stdbool.h>} or in the project's own include tree
# fails the build.
# ####################################################################################################################################################

add_test(
    NAME amio_header_c99_isolation
    COMMAND ${CMAKE_COMMAND} -DAMIO_C_COMPILER=${CMAKE_C_COMPILER} -DAMIO_PUBLIC_HEADER=${_amio_public_header}
            -DAMIO_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include -DAMIO_WORK_DIR=${_amio_test_workdir}/c99 -P ${_amio_scripts}/check_c99_isolation.cmake)
set_tests_properties(amio_header_c99_isolation PROPERTIES LABELS "header_isolation;c99" TIMEOUT 60)

# ####################################################################################################################################################
# 1. C++ inclusion compile (R10.3)
#
# Compile include/amio/amio.h as C++20.  The `extern "C"` guard must parse cleanly so a C++ host application can include the header.
# ####################################################################################################################################################

add_test(
    NAME amio_header_cxx_inclusion
    COMMAND ${CMAKE_COMMAND} -DAMIO_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DAMIO_PUBLIC_HEADER=${_amio_public_header}
            -DAMIO_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include -DAMIO_WORK_DIR=${_amio_test_workdir}/cxx -P ${_amio_scripts}/check_cxx_inclusion.cmake)
set_tests_properties(amio_header_cxx_inclusion PROPERTIES LABELS "header_isolation;cxx" TIMEOUT 60)

# ####################################################################################################################################################
# 1. Symbol mangling check (R10.5, R10.8)
#
# `nm -D $<TARGET_FILE:amio_core>` must list every text-segment (T/t) global symbol with the `amio_` C99 prefix.  Any C++ mangled (`_Z...`) symbol or
# any non-amio_ exported text symbol fails the check.  Generator expressions resolve at test-time so the test always picks up the freshly-built
# libamio.
# ####################################################################################################################################################

find_program(AMIO_NM_EXECUTABLE nm)
if(NOT AMIO_NM_EXECUTABLE)
    message(WARNING "tests/header_isolation: `nm` not found on PATH; the symbol-" "mangling check will report SKIPPED at test time.")
endif()

add_test(NAME amio_symbol_mangling COMMAND ${CMAKE_COMMAND} -DAMIO_NM_EXECUTABLE=${AMIO_NM_EXECUTABLE} -DAMIO_LIBRARY_FILE=$<TARGET_FILE:amio_core>
                                           -P ${_amio_scripts}/check_symbol_mangling.cmake)
set_tests_properties(amio_symbol_mangling PROPERTIES LABELS "header_isolation;symbols" TIMEOUT 60 SKIP_REGULAR_EXPRESSION "SKIPPED:")

# ####################################################################################################################################################
# 1. No-Rust / Cargo / Go dependency closure check (R13.2)
#
# `spack spec amio | grep -E 'rust|cargo|go@'` must produce zero matching lines.  When `spack` is absent on the local developer host, the test reports
# SKIPPED (CTest exit 77) so day-to-day `ctest` runs are not blocked.  CI agents set `-DAMIO_REQUIRE_SPACK_CHECK=ON` so a missing `spack` becomes a
# hard failure - the gate cannot be silently bypassed.
#
# The check operates on the vendored recipe at packages/amio.
# ####################################################################################################################################################

option(AMIO_REQUIRE_SPACK_CHECK "Treat absent `spack` as a hard failure for amio_spack_no_rust_cargo_go (CI)" OFF)

find_program(AMIO_SPACK_EXECUTABLE spack)
if(NOT AMIO_SPACK_EXECUTABLE AND NOT AMIO_REQUIRE_SPACK_CHECK)
    message(STATUS "tests/header_isolation: `spack` not found on PATH; the " "no-Rust/Cargo/Go closure check will report SKIPPED at test "
                   "time.  Set -DAMIO_REQUIRE_SPACK_CHECK=ON in CI to make " "missing spack a hard failure.")
endif()

add_test(
    NAME amio_spack_no_rust_cargo_go
    COMMAND
        ${CMAKE_COMMAND} -DAMIO_SPACK_EXECUTABLE=${AMIO_SPACK_EXECUTABLE} -DAMIO_REPO_DIR=${CMAKE_SOURCE_DIR}/packages
        -DAMIO_REQUIRE_SPACK_CHECK=${AMIO_REQUIRE_SPACK_CHECK} -DAMIO_WORK_DIR=${_amio_test_workdir}/spack -P
        ${_amio_scripts}/check_spack_no_rust.cmake)
set_tests_properties(amio_spack_no_rust_cargo_go PROPERTIES LABELS "header_isolation;spack" TIMEOUT 300 SKIP_REGULAR_EXPRESSION "SKIPPED:")
