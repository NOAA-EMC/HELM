# ####################################################################################################################################################
# check_spack_no_rust.cmake
#
# Run as a `cmake -P` script from the CTest target `amio_spack_no_rust_cargo_go`.  Verifies R13.2 by concretizing the vendored Spack recipe at
# packages/amio/ and asserting that no Rust, Cargo, or Go toolchain appears in the dependency closure:
#
# spack -C <empty> -r ${AMIO_REPO_DIR} spec amio    \\\ | grep -E '(^|[^a-z])(rust|cargo|go)@'
#
# Must produce zero matching lines.
#
# Required CMake variables (set by the parent test): AMIO_SPACK_EXECUTABLE      absolute path to spack, or empty when missing AMIO_REPO_DIR absolute
# path to the directory that contains packages/amio/  (i.e. the `packages/` parent that should be passed to `spack repo add`) AMIO_REQUIRE_SPACK_CHECK
# ON to make a missing spack a hard failure (CI); OFF to report SKIPPED (developer hosts) AMIO_WORK_DIR scratch directory; created if missing
# ####################################################################################################################################################

foreach(_var IN ITEMS AMIO_REPO_DIR AMIO_REQUIRE_SPACK_CHECK AMIO_WORK_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "check_spack_no_rust: required variable ${_var} not set")
    endif()
endforeach()

if(NOT EXISTS "${AMIO_REPO_DIR}/amio/package.py")
    message(FATAL_ERROR "check_spack_no_rust: AMIO Spack recipe not found at " "'${AMIO_REPO_DIR}/amio/package.py'.  Task 1.3 must be "
                        "complete before this check can run.")
endif()

# When `spack` is unavailable, decide whether to skip or fail based on the AMIO_REQUIRE_SPACK_CHECK toggle.  CI sets this ON so the check cannot be
# silently bypassed; developer hosts leave it OFF.
if(NOT DEFINED AMIO_SPACK_EXECUTABLE OR "${AMIO_SPACK_EXECUTABLE}" STREQUAL "" OR "${AMIO_SPACK_EXECUTABLE}" STREQUAL
                                                                                  "AMIO_SPACK_EXECUTABLE-NOTFOUND")
    if(AMIO_REQUIRE_SPACK_CHECK)
        message(FATAL_ERROR "amio_spack_no_rust_cargo_go FAILED: `spack` is required\n" "by AMIO_REQUIRE_SPACK_CHECK=ON but was not found on the\n"
                            "PATH for this CI agent.  Install Spack, or unset the\n" "requirement, before re-running ctest.")
    else()
        # Returning normally (exit 0) plus the literal `SKIPPED:` token tells CTest to mark the test as skipped via the SKIP_REGULAR_EXPRESSION
        # property set by the parent.
        message(STATUS "amio_spack_no_rust_cargo_go: SKIPPED: `spack` not on " "PATH.  (Set -DAMIO_REQUIRE_SPACK_CHECK=ON to make this "
                       "a hard failure; CI agents do.)")
        return()
    endif()
endif()

file(MAKE_DIRECTORY "${AMIO_WORK_DIR}")

# Build a hermetic Spack config scope that: * registers the vendored recipe at AMIO_REPO_DIR ahead of any builtin repo (so this `amio` is always the
# one we vendored); * relocates Spack's install tree under AMIO_WORK_DIR so the check never touches the developer's $HOME/.spack opt tree.
#
# This avoids `spack repo add --scope=...`, which would mutate a shared config scope on the host.
set(_amio_scope "${AMIO_WORK_DIR}/spack-config")
file(MAKE_DIRECTORY "${_amio_scope}")
file(WRITE "${_amio_scope}/repos.yaml" "repos:\n" "  - ${AMIO_REPO_DIR}\n")
file(WRITE "${_amio_scope}/config.yaml" "config:\n" "  install_tree:\n" "    root: ${AMIO_WORK_DIR}/opt\n" "  misc_cache: ${AMIO_WORK_DIR}/cache\n")

# `spec --reuse=false` forces a fresh concretization so the result reflects the recipe alone, not whatever happens to be installed on the local Spack.
# `-C ${_amio_scope}` injects the config we just wrote at the highest precedence so the vendored repo is always picked up.
execute_process(
    COMMAND ${AMIO_SPACK_EXECUTABLE} -C ${_amio_scope} spec --reuse=false amio
    WORKING_DIRECTORY ${AMIO_WORK_DIR}
    RESULT_VARIABLE _spec_rc
    OUTPUT_VARIABLE _spec_out
    ERROR_VARIABLE _spec_err
    TIMEOUT 240)

if(NOT _spec_rc EQUAL 0)
    message(
        FATAL_ERROR
            "amio_spack_no_rust_cargo_go FAILED: `spack spec amio`\n"
            "returned ${_spec_rc}.\n"
            "  spack  : ${AMIO_SPACK_EXECUTABLE}\n"
            "  stdout :\n${_spec_out}\n"
            "  stderr :\n${_spec_err}\n"
            "\n"
            "If concretization itself failed because Spack tried to\n"
            "introduce a Rust/Cargo/Go dependency that conflicts with\n"
            "the recipe's hard `conflicts(...)` rules, that IS the\n"
            "violation -- inspect the conflict message and remove the\n"
            "transitive dependency that pulled the toolchain in.")
endif()

# Match `rust@`, `cargo@`, `go@` (and similarly `rust:`/`cargo:` tokens that some `spack spec` formats use), but reject false positives like
# `mongodb-go-driver`, `polkadotgo`, or `cargo-bay` by requiring the token to be at the start of a word.
set(_violations "")
string(REPLACE "\n" ";" _spec_lines "${_spec_out}")
foreach(_line IN LISTS _spec_lines)
    if("${_line}" MATCHES "(^|[^A-Za-z0-9_-])(rust|cargo|go)([@ \t:].*)?$")
        list(APPEND _violations "${_line}")
    endif()
endforeach()

if(_violations)
    string(REPLACE ";" "\n    " _violation_block "${_violations}")
    message(
        FATAL_ERROR
            "amio_spack_no_rust_cargo_go FAILED.\n"
            "  `spack spec amio` reported one or more Rust/Cargo/Go\n"
            "  dependencies in the closure (R13.2):\n"
            "    ${_violation_block}\n\n"
            "AMIO disallows Rust, Cargo, and Go in its dependency\n"
            "closure (Requirement R13.2).  Either remove the offending\n"
            "transitive dependency, or change the recipe variant that\n"
            "introduced it.\n"
            "\n"
            "Full `spack spec` output for reference:\n${_spec_out}")
endif()

message(STATUS "amio_spack_no_rust_cargo_go: PASSED " "(no rust/cargo/go in closure of vendored Spack recipe).")
