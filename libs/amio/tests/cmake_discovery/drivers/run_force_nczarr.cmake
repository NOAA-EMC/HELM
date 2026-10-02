# ####################################################################################################################################################
# run_force_nczarr.cmake
#
# Driver script for the `amio_discovery_force_nczarr_disables_tensorstore` CTest entry.
#
# Verifies (Mermaid path: A -> B[FORCE_NCZARR?] -- ON --> F):
#
# 1. AMIO_HAS_TENSORSTORE is set to OFF.
# 2. TensorStore_FOUND is FALSE (find_package was short-circuited and not allowed to reach an upstream config file).
# 3. The reason text identifies AMIO_FORCE_NCZARR=ON as the cause.
# 4. FindTensorStore.cmake emits the documented STATUS line "TensorStore discovery short-circuited (AMIO_FORCE_NCZARR=ON)".
#
# All four observables must be present.  This is AND semantics; we emit AMIO_DISCOVERY_FORCE_NCZARR=OK only when every check passes,
# AMIO_DISCOVERY_FORCE_NCZARR=FAIL otherwise (with a diagnostic listing which observable failed).
#
# Inputs (all -D flags from tests/cmake_discovery/CMakeLists.txt): AMIO_TEST_CMAKE      - path to the cmake executable to invoke AMIO_TEST_GENERATOR -
# CMake generator string for the sub-build AMIO_TEST_PROBE_SRC  - path to the probes/force_nczarr/ source AMIO_TEST_PROBE_BIN  - path to a writable
# build directory AMIO_SOURCE_DIR      - root of the AMIO repository (so the probe can pull cmake/FindTensorStore.cmake from the source tree)
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)

foreach(_var IN ITEMS AMIO_TEST_CMAKE AMIO_TEST_GENERATOR AMIO_TEST_PROBE_SRC AMIO_TEST_PROBE_BIN AMIO_SOURCE_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "${_var} must be set on the cmake -P command line")
    endif()
endforeach()

# Wipe any prior build tree so cache state from a previous run cannot leak into the test.
file(REMOVE_RECURSE "${AMIO_TEST_PROBE_BIN}")
file(MAKE_DIRECTORY "${AMIO_TEST_PROBE_BIN}")

execute_process(
    COMMAND "${AMIO_TEST_CMAKE}" -S "${AMIO_TEST_PROBE_SRC}" -B "${AMIO_TEST_PROBE_BIN}" -G "${AMIO_TEST_GENERATOR}" -DAMIO_FORCE_NCZARR=ON
            -DAMIO_SOURCE_DIR=${AMIO_SOURCE_DIR}
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
    RESULT_VARIABLE _probe_rc)

set(_combined "${_probe_stdout}\n${_probe_stderr}")
message(STATUS "Probe output:\n${_combined}")

set(_failures "")

if(NOT _probe_rc EQUAL 0)
    list(APPEND _failures "probe configure exited with code ${_probe_rc}")
endif()

# Observable 1: AMIO_HAS_TENSORSTORE=OFF
string(FIND "${_combined}" "PROBE_AMIO_HAS_TENSORSTORE=OFF" _hit_has)
if(_hit_has EQUAL -1)
    list(APPEND _failures "AMIO_HAS_TENSORSTORE was not set to OFF")
endif()

# Observable 2: TensorStore_FOUND=FALSE
string(FIND "${_combined}" "PROBE_TENSORSTORE_FOUND=FALSE" _hit_found)
if(_hit_found EQUAL -1)
    list(APPEND _failures "TensorStore_FOUND was not FALSE")
endif()

# Observable 3: Reason names AMIO_FORCE_NCZARR=ON.
string(FIND "${_combined}" "AMIO_FORCE_NCZARR=ON" _hit_reason)
if(_hit_reason EQUAL -1)
    list(APPEND _failures "TensorStore_REASON does not name AMIO_FORCE_NCZARR=ON")
endif()

# Observable 4: documented short-circuit STATUS message.
string(FIND "${_combined}" "TensorStore discovery short-circuited (AMIO_FORCE_NCZARR=ON)" _hit_status)
if(_hit_status EQUAL -1)
    list(APPEND _failures "FindTensorStore.cmake did not emit the documented short-circuit STATUS")
endif()

if(_failures)
    foreach(_msg IN LISTS _failures)
        message(STATUS "FAIL: ${_msg}")
    endforeach()
    message(FATAL_ERROR "AMIO_DISCOVERY_FORCE_NCZARR=FAIL")
endif()

message(STATUS "AMIO_DISCOVERY_FORCE_NCZARR=OK")
