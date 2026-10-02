# ####################################################################################################################################################
# run_fallback_warning.cmake
#
# Driver script for the `amio_discovery_tensorstore_absent_emits_fallback_warning` CTest entry.
#
# Verifies (Mermaid path: A -> B[FORCE_NCZARR?] -- OFF --> C[find_package(TensorStore CONFIG)] -> D[FOUND?] -- NO --> F (fallback warning)):
#
# 1. AMIO_HAS_TENSORSTORE is set to OFF.
# 2. TensorStore_FOUND is FALSE.
# 3. A "CMake Warning" appears in the captured output (R13.4 - severity is WARNING when fallback is the *consequence* of TensorStore being absent, not
#    the user's explicit choice).
# 4. The fallback name "NCZarr fallback (sharding unavailable)" appears (R13.4 - names the fallback mode).
# 5. The reason text "TensorStore not located via find_package(TensorStore CONFIG)" appears (R13.4 - identifies the reason TensorStore was not
#    selected).
#
# We empty CMAKE_PREFIX_PATH and disable CMake's package registries in the sub-configure so any TensorStore install on the host cannot accidentally
# short-circuit the test.  This makes the test deterministic on developer machines that happen to have TensorStore available system-wide.
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.20)

foreach(_var IN ITEMS AMIO_TEST_CMAKE AMIO_TEST_GENERATOR AMIO_TEST_PROBE_SRC AMIO_TEST_PROBE_BIN AMIO_SOURCE_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "${_var} must be set on the cmake -P command line")
    endif()
endforeach()

file(REMOVE_RECURSE "${AMIO_TEST_PROBE_BIN}")
file(MAKE_DIRECTORY "${AMIO_TEST_PROBE_BIN}")

execute_process(
    COMMAND
        "${AMIO_TEST_CMAKE}" -S "${AMIO_TEST_PROBE_SRC}" -B "${AMIO_TEST_PROBE_BIN}" -G "${AMIO_TEST_GENERATOR}" -DAMIO_FORCE_NCZARR=OFF
        -DAMIO_SOURCE_DIR=${AMIO_SOURCE_DIR}
        # Empty out every search path so TensorStore cannot be located on the host machine.
        -DCMAKE_PREFIX_PATH= -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=OFF
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

# Observable 3: Warning severity present.
string(FIND "${_combined}" "CMake Warning" _hit_warn)
if(_hit_warn EQUAL -1)
    list(APPEND _failures "no CMake Warning was emitted for the fallback")
endif()

# Observable 4: documented fallback name appears.
string(FIND "${_combined}" "NCZarr fallback (sharding unavailable)" _hit_name)
if(_hit_name EQUAL -1)
    list(APPEND _failures "fallback message did not name 'NCZarr fallback'")
endif()

# Observable 5: documented reason text appears.
string(FIND "${_combined}" "TensorStore not located via find_package(TensorStore CONFIG)" _hit_reason)
if(_hit_reason EQUAL -1)
    list(APPEND _failures "fallback message did not identify the reason TensorStore was not selected")
endif()

if(_failures)
    foreach(_msg IN LISTS _failures)
        message(STATUS "FAIL: ${_msg}")
    endforeach()
    message(FATAL_ERROR "AMIO_DISCOVERY_FALLBACK_WARNING=FAIL")
endif()

message(STATUS "AMIO_DISCOVERY_FALLBACK_WARNING=OK")
