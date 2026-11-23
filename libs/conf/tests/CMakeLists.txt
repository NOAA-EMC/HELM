# ─── CONF Test Infrastructure ─────────────────────────────────────────────────
# Builds unit tests (GTest) and property-based tests (RapidCheck) for the CONF
# micro-library. All tests run single-process — CONF requires neither MPI nor
# any parallel launcher.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Locate Google Test ──────────────────────────────────────────────────────
find_package(GTest REQUIRED)

# ─── Locate RapidCheck (property-based testing) ──────────────────────────────
# RapidCheck is installed globally in the HELM Docker container (see Dockerfile),
# which provides both the `rapidcheck` and `rapidcheck_gtest` targets.
find_package(rapidcheck QUIET)
if(NOT rapidcheck_FOUND)
    # Fall back to FetchContent for environments without a global install.
    include(FetchContent)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES ""
    )
    set(RC_ENABLE_GTEST ON CACHE BOOL "Build RapidCheck GTest integration" FORCE)
    FetchContent_MakeAvailable(rapidcheck)
endif()

# ─── RapidCheck GTest extras ─────────────────────────────────────────────────
# The global RapidCheck install (RC_ENABLE_GTEST=ON) already provides the
# rapidcheck_gtest target via find_package. If it is missing (e.g. an older
# install), build it from the FetchContent source tree.
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            ${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    endif()
endif()

# ─── Helper function: Add a CONF unit test (single-process) ──────────────────
function(conf_add_unit_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})

    add_executable(${TEST_NAME} ${ARG_SOURCES})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            conf
            GTest::gtest
            GTest::gtest_main
            ${ARG_LIBRARIES}
    )
    target_include_directories(${TEST_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(${TEST_NAME} PUBLIC cxx_std_20)

    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 30
        LABELS "unit"
    )
endfunction()

# ─── Helper function: Add a CONF property-based test (single-process) ────────
# Runs with RC_PARAMS enforcing at least 100 generated iterations.
function(conf_add_property_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})

    add_executable(${TEST_NAME} ${ARG_SOURCES})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            conf
            GTest::gtest
            GTest::gtest_main
            rapidcheck
            ${ARG_LIBRARIES}
    )
    # Add rapidcheck_gtest if available
    if(TARGET rapidcheck_gtest)
        target_link_libraries(${TEST_NAME} PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(${TEST_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(${TEST_NAME} PUBLIC cxx_std_20)

    # Property tests run single-process with at least 100 iterations
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 120
        LABELS "property"
        ENVIRONMENT "RC_PARAMS=max_success=100"
    )
endfunction()

# ─── Unit Test Targets (single-process) ──────────────────────────────────────
# Only build tests whose source files exist (incremental development)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_error_model.cpp")
    conf_add_unit_test(test_error_model
        SOURCES test_error_model.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_config_load.cpp")
    conf_add_unit_test(test_config_load
        SOURCES test_config_load.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_malformed_yaml.cpp")
    conf_add_unit_test(test_malformed_yaml
        SOURCES test_malformed_yaml.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_missing_keys.cpp")
    conf_add_unit_test(test_missing_keys
        SOURCES test_missing_keys.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_type_casting.cpp")
    conf_add_unit_test(test_type_casting
        SOURCES test_type_casting.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_dotted_path.cpp")
    conf_add_unit_test(test_dotted_path
        SOURCES test_dotted_path.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_value.cpp")
    conf_add_unit_test(test_value
        SOURCES test_value.cpp
    )
endif()

# ─── Handle Registry Unit Test (needs private src/fortran/ include path) ─────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_handle_registry.cpp")
    conf_add_unit_test(test_handle_registry
        SOURCES test_handle_registry.cpp
    )
    target_include_directories(test_handle_registry
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── C Bridge Unit Test (needs src/fortran/ include path) ─────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_c_bridge.cpp")
    conf_add_unit_test(test_c_bridge
        SOURCES test_c_bridge.cpp
    )
    target_include_directories(test_c_bridge
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── Property-Based Test Targets (single-process, 100+ iterations) ───────────

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_round_trip.cpp")
    conf_add_property_test(prop_round_trip
        SOURCES prop_round_trip.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_type_safety.cpp")
    conf_add_property_test(prop_type_safety
        SOURCES prop_type_safety.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_path_resolution.cpp")
    conf_add_property_test(prop_path_resolution
        SOURCES prop_path_resolution.cpp
    )
endif()

# ─── Handle Registry Property Test (needs private src/fortran/ include path) ─
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_handle_registry.cpp")
    conf_add_property_test(prop_handle_registry
        SOURCES prop_handle_registry.cpp
    )
    target_include_directories(prop_handle_registry
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── C Bridge Property Tests (need src/fortran/ include path) ─────────────────
# These tests exercise the extern "C" interop layer (src/fortran/conf_c_interop.cpp),
# which is compiled into `conf` when CONF_BUILD_FORTRAN=ON. The src/fortran include
# directory is added so the tests can reach handle_registry.hpp.

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_no_leak_c_bridge.cpp")
    conf_add_property_test(prop_no_leak_c_bridge
        SOURCES prop_no_leak_c_bridge.cpp
    )
    target_include_directories(prop_no_leak_c_bridge
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_error_lockstep.cpp")
    conf_add_property_test(prop_error_lockstep
        SOURCES prop_error_lockstep.cpp
    )
    target_include_directories(prop_error_lockstep
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# Private resolver contract: clone only the selected snapshot without mutating YAML.
conf_add_unit_test(test_resolver_views SOURCES test_resolver_views.cpp LIBRARIES yaml-cpp)
