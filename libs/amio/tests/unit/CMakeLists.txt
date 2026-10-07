# ####################################################################################################################################################
# tests/unit/CMakeLists.txt
#
# Lightweight C++20 unit tests for the AMIO_C_API entry points.
#
# These tests deliberately avoid pulling in Catch2 or Rapidcheck -- task 13.1 is responsible for adding those dependencies for the property-based test
# suite.  At this stage of the build a plain `<cassert>`-based harness keeps the test target small and means a downstream consumer running `ctest`
# against AMIO needs nothing on their machine that AMIO itself does not already require.
#
# Each test_* executable links against AMIO::amio_core (so the real, installed `extern "C"` symbols are exercised, not a private copy) and against
# AMIO::public_headers for the C99 declarations.
#
# Per-test ownership map:
#
# * test_amio_strerror -- task 3.3, validates R12.5 - R12.8. Pins the strerror table contract that property test P17 (task 13.18) verifies
#   universally.
#
# * test_handle_table  -- task 3.2, validates R10.5 - R10.7. Compiles `handle_table.cpp` directly into the test binary because the header lives under
#   `src/c_boundary/` and is private to the AMIO_Core build.  Pins the generation-counter / reuse-after-free invariant that property test P4 (task
#   13.5) verifies universally.
#
# * test_amio_api_handle_validation -- task 3.2, validates the public C-Boundary handle / argument / NULL-output cordon (R10.6, R10.7, R12.2) through
#   the installed `extern "C"` AMIO_C_API symbols.  The implementation bodies behind each entry point are still stubs at this stage, so successful
#   cordon traversal lands in AMIO_ERR_BACKEND_FAILURE -- the cordon itself is what this test pins.
# ####################################################################################################################################################

add_executable(test_amio_strerror test_amio_strerror.cpp)

target_link_libraries(test_amio_strerror PRIVATE AMIO::public_headers AMIO::amio_core)

# The harness uses std::thread; on most Unix toolchains this requires linking pthreads.  CMake's Threads module exposes that portably.
find_package(Threads QUIET)
if(Threads_FOUND)
    target_link_libraries(test_amio_strerror PRIVATE Threads::Threads)
endif()

add_test(NAME unit.amio_strerror COMMAND test_amio_strerror)
set_tests_properties(unit.amio_strerror PROPERTIES LABELS "unit;c_boundary;errors" TIMEOUT 30)

# ####################################################################################################################################################
# test_handle_table
#
# Compiles `handle_table.cpp` directly into the test binary so the private header `src/c_boundary/handle_table.hpp` is reachable without exposing it
# through the public include tree.
# ####################################################################################################################################################
add_executable(test_handle_table test_handle_table.cpp ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp)

target_include_directories(test_handle_table PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_handle_table PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_handle_table PRIVATE Threads::Threads)
endif()

add_test(NAME unit.handle_table COMMAND test_handle_table)
set_tests_properties(unit.handle_table PROPERTIES LABELS "unit;c_boundary;handle_table" TIMEOUT 30)

# ####################################################################################################################################################
# test_amio_api_handle_validation
#
# Exercises the C-Boundary cordon through the installed AMIO_C_API symbols.  Links AMIO::amio_core to pick up the real `extern "C"` entry points; the
# underlying bodies are still stubs (task 3.2) but the cordon validation paths (NULL handle, garbage handle, NULL out-pointer, exception translation)
# are fully wired.
# ####################################################################################################################################################
add_executable(test_amio_api_handle_validation test_amio_api_handle_validation.cpp)

target_link_libraries(test_amio_api_handle_validation PRIVATE AMIO::public_headers AMIO::amio_core)

add_test(NAME unit.amio_api_handle_validation COMMAND test_amio_api_handle_validation)
set_tests_properties(unit.amio_api_handle_validation PROPERTIES LABELS "unit;c_boundary;handles" TIMEOUT 30)

# ####################################################################################################################################################
# test_thread_pinning
#
# Compiles `thread_pinning.cpp` directly into the test binary so the private header `src/workers/thread_pinning.hpp` is reachable without exposing it
# through the public include tree.
#
# Validates: R3.2, R3.3
# ####################################################################################################################################################
add_executable(test_thread_pinning test_thread_pinning.cpp ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp)

target_include_directories(test_thread_pinning PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_thread_pinning PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_thread_pinning PRIVATE Threads::Threads)
endif()

add_test(NAME unit.thread_pinning COMMAND test_thread_pinning)
set_tests_properties(unit.thread_pinning PROPERTIES LABELS "unit;workers;pinning" TIMEOUT 30)

# ####################################################################################################################################################
# test_comm_split
#
# Compiles `comm_split.cpp` directly into the test binary so the private header `src/workers/comm_split.hpp` is reachable without exposing it through
# the public include tree.
#
# Validates: R3.5, R3.6
# ####################################################################################################################################################
add_executable(test_comm_split test_comm_split.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)

target_include_directories(test_comm_split PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_comm_split PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_comm_split PRIVATE Threads::Threads)
endif()

add_test(NAME unit.comm_split COMMAND test_comm_split)
set_tests_properties(unit.comm_split PROPERTIES LABELS "unit;workers;comm_split" TIMEOUT 30)

# ####################################################################################################################################################
# test_staging_pool
#
# Compiles `staging_pool.cpp` directly into the test binary so the private header `src/staging/staging_pool.hpp` is reachable without exposing it
# through the public include tree.
#
# Validates: R1.3, R1.5, R2.2, R2.6, R2.9, R3.10, R5.6, R5.9
# ####################################################################################################################################################
add_executable(test_staging_pool test_staging_pool.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

target_include_directories(test_staging_pool PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_staging_pool PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_staging_pool PRIVATE Threads::Threads)
endif()

add_test(NAME unit.staging_pool COMMAND test_staging_pool)
set_tests_properties(unit.staging_pool PROPERTIES LABELS "unit;staging;pool" TIMEOUT 60)

# ####################################################################################################################################################
# test_worker_pool
#
# Compiles `worker_pool.cpp` directly into the test binary so the private header `src/workers/worker_pool.hpp` is reachable without exposing it
# through the public include tree.
#
# Also compiles `thread_pinning.cpp` and `comm_split.cpp` because worker_pool.hpp includes their headers for the WorkerPoolConfig integration (R3.2,
# R3.3, R3.5, R3.6).
#
# Validates: R3.1, R3.2, R3.3, R3.4, R3.5, R3.6, R3.7, R3.8, R6.1, R6.2, R6.3, R6.8, R6.9
# ####################################################################################################################################################
add_executable(
    test_worker_pool
    test_worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp)

target_include_directories(test_worker_pool PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_worker_pool PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_worker_pool PRIVATE Threads::Threads)
endif()

add_test(NAME unit.worker_pool COMMAND test_worker_pool)
set_tests_properties(unit.worker_pool PROPERTIES LABELS "unit;workers;pool" TIMEOUT 120)

# ####################################################################################################################################################
# test_exception_bridge
#
# Compiles `exception_bridge.cpp`, `worker_pool.cpp`, `thread_pinning.cpp`, `comm_split.cpp`, and `mpi_threading.cpp` directly into the test binary so
# the private headers under `src/workers/` are reachable.
#
# Validates: R12.1, R12.2, R12.3, R12.4, R12.9, R12.10
# ####################################################################################################################################################
add_executable(
    test_exception_bridge
    test_exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp)

target_include_directories(test_exception_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_exception_bridge PRIVATE AMIO::public_headers)
if(CONF_FOUND)
    target_link_libraries(test_exception_bridge PRIVATE HELM::CONF)
else()
    # Fallback: CONF include path from workspace until find_package(CONF) is integrated (task 10).
    target_include_directories(test_exception_bridge PRIVATE ${CMAKE_SOURCE_DIR}/../conf/include)
endif()
if(Threads_FOUND)
    target_link_libraries(test_exception_bridge PRIVATE Threads::Threads)
endif()

add_test(NAME unit.exception_bridge COMMAND test_exception_bridge)
set_tests_properties(unit.exception_bridge PROPERTIES LABELS "unit;workers;exception_bridge" TIMEOUT 60)

# ####################################################################################################################################################
# test_logs_integration
#
# Tests LOGS integration in AMIO: Logger initialization lifecycle, pre-initialization stderr fallback, and post-initialization routing through
# logs::Logger.  Compiles `exception_bridge.cpp` and its dependencies directly into the test binary so the private headers under `src/` are reachable.
#
# Validates: R6.7, R6.8, R12.1
# ####################################################################################################################################################
add_executable(
    test_logs_integration
    test_logs_integration.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp)

target_include_directories(test_logs_integration PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_logs_integration PRIVATE AMIO::public_headers)
if(CONF_FOUND)
    target_link_libraries(test_logs_integration PRIVATE HELM::CONF)
else()
    # Fallback: CONF include path from workspace until find_package(CONF) is integrated (task 10).
    target_include_directories(test_logs_integration PRIVATE ${CMAKE_SOURCE_DIR}/../conf/include)
endif()
if(LOGS_FOUND)
    target_link_libraries(test_logs_integration PRIVATE HELM::LOGS)
else()
    # Fallback: LOGS include path from workspace until find_package(LOGS) is integrated (task 10).
    target_include_directories(test_logs_integration PRIVATE ${CMAKE_SOURCE_DIR}/../logs/include)
endif()
if(Threads_FOUND)
    target_link_libraries(test_logs_integration PRIVATE Threads::Threads)
endif()

add_test(NAME unit.logs_integration COMMAND test_logs_integration)
set_tests_properties(unit.logs_integration PROPERTIES LABELS "unit;logs;integration" TIMEOUT 60)

# ####################################################################################################################################################
# test_backend_factory
#
# Compiles `backend_factory.cpp` directly into the test binary so the private header `src/factory/backend_factory.hpp` is reachable without exposing
# it through the public include tree.
#
# Validates: R4.1, R4.2, R4.3, R4.4, R4.5, R4.6, R4.7, R4.8
# ####################################################################################################################################################
add_executable(test_backend_factory test_backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp)

target_include_directories(test_backend_factory PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_backend_factory PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_backend_factory PRIVATE Threads::Threads)
endif()

add_test(NAME unit.backend_factory COMMAND test_backend_factory)
set_tests_properties(unit.backend_factory PROPERTIES LABELS "unit;factory;backend" TIMEOUT 30)

# ####################################################################################################################################################
# test_config_loader
#
# Compiles `config_loader.cpp` directly into the test binary so the private header `src/config/config_loader.hpp` is reachable without exposing it
# through the public include tree.
#
# Validates: R1.2, R1.3, R1.5, R11.3, R11.4, R11.5, R11.6, R11.7
# ####################################################################################################################################################
add_executable(test_config_loader test_config_loader.cpp ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp)

target_include_directories(test_config_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_config_loader PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_config_loader PRIVATE Threads::Threads)
endif()

add_test(NAME unit.config_loader COMMAND test_config_loader)
set_tests_properties(unit.config_loader PROPERTIES LABELS "unit;config;loader" TIMEOUT 30)

# ####################################################################################################################################################
# test_dataset_lifecycle
#
# Tests the dataset open/close lifecycle through the Backend_Factory (task 6.3).  Links against AMIO::amio_core for the real extern "C" entry points.
# Compiles backend_factory.cpp directly to register a mock driver for testing.
#
# Validates: R4.1, R4.6, R6.4, R6.5
# ####################################################################################################################################################
add_executable(test_dataset_lifecycle test_dataset_lifecycle.cpp)

target_link_libraries(test_dataset_lifecycle PRIVATE AMIO::public_headers AMIO::amio_core)
if(Threads_FOUND)
    target_link_libraries(test_dataset_lifecycle PRIVATE Threads::Threads)
endif()
# The NetCDF-4 backend now actually opens via nc_create_par/nc_open_par, which require the host to have initialized MPI.  The test main does so; link
# MPI for the symbols.
if(MPI_C_FOUND)
    target_link_libraries(test_dataset_lifecycle PRIVATE MPI::MPI_C)
endif()

add_test(NAME unit.dataset_lifecycle COMMAND test_dataset_lifecycle)
set_tests_properties(unit.dataset_lifecycle PROPERTIES LABELS "unit;factory;dataset;lifecycle" TIMEOUT 30)

# ####################################################################################################################################################
# test_write_snapshot
#
# Tests the synchronous snapshot write path (task 9.1).  Exercises input validation, snapshot semantics, and io_handle creation through the installed
# extern "C" entry points.
#
# Validates: R2.1, R2.2, R2.3, R2.4, R2.5, R2.7, R2.8, R2.10, R6.1
# ####################################################################################################################################################
add_executable(
    test_write_snapshot
    test_write_snapshot.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_api.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_write_snapshot PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_write_snapshot PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_write_snapshot PRIVATE Threads::Threads)
endif()

add_test(NAME unit.write_snapshot COMMAND test_write_snapshot)
set_tests_properties(unit.write_snapshot PROPERTIES LABELS "unit;write;snapshot;staging" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_prefetch
#
# Tests the read prefetch path (task 9.2).  Exercises the PrefetchQueue, amio_read wiring, and view lifecycle through direct unit tests of the
# PrefetchQueue class.
#
# Validates: R5.1, R5.2, R5.3, R5.4, R5.5, R5.7, R5.8
# ####################################################################################################################################################
add_executable(
    test_read_prefetch
    test_read_prefetch.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp)

target_include_directories(test_read_prefetch PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_prefetch PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_prefetch PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_prefetch COMMAND test_read_prefetch)
set_tests_properties(unit.read_prefetch PROPERTIES LABELS "unit;prefetch;read" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_bbox_validation
#
# Tests the selective-read bounding-box validation helper (task 9, design §9) through the real amio::detail::read coordinator.  Compiles the
# c_boundary read path and its dependencies directly into the test binary so the file-local validate_bbox helper is exercised via read() against a
# mock driver that reports a known variable shape through describe_variable.
#
# Validates: R12.1, R12.2, R12.3, R12.4
# ####################################################################################################################################################
add_executable(
    test_read_bbox_validation
    test_read_bbox_validation.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_read_bbox_validation PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_bbox_validation PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_bbox_validation PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_bbox_validation COMMAND test_read_bbox_validation)
set_tests_properties(unit.read_bbox_validation PROPERTIES LABELS "unit;read;bbox;validation" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_capacity_guard
#
# Tests the read-path buffer-capacity guard and prefetch failure surfacing (task 10, design §5 / Error Handling, Property 8) through the real
# amio::detail::read coordinator + PrefetchQueue.  Compiles the c_boundary read path and its dependencies directly into the test binary so the central
# capacity backstop in PrefetchQueue::sync_fetch is exercised via read() against mock drivers that (a) report a payload past dst.capacity_bytes, (b)
# throw a driver-side capacity guard, and (c) read within capacity.  Also pins that a recorded fetch failure surfaces on the next read of that
# timestep (failed_ map behavior).
#
# Validates: R4.4, R6.2, R6.3
# ####################################################################################################################################################
add_executable(
    test_read_capacity_guard
    test_read_capacity_guard.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_read_capacity_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_capacity_guard PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_capacity_guard PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_capacity_guard COMMAND test_read_capacity_guard)
set_tests_properties(unit.read_capacity_guard PROPERTIES LABELS "unit;read;capacity;guard" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_close_guard
#
# Tests the outstanding-view guard at close and read-path prefetch cancellation (task 11, design §8, Property 7) through the real amio::detail::read /
# release_view / close_dataset coordinator + PrefetchQueue.  Compiles the c_boundary read path and its dependencies directly into the test binary so
# the close guard and per-variable cancel_pending() are exercised via close_dataset() against a mock driver: close with an outstanding view returns
# AMIO_ERR_VIEWS_OUTSTANDING and performs no teardown; after releasing the view close succeeds; and closing a reader with completed-but-unread
# prefetched buffers releases them back to the Staging_Pool.
#
# Validates: R8.1, R8.2, R8.3
# ####################################################################################################################################################
add_executable(
    test_read_close_guard
    test_read_close_guard.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_read_close_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_close_guard PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_close_guard PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_close_guard COMMAND test_read_close_guard)
set_tests_properties(unit.read_close_guard PROPERTIES LABELS "unit;read;close;views" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_release_view
#
# Tests Memory_View release semantics (task 12, design §8, Req 7 / Property 7) through the real amio::detail::read coordinator AND the installed
# extern "C" amio_release_view entry point.  Compiles the c_boundary read path, the public C-boundary translation unit (amio_api.cpp), and their
# dependencies directly into the test binary so the View token minted by read() is validated and released through the SAME process_handle_table().
# This exercises the real stale-token rejection: a second amio_release_view of the same handle presents a token whose slot generation no longer
# matches, so the kind_dispatch lookup returns AMIO_ERR_INVALID_HANDLE before the body runs (double-release rejection, Req 7.5).  Single-release
# success + outstanding-view-at-close rejection are covered by test_read_close_guard.cpp (task 11) and are not duplicated here.
#
# Validates: R7.1, R7.2, R7.3, R7.4, R7.5
# ####################################################################################################################################################
add_executable(
    test_read_release_view
    test_read_release_view.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_api.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_read_release_view PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_release_view PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_release_view PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_release_view COMMAND test_read_release_view)
set_tests_properties(unit.read_release_view PROPERTIES LABELS "unit;read;view;release" TIMEOUT 60)

# ####################################################################################################################################################
# test_read_e2e_smoke
#
# End-to-end public-C-ABI read smoke test (task 20, design "Testing Strategy").  Drives the COMPLETE lifecycle through the installed extern "C" entry
# points -- amio_init -> amio_open_dataset(READ) -> amio_read -> amio_release_view -> amio_close_dataset -> amio_finalize -- and asserts the read is
# served from the runtime Staging_Pool with AMIO_OK and explicitly NOT AMIO_ERR_STAGING_BACKPRESSURE (the original core bug: init() never built the
# Staging_Pool, so every read failed with backpressure).  A tiny in-process mock Backend_Driver is registered under a private factory key so the test
# is hermetic (no parallel HDF5 / TensorStore / g2c needed); the real-driver byte fidelity is covered by unit.read_netcdf4 + rp1..rp8. Compiles the
# same c_boundary + factory + staging + workers + prefetch source set as test_read_release_view so the public symbols dispatch into the real init /
# open_dataset / read coordinator.
#
# Validates: Req 1.3, 2.1, 9.1
# ####################################################################################################################################################
add_executable(
    test_read_e2e_smoke
    test_read_e2e_smoke.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_api.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_core.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/handle_table.cpp
    ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
    ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/prefetch/prefetch_queue.cpp)

target_include_directories(test_read_e2e_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_read_e2e_smoke PRIVATE AMIO::public_headers)
if(Threads_FOUND)
    target_link_libraries(test_read_e2e_smoke PRIVATE Threads::Threads)
endif()

add_test(NAME unit.read_e2e_smoke COMMAND test_read_e2e_smoke)
set_tests_properties(unit.read_e2e_smoke PROPERTIES LABELS "unit;read;e2e;smoke" TIMEOUT 60)

# ####################################################################################################################################################
# test_lifecycle_netcdf4
#
# Full lifecycle integration test for the NetCDF-4 backend.  Exercises the complete init → open → write → flush → close → finalize path with the REAL
# netcdf4 driver writing to a temp file.  No mocks.  Accepts AMIO_ERR_BACKEND_FAILURE as valid if the underlying library isn't fully functional (e.g.,
# netCDF without parallel HDF5).  The key assertion: the lifecycle completes without crashing.
#
# Validates: end-to-end lifecycle correctness for netcdf4 backend.
# ####################################################################################################################################################
add_executable(test_lifecycle_netcdf4 test_lifecycle_netcdf4.cpp)

target_link_libraries(test_lifecycle_netcdf4 PRIVATE AMIO::public_headers AMIO::amio_core)
if(Threads_FOUND)
    target_link_libraries(test_lifecycle_netcdf4 PRIVATE Threads::Threads)
endif()
# The NetCDF-4 backend now actually opens via nc_create_par, which requires the host to have initialized MPI.  The test main does so; link MPI for the
# symbols.
if(MPI_C_FOUND)
    target_link_libraries(test_lifecycle_netcdf4 PRIVATE MPI::MPI_C)
endif()

add_test(NAME unit.lifecycle_netcdf4 COMMAND test_lifecycle_netcdf4)
set_tests_properties(unit.lifecycle_netcdf4 PROPERTIES LABELS "unit;lifecycle;netcdf4" TIMEOUT 60)

# ####################################################################################################################################################
# test_lifecycle_zarr
#
# Full lifecycle integration test for the Zarr v3 (NCZarr fallback) backend.  Exercises the complete init → open → write → flush → close → finalize
# path with the REAL zarr3 driver writing to a temp directory.  No mocks.  Accepts AMIO_ERR_BACKEND_FAILURE as valid if the underlying library isn't
# fully functional.  The key assertion: the lifecycle completes without crashing.
#
# Validates: end-to-end lifecycle correctness for zarr3 backend.
# ####################################################################################################################################################
add_executable(test_lifecycle_zarr test_lifecycle_zarr.cpp)

target_link_libraries(test_lifecycle_zarr PRIVATE AMIO::public_headers AMIO::amio_core)
if(Threads_FOUND)
    target_link_libraries(test_lifecycle_zarr PRIVATE Threads::Threads)
endif()

add_test(NAME unit.lifecycle_zarr COMMAND test_lifecycle_zarr)
set_tests_properties(unit.lifecycle_zarr PROPERTIES LABELS "unit;lifecycle;zarr3" TIMEOUT 60)

# ####################################################################################################################################################
# test_lifecycle_grib2
#
# Full lifecycle integration test for the GRIB2 backend.  Exercises the complete init → open → write → flush → close → finalize path with the REAL
# grib2 driver writing to a temp file.  No mocks.  Includes WMO code table config and DRT: libaec.  Accepts AMIO_ERR_BACKEND_FAILURE as valid if the
# underlying g2c library isn't fully functional or WMO tables aren't loaded.  The key assertion: the lifecycle completes without crashing.
#
# Validates: end-to-end lifecycle correctness for grib2 backend.
# ####################################################################################################################################################
add_executable(test_lifecycle_grib2 test_lifecycle_grib2.cpp)

target_link_libraries(test_lifecycle_grib2 PRIVATE AMIO::public_headers AMIO::amio_core)
if(Threads_FOUND)
    target_link_libraries(test_lifecycle_grib2 PRIVATE Threads::Threads)
endif()

add_test(NAME unit.lifecycle_grib2 COMMAND test_lifecycle_grib2)
set_tests_properties(unit.lifecycle_grib2 PROPERTIES LABELS "unit;lifecycle;grib2" TIMEOUT 60)

# ####################################################################################################################################################
# test_cf_attributes
#
# Integration test for CF/UGRID convention compliance + per-variable attributes.  Drives the real Zarr_Driver (NCZarr fallback, serial netCDF-c) end-
# to-end and reads the produced store back with netCDF-c to assert the written attributes.  Only built when netCDF is available and the build is in
# NCZarr fallback mode (the serial nc_create path needs no MPI).
# ####################################################################################################################################################
if(netCDF_FOUND AND NOT AMIO_HAS_TENSORSTORE)
    add_executable(
        test_cf_attributes
        test_cf_attributes.cpp ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_nczarr_fallback.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp)

    target_include_directories(test_cf_attributes PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_cf_attributes PRIVATE AMIO_NCZARR_FALLBACK)
    target_link_libraries(test_cf_attributes PRIVATE AMIO::public_headers netCDF::netcdf)
    if(Threads_FOUND)
        target_link_libraries(test_cf_attributes PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.cf_attributes COMMAND test_cf_attributes)
    set_tests_properties(unit.cf_attributes PROPERTIES LABELS "unit;conventions;attributes;cf;ugrid" TIMEOUT 60)
endif()

# ####################################################################################################################################################
# test_read_netcdf4
#
# Integration test for the NetCDF-4 READ path (task 13.1).  Drives the real NetCDF_Driver end-to-end: write a known 2-D float field, then re-open for
# reading and assert describe_variable metadata plus byte-for-byte fidelity on a full read, a unit-stride bbox read (nc_get_vara), and a strided bbox
# read (nc_get_vars).  Compiles the netcdf driver and its deps directly into the test binary (mirroring how test_cf_attributes compiles the zarr
# driver) so the real nc_inq_* / nc_get_vara / nc_get_vars paths are exercised.
#
# Only built when netCDF + MPI are available.  The NetCDF-4 driver issues nc_create_par / nc_open_par which require the host to have initialized MPI;
# the test main does so, mirroring test_lifecycle_netcdf4.  The driver/test link the SAME parallel netCDF runtime (AMIO_NETCDF_PAR_LIB) and parallel
# header path (AMIO_NETCDF_MPI_INCLUDE_DIR) that driver_netcdf uses.
#
# Validates: Req 9.1, 9.2, 9.3, 9.4 (NetCDF-4 read end-to-end).
# ####################################################################################################################################################
if(netCDF_FOUND AND MPI_C_FOUND)
    add_executable(
        test_read_netcdf4
        test_read_netcdf4.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp
        ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)

    target_include_directories(test_read_netcdf4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_read_netcdf4 PRIVATE AMIO_HAS_NETCDF=1)
    target_link_libraries(test_read_netcdf4 PRIVATE AMIO::public_headers)

    # Link the same parallel netCDF runtime as driver_netcdf so nc_*_par work at runtime; expose the parallel-only netcdf_par.h via its include dir.
    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_amio_read_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_read_nc_inc)
            target_include_directories(test_read_netcdf4 PRIVATE ${_amio_read_nc_inc})
        endif()
        target_link_libraries(test_read_netcdf4 PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_read_netcdf4 PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_NETCDF_MPI_INCLUDE_DIR)
        target_include_directories(test_read_netcdf4 PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
    endif()
    target_link_libraries(test_read_netcdf4 PRIVATE MPI::MPI_C)
    if(MPI_CXX_FOUND)
        # netcdf_driver.cpp includes <mpi.h> without OMPI_SKIP_MPICXX, so it references the deprecated C++ MPI bindings (libmpi_cxx); link them
        # exactly as driver_netcdf does.
        target_link_libraries(test_read_netcdf4 PRIVATE MPI::MPI_CXX)
    endif()
    if(Threads_FOUND)
        target_link_libraries(test_read_netcdf4 PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.read_netcdf4 COMMAND test_read_netcdf4)
    set_tests_properties(unit.read_netcdf4 PROPERTIES LABELS "unit;read;netcdf4;integration" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_read_zarr
#
# Integration test for the Zarr v3 READ path + round trip (task 15.2).  Drives the real Zarr_Driver in TensorStore mode end-to-end: write a known 2-D
# array, then re-open for reading and assert describe_variable metadata plus byte-for-byte fidelity on a full read, a unit-stride bbox read, a strided
# bbox read (TranslateSizedInterval domain subsetting), and a non-F32 (int32, float64) write->read round trip.  Compiles the zarr driver and its deps
# directly into the test binary (mirroring how test_cf_attributes compiles the zarr driver) so the real TensorStore read/write/describe paths and the
# dtype dispatch are exercised.
#
# Only built when TensorStore is available (AMIO_HAS_TENSORSTORE).  In the NCZarr-fallback build (AMIO_HAS_TENSORSTORE=OFF) this target is not added,
# so the guarded condition mirrors the TensorStore-mode build of driver_zarr in the root CMakeLists.txt (sources + AMIO_HAS_TENSORSTORE=1 +
# TensorStore::tensorstore).
#
# Validates: Req 10.1, 10.2, 10.3, 11.2, 11.3 (Zarr read end-to-end + dtype round trip).
# ####################################################################################################################################################
if(AMIO_HAS_TENSORSTORE)
    add_executable(
        test_read_zarr
        test_read_zarr.cpp ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_nczarr_fallback.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp)

    target_include_directories(test_read_zarr PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_read_zarr PRIVATE AMIO_HAS_TENSORSTORE=1)
    target_link_libraries(test_read_zarr PRIVATE AMIO::public_headers TensorStore::tensorstore)
    if(Threads_FOUND)
        target_link_libraries(test_read_zarr PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.read_zarr COMMAND test_read_zarr)
    set_tests_properties(unit.read_zarr PROPERTIES LABELS "unit;read;zarr3;integration" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_read_grib2
#
# Integration test for the GRIB2 READ/decode path (tasks 17, 17.1).  Drives the real GRIB2_Driver end-to-end: encode known F32 grids with a lossless
# DRT (libaec 5.42 / JPEG2000 5.40), then re-open for reading and assert describe_variable metadata plus decode-to-F32 byte equality on a full read, a
# lossless write->read round trip, a unit-stride bbox read, and a strided bbox read (decode-then-subset).  Compiles the grib2 driver and its deps
# directly into the test binary (mirroring how test_read_netcdf4 compiles the netcdf driver) so the real g2c g2_getfld decode path is exercised.
#
# Only built when g2c is available (g2c_FOUND -> AMIO_HAS_G2C).  Unlike the NetCDF-4 driver, the GRIB2 driver issues no MPI calls, so no MPI runtime
# is linked or initialized here.
#
# Validates: Req 13.2, 13.3, 13.4, 13.5 (GRIB2 decode + bbox + lossless round trip).
# ####################################################################################################################################################
if(g2c_FOUND)
    add_executable(test_read_grib2 test_read_grib2.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                   ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

    target_include_directories(test_read_grib2 PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_read_grib2 PRIVATE AMIO_HAS_G2C=1)
    target_link_libraries(test_read_grib2 PRIVATE AMIO::public_headers g2c::g2c)
    if(Threads_FOUND)
        target_link_libraries(test_read_grib2 PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.read_grib2 COMMAND test_read_grib2)
    set_tests_properties(unit.read_grib2 PROPERTIES LABELS "unit;read;grib2;integration" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_read_mixed_format
#
# Integration test for the mixed-format read -> (reformat) -> write -> read path (task 18, design §10, Req 14).  A host opens one source dataset for
# reading and one target dataset of a DIFFERENT format for writing on a single shared Staging_Pool runtime, reads a variable from the source, writes
# those SAME staging bytes to the target (no external post-processing), reads the target back, and asserts byte equality.
#
# Always-on core: NetCDF-4 -> GRIB2 -> read for an F32 field (the robustly verifiable mixed-format path in every build of this repo).  The NetCDF-4
# driver issues nc_create_par / nc_open_par which require the host to have initialized MPI; the test main does so, mirroring test_read_netcdf4.  The
# GRIB2 driver issues no MPI calls.  The driver/test link the SAME parallel netCDF runtime (AMIO_NETCDF_PAR_LIB) + parallel header path
# (AMIO_NETCDF_MPI_INCLUDE_DIR) as driver_netcdf, plus g2c::g2c for the GRIB2 leg.
#
# Optional leg: NetCDF-4 -> Zarr v3 -> read, guarded behind AMIO_HAS_TENSORSTORE so it compiles/runs ONLY in the TensorStore build (mirroring
# test_read_zarr.cpp, task 15.2).  In the NCZarr-fallback build (AMIO_HAS_TENSORSTORE=OFF) the Zarr leg is compiled out entirely and no Zarr sources
# are added to the target.
#
# Built when netCDF + g2c + MPI are all available (the always-on core needs all three).
#
# Validates: Req 14.1, 14.2, 14.3 (mixed-format read -> write translation).
# ####################################################################################################################################################
if(netCDF_FOUND AND g2c_FOUND AND MPI_C_FOUND)
    set(_amio_mixed_sources
        test_read_mixed_format.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp
        ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)

    # The Zarr (TensorStore) leg only compiles in the TensorStore build.
    if(AMIO_HAS_TENSORSTORE)
        list(APPEND _amio_mixed_sources ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_driver.cpp
             ${CMAKE_SOURCE_DIR}/src/drivers/zarr/zarr_nczarr_fallback.cpp)
    endif()

    add_executable(test_read_mixed_format ${_amio_mixed_sources})

    target_include_directories(test_read_mixed_format PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_read_mixed_format PRIVATE AMIO_HAS_NETCDF=1 AMIO_HAS_G2C=1)
    if(AMIO_HAS_TENSORSTORE)
        target_compile_definitions(test_read_mixed_format PRIVATE AMIO_HAS_TENSORSTORE=1)
    endif()

    target_link_libraries(test_read_mixed_format PRIVATE AMIO::public_headers g2c::g2c)

    # Link the same parallel netCDF runtime as driver_netcdf so nc_*_par work at runtime; expose the parallel-only netcdf_par.h via its include dir
    # (mirroring test_read_netcdf4).
    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_amio_mixed_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_mixed_nc_inc)
            target_include_directories(test_read_mixed_format PRIVATE ${_amio_mixed_nc_inc})
        endif()
        target_link_libraries(test_read_mixed_format PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_read_mixed_format PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_NETCDF_MPI_INCLUDE_DIR)
        target_include_directories(test_read_mixed_format PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
    endif()
    target_link_libraries(test_read_mixed_format PRIVATE MPI::MPI_C)
    if(MPI_CXX_FOUND)
        # netcdf_driver.cpp includes <mpi.h> without OMPI_SKIP_MPICXX, so it references the deprecated C++ MPI bindings (libmpi_cxx); link them
        # exactly as driver_netcdf / test_read_netcdf4 do.
        target_link_libraries(test_read_mixed_format PRIVATE MPI::MPI_CXX)
    endif()
    if(AMIO_HAS_TENSORSTORE)
        target_link_libraries(test_read_mixed_format PRIVATE TensorStore::tensorstore)
    endif()
    if(Threads_FOUND)
        target_link_libraries(test_read_mixed_format PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.read_mixed_format COMMAND test_read_mixed_format)
    set_tests_properties(unit.read_mixed_format PROPERTIES LABELS "unit;read;mixed;integration" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_e2e_netcdf_roundtrip
#
# End-to-end public-C-API round-trip test for the NetCDF-4 backend.  Exercises the FULL lifecycle: amio_init → amio_open_dataset(WRITE) → amio_write →
# amio_flush → amio_close_dataset → amio_open_dataset(READ) → amio_read → memcmp(written, read) → amio_release_view → amio_close_dataset →
# amio_finalize.  Writes a known 10x10 F32 array via the real netcdf4 driver, then reads it back and asserts byte-for-byte equality.
#
# Links against AMIO::amio_core (the real libamio.so with all drivers) so the write-path unstub (driver->write) and the read-path are exercised end to
# end with no mocks.  Uses internal detail headers (handle_table, amio_core, staging_pool) to extract the view data pointer for byte comparison since
# the public C API does not expose amio_view_data().
#
# Only built when netCDF + MPI are available (the netcdf4 backend requires parallel HDF5 + MPI).
#
# Validates: write-path unstub, read-path fidelity, full C API lifecycle.
# ####################################################################################################################################################
if(netCDF_FOUND AND MPI_C_FOUND)
    add_executable(test_e2e_netcdf_roundtrip test_e2e_netcdf_roundtrip.cpp)

    target_include_directories(test_e2e_netcdf_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(test_e2e_netcdf_roundtrip PRIVATE AMIO::public_headers AMIO::amio_core MPI::MPI_C)

    # Link the netCDF C library for the direct nc_open/nc_get_var verification.
    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_amio_e2e_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_e2e_nc_inc)
            target_include_directories(test_e2e_netcdf_roundtrip PRIVATE ${_amio_e2e_nc_inc})
        endif()
        target_link_libraries(test_e2e_netcdf_roundtrip PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_e2e_netcdf_roundtrip PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_NETCDF_MPI_INCLUDE_DIR)
        target_include_directories(test_e2e_netcdf_roundtrip PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
    endif()
    if(Threads_FOUND)
        target_link_libraries(test_e2e_netcdf_roundtrip PRIVATE Threads::Threads)
    endif()

    add_test(NAME unit.e2e_netcdf_roundtrip COMMAND test_e2e_netcdf_roundtrip)
    set_tests_properties(unit.e2e_netcdf_roundtrip PROPERTIES LABELS "unit;e2e;netcdf4;roundtrip" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# HELM Dependencies for worker-pool & exception-bridge test targets
#
# These targets compile `exception_bridge.cpp` and other private files directly into themselves. When MPI/HELM is found, these implementation files
# include <logs/logger.hpp> and <halo/environment.hpp> and thus require HELM::LOGS and HELM::HALO to resolve headers and link options.
# ####################################################################################################################################################
set(_amio_worker_test_targets
    test_worker_pool
    test_exception_bridge
    test_backend_factory
    test_write_snapshot
    test_read_prefetch
    test_read_bbox_validation
    test_read_capacity_guard
    test_read_close_guard
    test_read_release_view
    test_read_e2e_smoke
    test_config_loader
    test_cf_attributes
    test_read_netcdf4
    test_read_zarr
    test_read_grib2
    test_read_mixed_format)
foreach(_target IN LISTS _amio_worker_test_targets)
    if(TARGET ${_target})
        if(LOGS_FOUND)
            target_link_libraries(${_target} PRIVATE HELM::LOGS)
        endif()
        if(HALO_FOUND)
            target_link_libraries(${_target} PRIVATE HELM::HALO)
        endif()
        if(CONF_FOUND)
            target_link_libraries(${_target} PRIVATE HELM::CONF)
        endif()
    endif()
endforeach()
