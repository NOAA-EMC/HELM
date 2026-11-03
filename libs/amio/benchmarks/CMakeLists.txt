find_package(Threads REQUIRED)

if(NOT AMIO_NETCDF_PAR_LIB OR NOT TARGET MPI::MPI_C)
    message(STATUS "AMIO benchmark requires parallel NetCDF and MPI C; skipping")
    return()
endif()

add_executable(amio_benchmark amio_benchmark.cpp)
target_compile_features(amio_benchmark PRIVATE cxx_std_20)
target_include_directories(amio_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(amio_benchmark PRIVATE AMIO::amio_core ${AMIO_NETCDF_PAR_LIB} MPI::MPI_CXX)
target_sources(
    amio_benchmark
    PRIVATE ${CMAKE_SOURCE_DIR}/src/execution/aggregated_execution.cpp
            ${CMAKE_SOURCE_DIR}/src/execution/context.cpp
            ${CMAKE_SOURCE_DIR}/src/execution/role_map.cpp
            ${CMAKE_SOURCE_DIR}/src/execution/transport_protocol.cpp
            ${CMAKE_SOURCE_DIR}/src/execution/mpi_transport.cpp
            ${CMAKE_SOURCE_DIR}/src/execution/credit_window.cpp
            ${CMAKE_SOURCE_DIR}/src/staging/resource_budget.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/worker_pool.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/mpi_threading.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/exception_bridge.cpp
            ${CMAKE_SOURCE_DIR}/src/c_boundary/amio_strerror.cpp)
target_include_directories(amio_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(amio_benchmark PRIVATE AMIO_HAS_MPI=1)
target_link_libraries(amio_benchmark PRIVATE HELM::HALO HELM::CONF HELM::LOGS Threads::Threads)
set_target_properties(amio_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/amio/bin")

if(AMIO_BUILD_TESTING AND MPIEXEC_EXECUTABLE AND TARGET MPI::MPI_CXX)
    add_test(
        NAME benchmark.amio_aggregated_r2
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:amio_benchmark> --mode amio-aggregated --manifest
            ${CMAKE_CURRENT_SOURCE_DIR}/qualification-manifest.txt --output ${CMAKE_CURRENT_BINARY_DIR}/aggregate-smoke --diagnostics enabled
            ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        benchmark.amio_aggregated_r2
        PROPERTIES LABELS
                   "benchmark;amio_aggregate"
                   TIMEOUT
                   120
                   PROCESSORS
                   2
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
endif()
