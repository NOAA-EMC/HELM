find_package(Threads REQUIRED)

if(TARGET test_cli_copy)
    add_test(NAME qualification.amio_utilities COMMAND test_cli_copy)
    set_tests_properties(qualification.amio_utilities PROPERTIES LABELS "qualification;utilities;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)
endif()

add_executable(amio_qualify amio_qualify.cpp backpressure.cpp)
target_compile_features(amio_qualify PRIVATE cxx_std_20)
target_include_directories(amio_qualify PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(amio_qualify PRIVATE AMIO::amio_core)
set_target_properties(amio_qualify PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/amio/bin")

if(MPI_CXX_FOUND)
    target_sources(
        amio_qualify
        PRIVATE aggregation.cpp
                stream.cpp
                ${CMAKE_SOURCE_DIR}/src/execution/stream_retention.cpp
                ${CMAKE_SOURCE_DIR}/src/execution/aggregated_execution.cpp
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
    target_compile_definitions(amio_qualify PRIVATE AMIO_QUALIFY_HAS_MPI=1 AMIO_HAS_MPI=1)
    target_link_libraries(amio_qualify PRIVATE MPI::MPI_CXX HELM::HALO HELM::CONF HELM::LOGS Threads::Threads)
endif()

if(AMIO_BUILD_TESTING)
    add_test(NAME qualification.reference_fields COMMAND amio_qualify --self-test)
    set_tests_properties(qualification.reference_fields PROPERTIES LABELS "qualification;reference" TIMEOUT 30)

    add_test(NAME qualification.us2_backpressure COMMAND amio_qualify backpressure --delay-seconds 0 --budget-mib 64 --output
                                                         ${CMAKE_CURRENT_BINARY_DIR}/backpressure-smoke)
    set_tests_properties(qualification.us2_backpressure PROPERTIES LABELS "qualification;amio_aggregate;backpressure" TIMEOUT 30)

    if(MPI_CXX_FOUND)
        foreach(_amio_stream_case IN ITEMS reliable latest slow)
            if(_amio_stream_case STREQUAL "reliable")
                set(_amio_stream_args --policy reliable)
            elseif(_amio_stream_case STREQUAL "latest")
                set(_amio_stream_args --policy latest)
            else()
                set(_amio_stream_args --policy reliable --slow-consumer-ms 1)
            endif()
            add_test(NAME qualification.us3_stream_${_amio_stream_case}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:amio_qualify> stream
                             ${_amio_stream_args} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(
                qualification.us3_stream_${_amio_stream_case}
                PROPERTIES LABELS
                           "qualification;amio_stream;reference;${_amio_stream_case}"
                           TIMEOUT
                           180
                           PROCESSORS
                           6
                           RUN_SERIAL
                           TRUE
                           ENVIRONMENT
                           "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
        endforeach()
        if(TARGET test_mpi_stream)
            add_test(NAME qualification.us3_stream_r6 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS}
                                                              $<TARGET_FILE:test_mpi_stream> ${MPIEXEC_POSTFLAGS})
            set_tests_properties(
                qualification.us3_stream_r6
                PROPERTIES LABELS
                           "qualification;amio_stream;retention;reference"
                           TIMEOUT
                           180
                           PROCESSORS
                           6
                           RUN_SERIAL
                           TRUE
                           ENVIRONMENT
                           "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
        endif()
        foreach(_amio_ranks IN ITEMS 2 4 8)
            math(EXPR _amio_writers "${_amio_ranks} / 2")
            add_test(
                NAME qualification.us2_aggregation_r${_amio_ranks}
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_amio_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:amio_qualify> aggregate --writers
                    ${_amio_writers} --backend netcdf4 --output ${CMAKE_CURRENT_BINARY_DIR}/aggregation-r${_amio_ranks} ${MPIEXEC_POSTFLAGS})
            set_tests_properties(
                qualification.us2_aggregation_r${_amio_ranks}
                PROPERTIES LABELS
                           "qualification;amio_aggregate;reference"
                           TIMEOUT
                           120
                           PROCESSORS
                           ${_amio_ranks}
                           RUN_SERIAL
                           TRUE
                           ENVIRONMENT
                           "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
        endforeach()
    endif()

    if(AMIO_NETCDF_PAR_LIB AND TARGET MPI::MPI_CXX)
        add_executable(test_us1_persistent persistent.cpp native_readers.cpp)
        target_compile_features(test_us1_persistent PRIVATE cxx_std_20)
        target_include_directories(test_us1_persistent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(test_us1_persistent PRIVATE AMIO::amio_core MPI::MPI_CXX ${AMIO_NETCDF_PAR_LIB})
        get_target_property(_amio_qual_netcdf_include netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_qual_netcdf_include)
            target_include_directories(test_us1_persistent PRIVATE ${_amio_qual_netcdf_include})
        endif()
        foreach(_amio_ranks IN ITEMS 1 2 4 8)
            add_test(NAME qualification.us2_direct_r${_amio_ranks}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_amio_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_us1_persistent>
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(
                qualification.us2_direct_r${_amio_ranks}
                PROPERTIES LABELS
                           "qualification;amio_step;amio_native_read;amio_aggregate"
                           TIMEOUT
                           600
                           PROCESSORS
                           ${_amio_ranks}
                           RUN_SERIAL
                           TRUE
                           ENVIRONMENT
                           "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
        endforeach()
    endif()
endif()
