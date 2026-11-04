# ####################################################################################################################################################
# tests/integration/CMakeLists.txt
#
# Production-like multi-timestep integration tests with real backends.
#
# These tests exercise realistic grid sizes (1-degree global, 181×360) and sequential timestep reads to validate the full encode→decode pipeline under
# production-representative workloads.
# ####################################################################################################################################################

find_package(Threads QUIET)

add_executable(test_cli_contract test_cli_contract.cpp ${CMAKE_SOURCE_DIR}/tools/result_writer.cpp)
target_include_directories(test_cli_contract PRIVATE ${CMAKE_SOURCE_DIR}/tools)
add_test(NAME integration.cli_contract COMMAND test_cli_contract)
set_tests_properties(integration.cli_contract PROPERTIES LABELS "integration;cli" TIMEOUT 30)
add_executable(test_cli_conversion test_cli_conversion.cpp ${CMAKE_SOURCE_DIR}/tools/selection_parse.cpp)
target_include_directories(test_cli_conversion PRIVATE ${CMAKE_SOURCE_DIR}/tools ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
add_test(NAME integration.cli_conversion COMMAND test_cli_conversion)
set_tests_properties(integration.cli_conversion PROPERTIES LABELS "integration;cli" TIMEOUT 30)
add_executable(
    test_cli_copy
    test_cli_copy.cpp
    ${CMAKE_SOURCE_DIR}/tests/qualification/utility_fixtures.cpp
    ${CMAKE_SOURCE_DIR}/tools/result_writer.cpp
    ${CMAKE_SOURCE_DIR}/tools/result_helpers.cpp
    ${CMAKE_SOURCE_DIR}/tools/selection_parse.cpp
    ${CMAKE_SOURCE_DIR}/tools/inspect.cpp
    ${CMAKE_SOURCE_DIR}/tools/validate.cpp
    ${CMAKE_SOURCE_DIR}/tools/copy.cpp
    ${CMAKE_SOURCE_DIR}/tools/diagnostics.cpp
    ${CMAKE_SOURCE_DIR}/src/catalog/catalog_format.cpp
    ${CMAKE_SOURCE_DIR}/src/catalog/collection_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/schema/schema.cpp)
target_include_directories(test_cli_copy PRIVATE ${CMAKE_SOURCE_DIR}/tools ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
                                                 ${CMAKE_SOURCE_DIR}/tests/qualification)
target_link_libraries(test_cli_copy PRIVATE AMIO::amio_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM|Intel")
    target_compile_options(test_cli_copy PRIVATE -UNDEBUG)
endif()
if(MPI_CXX_FOUND)
    target_compile_definitions(test_cli_copy PRIVATE AMIO_TOOLS_HAS_MPI=1 AMIO_CLI_TEST_HAS_MPI=1)
    target_link_libraries(test_cli_copy PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME integration.amio_utilities COMMAND test_cli_copy)
set_tests_properties(integration.amio_utilities PROPERTIES LABELS "integration;utilities;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)

# Host-only accelerator qualification: unsupported device/ownership descriptors and malformed layouts must be rejected without reading their payloads.
if(TARGET test_memory_submission)
    add_test(NAME integration.accelerator_host_rejections COMMAND test_memory_submission)
    set_tests_properties(integration.accelerator_host_rejections PROPERTIES LABELS "integration;amio_accelerator;host_only" TIMEOUT 30)
endif()

if(AMIO_HAS_KOKKOS_RUNTIME)
    add_executable(test_device_snapshot test_device_snapshot.cpp ${CMAKE_SOURCE_DIR}/src/memory/memory_provider.cpp
                                        ${CMAKE_SOURCE_DIR}/src/memory/host_snapshot.cpp ${CMAKE_SOURCE_DIR}/src/memory/kokkos_snapshot.cpp)
    target_include_directories(test_device_snapshot PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_device_snapshot PRIVATE AMIO_HAS_KOKKOS_RUNTIME=1)
    target_link_libraries(test_device_snapshot PRIVATE ${AMIO_MEMORY_BACKEND_TARGET})
    add_test(NAME integration.device_snapshot COMMAND test_device_snapshot)
    set_tests_properties(integration.device_snapshot PROPERTIES LABELS "integration;amio_accelerator;kokkos_snapshot" TIMEOUT 60)
endif()

add_executable(test_catalog_guarantees test_catalog_guarantees.cpp ${CMAKE_SOURCE_DIR}/src/catalog/catalog_format.cpp
                                       ${CMAKE_SOURCE_DIR}/src/catalog/posix_catalog.cpp ${CMAKE_SOURCE_DIR}/src/catalog/collection_reader.cpp)
target_include_directories(test_catalog_guarantees PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_catalog_guarantees PRIVATE AMIO::public_headers)
add_test(NAME integration.catalog_guarantees COMMAND test_catalog_guarantees)
set_tests_properties(integration.catalog_guarantees PROPERTIES LABELS "integration;catalog;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)

if(MPI_CXX_FOUND AND AMIO_NETCDF_PAR_LIB)
    add_executable(test_operator_roundtrip test_operator_roundtrip.cpp)
    target_link_libraries(test_operator_roundtrip PRIVATE AMIO::amio_core MPI::MPI_CXX)
    if(AMIO_NETCDF_PAR_LIB)
        target_link_libraries(test_operator_roundtrip PRIVATE ${AMIO_NETCDF_PAR_LIB})
        if(AMIO_NETCDF_MPI_INCLUDE_DIR)
            target_include_directories(test_operator_roundtrip BEFORE PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
        endif()
    else()
        target_link_libraries(test_operator_roundtrip PRIVATE netCDF::netcdf)
    endif()
    add_test(NAME integration.operator_roundtrip COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                         $<TARGET_FILE:test_operator_roundtrip> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.operator_roundtrip
        PROPERTIES LABELS
                   "integration;mpi;operators;integrity;amio_operators"
                   TIMEOUT
                   120
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
endif()

if(MPI_CXX_FOUND AND g2c_FOUND)
    add_executable(test_operator_grib2 test_operator_grib2.cpp)
    target_link_libraries(test_operator_grib2 PRIVATE AMIO::amio_core MPI::MPI_CXX)
    add_test(NAME integration.operator_grib2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                                                     $<TARGET_FILE:test_operator_grib2> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.operator_grib2
        PROPERTIES LABELS
                   "integration;mpi;operators;amio_operators;grib2"
                   TIMEOUT
                   120
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
endif()

if(MPI_CXX_FOUND)
    add_executable(
        test_mpi_stream
        test_mpi_stream.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/stream_execution.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/stream_retention.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/context.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/role_map.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/transport_protocol.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/mpi_transport.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/credit_window.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)
    target_include_directories(test_mpi_stream PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_mpi_stream PRIVATE AMIO_HAS_MPI=1)
    target_link_libraries(test_mpi_stream PRIVATE AMIO::amio_core AMIO::public_headers MPI::MPI_CXX HELM::HALO Threads::Threads)
    add_test(NAME integration.mpi_stream COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_stream>
                                                 ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.mpi_stream
        PROPERTIES LABELS
                   "integration;mpi;stream"
                   TIMEOUT
                   120
                   PROCESSORS
                   6
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")

    add_executable(test_mpi_core_roles test_mpi_core_roles.cpp)
    target_link_libraries(test_mpi_core_roles PRIVATE AMIO::amio_core MPI::MPI_CXX)
    add_test(NAME integration.mpi_core_roles COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                     $<TARGET_FILE:test_mpi_core_roles> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.mpi_core_roles
        PROPERTIES LABELS
                   "integration;mpi;roles;amio_aggregate"
                   TIMEOUT
                   60
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

    add_executable(test_mpi_role_api test_mpi_role_api.cpp)
    target_link_libraries(test_mpi_role_api PRIVATE AMIO::amio_core MPI::MPI_CXX)
    add_test(NAME integration.mpi_role_api COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                                                   $<TARGET_FILE:test_mpi_role_api> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.mpi_role_api
        PROPERTIES LABELS
                   "integration;mpi;roles;amio_aggregate"
                   TIMEOUT
                   60
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

    add_executable(
        test_mpi_aggregation
        test_mpi_aggregation.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/aggregated_execution.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/context.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/role_map.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/transport_protocol.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/mpi_transport.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/thread_pinning.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/credit_window.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_steps.cpp)
    target_include_directories(test_mpi_aggregation PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_mpi_aggregation PRIVATE AMIO_HAS_MPI=1)
    target_link_libraries(test_mpi_aggregation PRIVATE AMIO::public_headers MPI::MPI_CXX HELM::HALO HELM::CONF Threads::Threads)
    if(netCDF_FOUND AND AMIO_NETCDF_PAR_LIB)
        target_sources(
            test_mpi_aggregation
            PRIVATE ${CMAKE_SOURCE_DIR}/src/schema/schema.cpp ${CMAKE_SOURCE_DIR}/src/staging/resource_budget.cpp
                    ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp)
        target_compile_definitions(test_mpi_aggregation PRIVATE AMIO_HAS_NETCDF=1 AMIO_TEST_NATIVE_NETCDF=1)
        target_link_libraries(test_mpi_aggregation PRIVATE driver_netcdf HELM::CONF ${AMIO_NETCDF_PAR_LIB})
        if(AMIO_NETCDF_MPI_INCLUDE_DIR)
            target_include_directories(test_mpi_aggregation BEFORE PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
        endif()
        get_target_property(_amio_agg_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_agg_nc_inc)
            target_include_directories(test_mpi_aggregation PRIVATE ${_amio_agg_nc_inc})
        endif()
    endif()
    add_test(NAME integration.mpi_aggregation COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                                                      $<TARGET_FILE:test_mpi_aggregation> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.mpi_aggregation
        PROPERTIES LABELS
                   "integration;mpi;aggregation;amio_aggregate"
                   TIMEOUT
                   60
                   RUN_SERIAL
                   TRUE
                   ENVIRONMENT
                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

    add_executable(
        test_mpi_steps
        test_mpi_steps.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/context.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/role_map.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
        ${CMAKE_SOURCE_DIR}/src/steps/collective_preflight.cpp
        ${CMAKE_SOURCE_DIR}/src/schema/partition_validator.cpp
        ${CMAKE_SOURCE_DIR}/src/staging/resource_budget.cpp)
    target_include_directories(test_mpi_steps PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_mpi_steps PRIVATE AMIO_HAS_MPI=1)
    target_link_libraries(test_mpi_steps PRIVATE AMIO::public_headers MPI::MPI_CXX HELM::HALO)
    add_test(NAME integration.mpi_steps COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_steps>
                                                ${MPIEXEC_POSTFLAGS})
    set_tests_properties(integration.mpi_steps PROPERTIES LABELS "integration;mpi;steps;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)

    add_executable(test_mpi_step_failure test_mpi_step_failure.cpp ${CMAKE_SOURCE_DIR}/src/execution/context.cpp
                                         ${CMAKE_SOURCE_DIR}/src/execution/role_map.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)
    target_include_directories(test_mpi_step_failure PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_mpi_step_failure PRIVATE AMIO_HAS_MPI=1)
    target_link_libraries(test_mpi_step_failure PRIVATE AMIO::public_headers MPI::MPI_CXX HELM::HALO)
    add_test(NAME integration.mpi_step_failure COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                       $<TARGET_FILE:test_mpi_step_failure> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(integration.mpi_step_failure PROPERTIES LABELS "integration;mpi;steps;failure;amio_fault" TIMEOUT 60 RUN_SERIAL TRUE)
endif()

# ####################################################################################################################################################
# test_netcdf4_multistep
#
# Writes a 3D [6, 181, 360] field via the real NetCDF_Driver (single write), then reads back each timestep slice via a rank-3 bounding box and asserts
# byte equality.  Grid: 6 forecast hours × 181 lat × 360 lon (65,160 floats/step ≈ 254 KiB/step, ~1.5 MiB total).
#
# Requires: netCDF (parallel HDF5), MPI.
# ####################################################################################################################################################
if(netCDF_FOUND AND MPI_C_FOUND)
    add_executable(
        test_netcdf4_multistep
        test_netcdf4_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_steps.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)

    target_include_directories(test_netcdf4_multistep PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_netcdf4_multistep PRIVATE AMIO_HAS_NETCDF=1)
    target_link_libraries(test_netcdf4_multistep PRIVATE AMIO::public_headers)

    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_nc_inc)
            target_include_directories(test_netcdf4_multistep PRIVATE ${_nc_inc})
        endif()
        target_link_libraries(test_netcdf4_multistep PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_netcdf4_multistep PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_NETCDF_MPI_INCLUDE_DIR)
        target_include_directories(test_netcdf4_multistep PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
    endif()
    target_link_libraries(test_netcdf4_multistep PRIVATE MPI::MPI_C)
    if(MPI_CXX_FOUND)
        target_link_libraries(test_netcdf4_multistep PRIVATE MPI::MPI_CXX)
    endif()
    if(Threads_FOUND)
        target_link_libraries(test_netcdf4_multistep PRIVATE Threads::Threads)
    endif()

    add_test(NAME integration.netcdf4_multistep COMMAND test_netcdf4_multistep)
    set_tests_properties(integration.netcdf4_multistep PROPERTIES LABELS "integration;netcdf4;multistep" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_grib2_multistep
#
# Writes 4 timesteps of a 181×360 F32 field via the real GRIB2_Driver (JPEG2000 DRT, lossless for integer-valued floats), then reads back each
# timestep and asserts byte equality.  Grid: 4 timesteps × 181 lat × 360 lon (65,160 floats/step ≈ 254 KiB/step).
#
# Requires: g2c.
# ####################################################################################################################################################
if(g2c_FOUND)
    add_executable(
        test_grib2_multistep
        test_grib2_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_steps.cpp
        ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

    target_include_directories(test_grib2_multistep PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_grib2_multistep PRIVATE AMIO_HAS_G2C=1)
    target_link_libraries(test_grib2_multistep PRIVATE AMIO::public_headers g2c::g2c)
    if(Threads_FOUND)
        target_link_libraries(test_grib2_multistep PRIVATE Threads::Threads)
    endif()

    add_test(NAME integration.grib2_multistep COMMAND test_grib2_multistep)
    set_tests_properties(integration.grib2_multistep PROPERTIES LABELS "integration;grib2;multistep" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# test_grib2_composition_roundtrip
#
# Full encode-decode round-trip for composition PDTs (4.8, 4.40, 4.44, 4.45, 4.46, 4.48, 4.49) and GDT 3.40. Encodes an F32 field with lossless DRT,
# decodes, verifies byte-equal grid-point values, Ni/Nj geometry, and Field_Identity_Name consistency across encode/decode.  Also verifies backward
# compatibility: PDT 4.0 / GDT 3.0 produces unchanged identity names and identical output.
#
# Requires: g2c.
# ####################################################################################################################################################
if(g2c_FOUND)
    add_executable(
        test_grib2_composition_roundtrip
        test_grib2_composition_roundtrip.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_steps.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

    target_include_directories(test_grib2_composition_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_grib2_composition_roundtrip PRIVATE AMIO_HAS_G2C=1)
    target_link_libraries(test_grib2_composition_roundtrip PRIVATE AMIO::public_headers g2c::g2c)
    if(Threads_FOUND)
        target_link_libraries(test_grib2_composition_roundtrip PRIVATE Threads::Threads)
    endif()

    add_test(NAME integration.grib2_composition_roundtrip COMMAND test_grib2_composition_roundtrip)
    set_tests_properties(integration.grib2_composition_roundtrip PROPERTIES LABELS "integration;grib2;composition;roundtrip" TIMEOUT 120)

    add_executable(test_evolution_grib2 test_evolution_grib2.cpp)
    target_link_libraries(test_evolution_grib2 PRIVATE AMIO::amio_core)
    if(MPI_CXX_FOUND)
        target_compile_definitions(test_evolution_grib2 PRIVATE AMIO_TEST_HAS_MPI=1)
        target_link_libraries(test_evolution_grib2 PRIVATE MPI::MPI_CXX)
    endif()
    add_test(NAME integration.evolution_grib2 COMMAND test_evolution_grib2)
    set_tests_properties(integration.evolution_grib2 PROPERTIES LABELS "integration;grib2;evolution;roundtrip" TIMEOUT 120)
endif()

# ####################################################################################################################################################
# HELM CONF dependency
#
# These targets parse their driver manifests with conf::Config and compile the driver sources (which include <conf/config.hpp>) directly into the test
# binary. HELM::CONF is PRIVATE to amio_core, so it must be linked explicitly here for the conf headers to resolve.
# ####################################################################################################################################################
foreach(_target IN ITEMS test_netcdf4_multistep test_grib2_multistep test_grib2_composition_roundtrip)
    if(TARGET ${_target} AND CONF_FOUND)
        target_link_libraries(${_target} PRIVATE HELM::CONF)
    endif()
endforeach()

# Shared-dataset parallel writes must be exercised with actual MPI ranks.
if(netCDF_FOUND AND MPI_CXX_FOUND)
    if(AMIO_NETCDF_PAR_LIB)
        add_executable(
            test_mpi_netcdf_step_payload
            test_mpi_netcdf_step_payload.cpp ${CMAKE_SOURCE_DIR}/src/schema/schema.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
            ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)
        target_link_libraries(test_mpi_netcdf_step_payload PRIVATE AMIO::public_headers driver_netcdf MPI::MPI_CXX HELM::CONF HELM::HALO)
        get_target_property(_amio_mpi_step_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_mpi_step_nc_inc)
            target_include_directories(test_mpi_netcdf_step_payload PRIVATE ${_amio_mpi_step_nc_inc})
        endif()
        target_include_directories(test_mpi_netcdf_step_payload PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_compile_definitions(test_mpi_netcdf_step_payload PRIVATE AMIO_HAS_MPI=1 AMIO_HAS_NETCDF=1)
        if(AMIO_NETCDF_MPI_INCLUDE_DIR)
            target_include_directories(test_mpi_netcdf_step_payload BEFORE PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
        endif()
        target_link_libraries(test_mpi_netcdf_step_payload PRIVATE ${AMIO_NETCDF_PAR_LIB})
        if(CONF_FOUND)
            target_link_libraries(test_mpi_netcdf_step_payload PRIVATE HELM::CONF)
        endif()
        add_test(NAME integration.mpi_netcdf_step_payload COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                                                                  $<TARGET_FILE:test_mpi_netcdf_step_payload> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(integration.mpi_netcdf_step_payload PROPERTIES LABELS "integration;mpi;netcdf;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)
    endif()

    add_executable(test_mpi_region_write test_mpi_region_write.cpp)
    target_link_libraries(test_mpi_region_write PRIVATE AMIO::amio_core HELM::HALO HELM::LOGS)
    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_nc_inc)
            target_include_directories(test_mpi_region_write PRIVATE ${_nc_inc})
        endif()
        target_link_libraries(test_mpi_region_write PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_mpi_region_write PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_HAS_TENSORSTORE)
        target_compile_definitions(test_mpi_region_write PRIVATE AMIO_TEST_TENSORSTORE=1)
        target_link_libraries(test_mpi_region_write PRIVATE TensorStore::tensorstore)
    endif()
    add_test(NAME integration.mpi_netcdf_regions COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                         $<TARGET_FILE:test_mpi_region_write> ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_prepared_schema COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                                 $<TARGET_FILE:test_mpi_region_write> prepared-schema ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_schema_mismatch COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                                 $<TARGET_FILE:test_mpi_region_write> schema-mismatch ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_late_schema COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                             $<TARGET_FILE:test_mpi_region_write> late-schema ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_batch COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                       $<TARGET_FILE:test_mpi_region_write> batch ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_batch_invalid COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                               $<TARGET_FILE:test_mpi_region_write> batch-invalid ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_batch_mismatch COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                                $<TARGET_FILE:test_mpi_region_write> batch-mismatch ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_netcdf_sync COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                      $<TARGET_FILE:test_mpi_region_write> sync ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_thread_gate COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                      $<TARGET_FILE:test_mpi_region_write> single ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_single_rank_netcdf COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                                                             $<TARGET_FILE:test_mpi_region_write> single ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        integration.mpi_netcdf_regions
        integration.mpi_netcdf_prepared_schema
        integration.mpi_netcdf_schema_mismatch
        integration.mpi_netcdf_late_schema
        integration.mpi_netcdf_batch
        integration.mpi_netcdf_batch_invalid
        integration.mpi_netcdf_batch_mismatch
        integration.mpi_netcdf_sync
        integration.mpi_thread_gate
        integration.mpi_single_rank_netcdf
        PROPERTIES LABELS "integration;mpi;write" TIMEOUT 90 RUN_SERIAL TRUE)
    if(AMIO_HAS_TENSORSTORE)
        add_test(NAME integration.mpi_zarr_regions COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                           $<TARGET_FILE:test_mpi_region_write> zarr3 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(integration.mpi_zarr_regions PROPERTIES LABELS "integration;mpi;write;zarr3" TIMEOUT 90 RUN_SERIAL TRUE)
    else()
        add_test(NAME integration.mpi_nczarr_rejected COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                              $<TARGET_FILE:test_mpi_region_write> nczarr ${MPIEXEC_POSTFLAGS})
        set_tests_properties(integration.mpi_nczarr_rejected PROPERTIES LABELS "integration;mpi;write;zarr3" TIMEOUT 90 RUN_SERIAL TRUE)
    endif()
endif()

if(g2c_FOUND AND MPI_CXX_FOUND)
    add_executable(test_mpi_grib2_write test_mpi_grib2_write.cpp)
    target_link_libraries(test_mpi_grib2_write PRIVATE AMIO::amio_core HELM::HALO HELM::LOGS g2c::g2c)
    add_test(NAME integration.mpi_grib2_messages COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                         $<TARGET_FILE:test_mpi_grib2_write> ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_grib2_thread_gate COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                            $<TARGET_FILE:test_mpi_grib2_write> ${MPIEXEC_POSTFLAGS} single)
    add_test(NAME integration.mpi_grib2_single_rank COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                                                            $<TARGET_FILE:test_mpi_grib2_write> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(integration.mpi_grib2_messages integration.mpi_grib2_thread_gate integration.mpi_grib2_single_rank
                         PROPERTIES LABELS "integration;mpi;write;grib2" TIMEOUT 120 RUN_SERIAL TRUE)
endif()
