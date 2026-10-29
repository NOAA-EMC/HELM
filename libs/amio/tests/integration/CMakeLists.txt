# ####################################################################################################################################################
# tests/integration/CMakeLists.txt
#
# Production-like multi-timestep integration tests with real backends.
#
# These tests exercise realistic grid sizes (1-degree global, 181×360) and sequential timestep reads to validate the full encode→decode pipeline under
# production-representative workloads.
# ####################################################################################################################################################

find_package(Threads QUIET)

add_executable(test_catalog_guarantees test_catalog_guarantees.cpp
    ${CMAKE_SOURCE_DIR}/src/catalog/catalog_format.cpp
    ${CMAKE_SOURCE_DIR}/src/catalog/posix_catalog.cpp
    ${CMAKE_SOURCE_DIR}/src/catalog/collection_reader.cpp)
target_include_directories(test_catalog_guarantees PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_catalog_guarantees PRIVATE AMIO::public_headers)
add_test(NAME integration.catalog_guarantees COMMAND test_catalog_guarantees)
set_tests_properties(integration.catalog_guarantees PROPERTIES
    LABELS "integration;catalog;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)

if(MPI_CXX_FOUND)
    add_executable(
        test_mpi_steps test_mpi_steps.cpp ${CMAKE_SOURCE_DIR}/src/execution/context.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp
                         ${CMAKE_SOURCE_DIR}/src/steps/collective_preflight.cpp ${CMAKE_SOURCE_DIR}/src/schema/partition_validator.cpp
                         ${CMAKE_SOURCE_DIR}/src/staging/resource_budget.cpp)
    target_include_directories(test_mpi_steps PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_mpi_steps PRIVATE AMIO_HAS_MPI=1)
    target_link_libraries(test_mpi_steps PRIVATE AMIO::public_headers MPI::MPI_CXX HELM::HALO)
    add_test(NAME integration.mpi_steps COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
                                                $<TARGET_FILE:test_mpi_steps> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(integration.mpi_steps PROPERTIES LABELS "integration;mpi;steps;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)

    add_executable(test_mpi_step_failure test_mpi_step_failure.cpp ${CMAKE_SOURCE_DIR}/src/execution/context.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)
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
        test_netcdf4_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp
        ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)

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
    add_executable(test_grib2_multistep test_grib2_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                        ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_steps.cpp
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
        test_grib2_composition_roundtrip test_grib2_composition_roundtrip.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                         ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_steps.cpp
                                         ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

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
        add_executable(test_mpi_netcdf_step_payload test_mpi_netcdf_step_payload.cpp
            ${CMAKE_SOURCE_DIR}/src/schema/schema.cpp
            ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp
            ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp
            ${CMAKE_SOURCE_DIR}/src/workers/comm_split.cpp)
        target_link_libraries(test_mpi_netcdf_step_payload PRIVATE AMIO::public_headers driver_netcdf MPI::MPI_CXX HELM::CONF HELM::HALO)
        get_target_property(_amio_mpi_step_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_mpi_step_nc_inc)
            target_include_directories(test_mpi_netcdf_step_payload PRIVATE ${_amio_mpi_step_nc_inc})
        endif()
        target_include_directories(test_mpi_netcdf_step_payload PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(test_mpi_netcdf_step_payload PRIVATE ${AMIO_NETCDF_PAR_LIB})
        if(CONF_FOUND)
            target_link_libraries(test_mpi_netcdf_step_payload PRIVATE HELM::CONF)
        endif()
        add_test(NAME integration.mpi_netcdf_step_payload
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_netcdf_step_payload> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(integration.mpi_netcdf_step_payload PROPERTIES
            LABELS "integration;mpi;netcdf;amio_step" TIMEOUT 60 RUN_SERIAL TRUE)
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
    add_test(NAME integration.mpi_netcdf_sync COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                      $<TARGET_FILE:test_mpi_region_write> sync ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_thread_gate COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                      $<TARGET_FILE:test_mpi_region_write> single ${MPIEXEC_POSTFLAGS})
    add_test(NAME integration.mpi_single_rank_netcdf COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                                                             $<TARGET_FILE:test_mpi_region_write> single ${MPIEXEC_POSTFLAGS})
    set_tests_properties(integration.mpi_netcdf_regions integration.mpi_netcdf_sync integration.mpi_thread_gate integration.mpi_single_rank_netcdf
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
