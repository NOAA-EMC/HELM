# ####################################################################################################################################################
# tests/integration/CMakeLists.txt
#
# Production-like multi-timestep integration tests with real backends.
#
# These tests exercise realistic grid sizes (1-degree global, 181×360) and sequential timestep reads to validate the full encode→decode pipeline under
# production-representative workloads.
# ####################################################################################################################################################

find_package(Threads QUIET)

# ####################################################################################################################################################
# test_netcdf4_multistep
#
# Writes a 3D [6, 181, 360] field via the real NetCDF_Driver (single write), then reads back each timestep slice via a rank-3 bounding box and asserts
# byte equality.  Grid: 6 forecast hours × 181 lat × 360 lon (65,160 floats/step ≈ 254 KiB/step, ~1.5 MiB total).
#
# Requires: netCDF (parallel HDF5), MPI, eckit.
# ####################################################################################################################################################
if(netCDF_FOUND AND MPI_C_FOUND AND eckit_FOUND)
    add_executable(
        test_netcdf4_multistep test_netcdf4_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp
                               ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp)

    target_include_directories(test_netcdf4_multistep PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_netcdf4_multistep PRIVATE AMIO_HAS_NETCDF=1 AMIO_HAS_ECKIT=1)
    target_link_libraries(test_netcdf4_multistep PRIVATE AMIO::public_headers eckit)

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
# Requires: g2c, eckit.
# ####################################################################################################################################################
if(g2c_FOUND AND eckit_FOUND)
    add_executable(test_grib2_multistep test_grib2_multistep.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                        ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

    target_include_directories(test_grib2_multistep PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_grib2_multistep PRIVATE AMIO_HAS_G2C=1 AMIO_HAS_ECKIT=1)
    target_link_libraries(test_grib2_multistep PRIVATE AMIO::public_headers eckit g2c::g2c)
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
# Requires: g2c, eckit.
# ####################################################################################################################################################
if(g2c_FOUND AND eckit_FOUND)
    add_executable(
        test_grib2_composition_roundtrip test_grib2_composition_roundtrip.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                         ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)

    target_include_directories(test_grib2_composition_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_grib2_composition_roundtrip PRIVATE AMIO_HAS_G2C=1 AMIO_HAS_ECKIT=1)
    target_link_libraries(test_grib2_composition_roundtrip PRIVATE AMIO::public_headers eckit g2c::g2c)
    if(Threads_FOUND)
        target_link_libraries(test_grib2_composition_roundtrip PRIVATE Threads::Threads)
    endif()

    add_test(NAME integration.grib2_composition_roundtrip COMMAND test_grib2_composition_roundtrip)
    set_tests_properties(integration.grib2_composition_roundtrip PROPERTIES LABELS "integration;grib2;composition;roundtrip" TIMEOUT 120)
endif()
