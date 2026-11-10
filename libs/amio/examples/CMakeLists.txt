# ####################################################################################################################################################
# AMIO Examples
#
# Builds standalone example executables demonstrating the AMIO C and Fortran APIs with each backend driver (NetCDF-4, Zarr v3, GRIB2).
#
# Enable with: cmake .. -DAMIO_BUILD_EXAMPLES=ON
# ####################################################################################################################################################

# -------------------------------------------------------------------
# C Examples
#
# Each C example links against AMIO::amio_core (which transitively provides the public headers via AMIO::public_headers).
# -------------------------------------------------------------------

add_executable(example_netcdf4_write_c c/example_netcdf4_write.c)
target_link_libraries(example_netcdf4_write_c PRIVATE AMIO::amio_core m)

add_executable(example_zarr_write_c c/example_zarr_write.c)
target_link_libraries(example_zarr_write_c PRIVATE AMIO::amio_core m)

add_executable(example_grib2_write_c c/example_grib2_write.c)
target_link_libraries(example_grib2_write_c PRIVATE AMIO::amio_core m)

add_library(example_device_snapshot_c OBJECT c/example_device_snapshot.c)
target_link_libraries(example_device_snapshot_c PRIVATE AMIO::amio_core)

add_executable(example_netcdf4_read_c c/example_netcdf4_read.c)
target_link_libraries(example_netcdf4_read_c PRIVATE AMIO::amio_core m)

add_executable(example_step_collection_c c/example_step_collection.c)
target_link_libraries(example_step_collection_c PRIVATE AMIO::amio_core)
if(TARGET MPI::MPI_C)
    target_link_libraries(example_step_collection_c PRIVATE MPI::MPI_C)
    target_compile_definitions(example_step_collection_c PRIVATE AMIO_EXAMPLE_HAS_MPI=1)
endif()

if(TARGET MPI::MPI_C)
    add_executable(example_stream_c c/example_stream.c)
    target_link_libraries(example_stream_c PRIVATE AMIO::amio_core MPI::MPI_C)
    if(AMIO_BUILD_TESTING AND MPIEXEC_EXECUTABLE)
        add_test(NAME example.stream_c COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:example_stream_c>
                                               ${MPIEXEC_POSTFLAGS})
        set_tests_properties(
            example.stream_c PROPERTIES LABELS "examples;amio_stream" TIMEOUT 60 ENVIRONMENT
                                        "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
    endif()

    add_executable(example_aggregated_steps_c c/example_aggregated_steps.c)
    target_link_libraries(example_aggregated_steps_c PRIVATE AMIO::amio_core MPI::MPI_C)
    if(AMIO_BUILD_TESTING AND MPIEXEC_EXECUTABLE)
        add_test(NAME example.aggregated_steps_c COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                         $<TARGET_FILE:example_aggregated_steps_c> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(example.aggregated_steps_c PROPERTIES LABELS "examples;amio_aggregate" TIMEOUT 60 ENVIRONMENT
                                                                   "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()
endif()

# -------------------------------------------------------------------
# Fortran Examples
#
# Each Fortran example links against AMIO::amio_fortran (which transitively provides the Fortran module directory and links against AMIO::amio_core
# for the runtime symbols).
# -------------------------------------------------------------------

if(MPI_Fortran_FOUND)
    add_executable(example_stream_fortran fortran/example_stream.f90)
    set_target_properties(example_stream_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(example_stream_fortran PRIVATE AMIO::amio_fortran MPI::MPI_Fortran)
    if(AMIO_BUILD_TESTING AND MPIEXEC_EXECUTABLE)
        add_test(NAME example.stream_fortran COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS}
                                                     $<TARGET_FILE:example_stream_fortran> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(
            example.stream_fortran PROPERTIES LABELS "examples;amio_stream;amio_fortran" TIMEOUT 60 ENVIRONMENT
                                              "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=1")
    endif()

    add_executable(example_netcdf4_write_f90 fortran/example_netcdf4_write.f90)
    set_target_properties(example_netcdf4_write_f90 PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(example_netcdf4_write_f90 PRIVATE AMIO::amio_fortran MPI::MPI_Fortran)

    add_executable(example_step_collection_f90 fortran/example_step_collection.f90)
    target_link_libraries(example_step_collection_f90 PRIVATE AMIO::amio_fortran MPI::MPI_Fortran)
    set_target_properties(example_step_collection_f90 PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(example_aggregated_steps_f90 fortran/example_aggregated_steps.f90)
    target_link_libraries(example_aggregated_steps_f90 PRIVATE AMIO::amio_fortran MPI::MPI_Fortran)
    set_target_properties(example_aggregated_steps_f90 PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(example_region_batch_write_f90 fortran/example_region_batch_write.f90)
    target_link_libraries(example_region_batch_write_f90 PRIVATE AMIO::amio_fortran MPI::MPI_Fortran)
    set_target_properties(example_region_batch_write_f90 PROPERTIES LINKER_LANGUAGE Fortran)
    if(AMIO_BUILD_TESTING AND MPIEXEC_EXECUTABLE)
        add_test(NAME example.region_batch_write_fortran COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                                 $<TARGET_FILE:example_region_batch_write_f90> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(
            example.region_batch_write_fortran
            PROPERTIES LABELS
                       "examples;amio_region;amio_fortran;integration;mpi"
                       WORKING_DIRECTORY
                       "${CMAKE_BINARY_DIR}"
                       TIMEOUT
                       60
                       ENVIRONMENT
                       "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()
    if(AMIO_BUILD_TESTING)
        add_test(NAME example.aggregated_steps_fortran COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                               $<TARGET_FILE:example_aggregated_steps_f90> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(example.aggregated_steps_fortran PROPERTIES LABELS "examples;amio_aggregate;amio_fortran" TIMEOUT 60 ENVIRONMENT
                                                                         "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()

    if(AMIO_BUILD_TESTING)
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/examples/manifests")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/manifests/netcdf4_manifest.yaml" "${CMAKE_BINARY_DIR}/examples/manifests/netcdf4_manifest.yaml"
                       COPYONLY)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/manifests/netcdf4_region_batch_manifest.yaml"
                       "${CMAKE_BINARY_DIR}/examples/manifests/netcdf4_region_batch_manifest.yaml" COPYONLY)
        add_test(NAME example.netcdf4_write_fortran COMMAND example_netcdf4_write_f90)
        set_tests_properties(example.netcdf4_write_fortran PROPERTIES LABELS "examples" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
        add_test(NAME example.step_collection_fortran COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                                              $<TARGET_FILE:example_step_collection_f90> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(
            example.step_collection_fortran
            PROPERTIES LABELS
                       "examples;amio_step;amio_fortran;integration;mpi"
                       WORKING_DIRECTORY
                       "${CMAKE_BINARY_DIR}"
                       TIMEOUT
                       60
                       ENVIRONMENT
                       "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()
else()
    message(STATUS "AMIO NetCDF-4 Fortran example requires MPI Fortran; skipping")
endif()

add_library(example_device_snapshot_fortran OBJECT fortran/example_device_snapshot.f90)
target_link_libraries(example_device_snapshot_fortran PRIVATE AMIO::amio_fortran)

add_executable(example_zarr_write_f90 fortran/example_zarr_write.f90)
set_target_properties(example_zarr_write_f90 PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(example_zarr_write_f90 PRIVATE AMIO::amio_fortran)

add_executable(example_grib2_write_f90 fortran/example_grib2_write.f90)
set_target_properties(example_grib2_write_f90 PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(example_grib2_write_f90 PRIVATE AMIO::amio_fortran)
