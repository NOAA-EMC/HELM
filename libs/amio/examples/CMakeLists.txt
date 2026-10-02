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

add_executable(example_netcdf4_read_c c/example_netcdf4_read.c)
target_link_libraries(example_netcdf4_read_c PRIVATE AMIO::amio_core m)

# -------------------------------------------------------------------
# Fortran Examples
#
# Each Fortran example links against AMIO::amio_fortran (which transitively provides the Fortran module directory and links against AMIO::amio_core
# for the runtime symbols).
# -------------------------------------------------------------------

add_executable(example_netcdf4_write_f90 fortran/example_netcdf4_write.f90)
target_link_libraries(example_netcdf4_write_f90 PRIVATE AMIO::amio_fortran)

add_executable(example_zarr_write_f90 fortran/example_zarr_write.f90)
target_link_libraries(example_zarr_write_f90 PRIVATE AMIO::amio_fortran)

add_executable(example_grib2_write_f90 fortran/example_grib2_write.f90)
target_link_libraries(example_grib2_write_f90 PRIVATE AMIO::amio_fortran)
