add_executable(amio_qualify amio_qualify.cpp)
target_compile_features(amio_qualify PRIVATE cxx_std_20)
set_target_properties(amio_qualify PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/amio/bin")

if(AMIO_BUILD_TESTING)
    add_test(NAME qualification.reference_fields COMMAND amio_qualify --self-test)
    set_tests_properties(qualification.reference_fields PROPERTIES LABELS "qualification;reference" TIMEOUT 30)

    if(AMIO_NETCDF_PAR_LIB AND TARGET MPI::MPI_CXX)
        add_executable(test_us1_persistent persistent.cpp native_readers.cpp)
        target_compile_features(test_us1_persistent PRIVATE cxx_std_20)
        target_include_directories(test_us1_persistent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(test_us1_persistent PRIVATE AMIO::amio_core MPI::MPI_CXX ${AMIO_NETCDF_PAR_LIB})
        get_target_property(_amio_qual_netcdf_include netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_amio_qual_netcdf_include)
            target_include_directories(test_us1_persistent PRIVATE ${_amio_qual_netcdf_include})
        endif()
        add_test(NAME qualification.us1_persistent
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_us1_persistent> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(qualification.us1_persistent PROPERTIES
            LABELS "qualification;amio_step;amio_native_read" TIMEOUT 600 PROCESSORS 4
            ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()
endif()
