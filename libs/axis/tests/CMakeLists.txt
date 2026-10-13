# ##############################################################################
# tests/CMakeLists.txt
#
# AXIS test infrastructure. Included from the root CMakeLists.txt only when
# AXIS_BUILD_TESTING=ON. Provides two test executable targets:
#
#   axis_unit_tests      — GTest-based unit/integration tests
#   axis_property_tests  — RapidCheck property-based tests (≥100 iterations)
#
# Both targets link against HELM::AXIS (via the axis alias), GTest::gtest_main,
# and (for property tests) rapidcheck.
# ##############################################################################

include(GoogleTest)

option(AXIS_BUILD_PROPERTY_TESTING "Build RapidCheck-based AXIS property tests" ON)

# ##############################################################################
# Locate test dependencies
# ##############################################################################

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Fall back to FetchContent to obtain GTest if not found on the system.
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
        GIT_SHALLOW    TRUE
    )
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(googletest)
    set(GTest_FOUND TRUE)
endif()

if(AXIS_BUILD_PROPERTY_TESTING)
    find_package(rapidcheck CONFIG QUIET)
    if(NOT rapidcheck_FOUND)
        # Fall back to FetchContent (matches HALO's pattern).
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
        set(rapidcheck_FOUND TRUE)
    endif()
else()
    set(rapidcheck_FOUND FALSE)
endif()

# Ensure rapidcheck_gtest target is available
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            ${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    endif()
endif()

# ##############################################################################
# Unit test sources (GTest)
# ##############################################################################

set(AXIS_UNIT_TEST_SOURCES
    test_conservation.cpp
    test_constant_field.cpp
    test_linear_conservative.cpp
    test_performance_changes.cpp
    test_bilinear_exactness.cpp
    test_bilinear_rect.cpp
    test_bilinear_rect_nonuniform.cpp
    test_conservative_rect_nonuniform.cpp
    test_projected_lambert_conformal.cpp
    test_spmv_apply.cpp
    test_descriptor_roundtrip.cpp
    test_producer_equivalence.cpp
    test_descriptor_validation.cpp
    test_named_and_rules.cpp
    test_gmsh_roundtrip.cpp
    test_distributed_apply.cpp
    test_conservation_norm.cpp
    test_raii_handles.cpp
    test_spherical_geometry.cpp
    test_mesh_builder.cpp
    test_structured_grid_conversion.cpp
    test_reduced_gaussian_grid.cpp
    test_multi_face_grid.cpp
    test_gnomonic_projector.cpp
    test_masking.cpp
    test_morton_sort.cpp
    test_planar_clipper.cpp
    test_regular_grid_detector.cpp
    test_rectilinear_grid_detector.cpp
    test_trig_cache.cpp
    test_spherical_cap_filter.cpp
    test_vertical_regridder.cpp
    test_vertical_off_knots.cpp
    test_vertical_conservative.cpp
    test_vector_regridder.cpp
    test_tripolar_regridder.cpp
    test_coastal_renormalization.cpp
    test_warp_cooperative_clipping.cpp
    test_noaa_grib_named_grids.cpp
    test_distributed_weight_generator.cpp
    test_bicubic.cpp
)

add_executable(axis_unit_tests ${AXIS_UNIT_TEST_SOURCES})

target_link_libraries(axis_unit_tests
    PRIVATE
        axis
        GTest::gtest_main
)

if(AXIS_HAVE_NETCDF)
    target_link_libraries(axis_unit_tests PRIVATE axis_io)
    target_sources(axis_unit_tests PRIVATE test_esmf_weight_io.cpp)
endif()

target_include_directories(axis_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(axis_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(axis_unit_tests
    PROPERTIES LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

# ##############################################################################
# Property-based test sources (RapidCheck + GTest)
# ##############################################################################

if(rapidcheck_FOUND)
    set(AXIS_PROPERTY_TEST_SOURCES
        prop_field_view_roundtrip.cpp
        prop_raii_handles.cpp
        prop_structured_to_unstructured.cpp
        prop_band_corner_synthesis.cpp
        prop_reduced_gaussian_grid.cpp
        prop_csr_connectivity.cpp
        prop_named_grid_deterministic.cpp
        prop_named_grid_registration.cpp
        prop_rule_generation.cpp
        prop_descriptor_validation.cpp
        prop_producer_equivalence.cpp
        prop_buffer_adoption.cpp
        prop_sparse_matrix_bounds.cpp
        prop_conservative_nonneg.cpp
        prop_partition_of_unity.cpp
        prop_conservation.cpp
        prop_bilinear_exactness.cpp
        prop_spmv_apply.cpp
        prop_apply_extent_validation.cpp
        prop_dstarea_norm.cpp
        prop_fracarea_norm.cpp
        prop_unmapped_handling.cpp
        prop_coordsystem_consistency.cpp
        prop_halo_pattern.cpp
        prop_gmsh_roundtrip.cpp
        prop_descriptor_roundtrip.cpp
        prop_host_device_equivalence.cpp
        prop_spherical_clipper.cpp
        prop_true_bilinear.cpp
        prop_conservative_2nd_order.cpp
        prop_weight_cache.cpp
        prop_csr_format.cpp
        prop_batch_apply.cpp
        prop_dateline.cpp
        prop_masking.cpp
        prop_scrip_egress.cpp
        prop_regular_grid_detector.cpp
        prop_planar_clipper.cpp
        prop_spherical_cap_filter.cpp
        prop_trig_cache.cpp
        prop_morton_sort.cpp
        prop_conservation_optimized.cpp
        prop_optimized_equivalence.cpp
        prop_bilinear_rect_unity.cpp
        prop_bilinear_rect_agreement.cpp
        prop_bilinear_rect_sparsity.cpp
        prop_bilinear_rect_periodic.cpp
        prop_bilinear_rect_unmapped.cpp
        prop_bilinear_rect_cell_location.cpp
        prop_vector_regridder.cpp
        prop_vertical_regridder.cpp
        prop_coastal_renormalization.cpp
        prop_grid_combinations.cpp
    )

    add_executable(axis_property_tests ${AXIS_PROPERTY_TEST_SOURCES})

    target_link_libraries(axis_property_tests
        PRIVATE
            axis
            GTest::gtest_main
            rapidcheck
            $<TARGET_NAME_IF_EXISTS:rapidcheck_gtest>
    )

    target_include_directories(axis_property_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_target_properties(axis_property_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    gtest_discover_tests(axis_property_tests
        PROPERTIES LABELS "property"
        DISCOVERY_TIMEOUT 60
    )
else()
    message(STATUS "AXIS: RapidCheck not found — skipping property-based tests")
endif()

# MPI integration tests need their own MPI/Kokkos lifecycle and launcher.
if(TARGET halo)
    add_executable(axis_distributed_mpi_tests test_distributed_regridder_mpi.cpp)
    target_link_libraries(axis_distributed_mpi_tests PRIVATE axis halo)
    foreach(ranks IN ITEMS 1 2 4 8)
        add_test(NAME axis_distributed_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:axis_distributed_mpi_tests> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(axis_distributed_${ranks}_ranks PROPERTIES PROCESSORS ${ranks} TIMEOUT 180
            ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
    endforeach()
endif()

if(TARGET halo AND Kokkos_ENABLE_OPENMP)
    foreach(ranks IN ITEMS 2 4)
        math(EXPR slots "${ranks} * ${ranks}")
        add_test(NAME axis_distributed_hybrid_${ranks}x${ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:axis_distributed_mpi_tests> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(axis_distributed_hybrid_${ranks}x${ranks} PROPERTIES PROCESSORS ${slots} TIMEOUT 180
            ENVIRONMENT "OMP_NUM_THREADS=${ranks};OMP_PROC_BIND=false")
    endforeach()
endif()
