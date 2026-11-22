# ##############################################################################
# AXIS Examples — Compiled documentation examples
#
# These examples are compiled (and optionally run) as part of CI to prevent
# documentation rot. They demonstrate correct usage of the AXIS API.
# ##############################################################################

cmake_minimum_required(VERSION 3.21)

# Only build examples when AXIS is the top-level project or AXIS_BUILD_EXAMPLES is ON
option(AXIS_BUILD_EXAMPLES "Build AXIS runnable examples" ON)

if(NOT AXIS_BUILD_EXAMPLES)
    return()
endif()

# ##############################################################################
# C++ Examples
# ##############################################################################

# Example 1: Basic bilinear interpolation workflow
add_executable(example_bilinear_regrid example_bilinear_regrid.cpp)
target_link_libraries(example_bilinear_regrid PRIVATE axis Kokkos::kokkos)
target_compile_features(example_bilinear_regrid PRIVATE cxx_std_20)

# Example 2: Conservative remapping with weight caching
add_executable(example_conservative_cache example_conservative_cache.cpp)
target_link_libraries(example_conservative_cache PRIVATE axis Kokkos::kokkos)
target_compile_features(example_conservative_cache PRIVATE cxx_std_20)

# Example 3: Batch apply for multi-variable regridding
add_executable(example_batch_apply example_batch_apply.cpp)
target_link_libraries(example_batch_apply PRIVATE axis Kokkos::kokkos)
target_compile_features(example_batch_apply PRIVATE cxx_std_20)

# Register examples as CTest targets for CI documentation-rot detection
add_test(NAME ExampleBilinearRegrid COMMAND example_bilinear_regrid)
add_test(NAME ExampleConservativeCache COMMAND example_conservative_cache)
add_test(NAME ExampleBatchApply COMMAND example_batch_apply)

set_tests_properties(
    ExampleBilinearRegrid
    ExampleConservativeCache
    ExampleBatchApply
    PROPERTIES
        LABELS "examples;documentation"
        TIMEOUT 30
)

add_executable(example_cece_point_mapping example_cece_point_mapping.cpp)
target_link_libraries(example_cece_point_mapping PRIVATE axis)
add_test(NAME ExampleCecePointMapping COMMAND example_cece_point_mapping)
set_tests_properties(ExampleCecePointMapping PROPERTIES LABELS "examples;documentation"
    ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false" TIMEOUT 30)
