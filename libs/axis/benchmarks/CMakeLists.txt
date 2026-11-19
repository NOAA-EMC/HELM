# ##############################################################################
# benchmarks/CMakeLists.txt
#
# AXIS performance benchmark harness. Included from the root CMakeLists.txt
# only when AXIS_BUILD_BENCHMARKS=ON. Produces a single executable:
#
#   axis_benchmarks — runs all benchmark cases and outputs JSON results
#
# Uses Kokkos::HostSpace for reproducible timing across CI environments
# (Requirement 13.5).
# ##############################################################################

add_executable(axis_benchmarks axis_benchmark.cpp)

target_link_libraries(axis_benchmarks
    PRIVATE
        axis
)

target_include_directories(axis_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(axis_benchmarks PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Focused linear-conservative setup/apply diagnostic (docs/linear_conservative.md).
add_executable(axis_linear_benchmark linear_conservative.cpp)
target_link_libraries(axis_linear_benchmark PRIVATE axis)
set_target_properties(axis_linear_benchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Real-MPI distributed scaling harness (docs/distributed.md); needs HALO.
if(TARGET halo)
    add_executable(axis_distributed_scaling distributed_scaling.cpp)
    target_link_libraries(axis_distributed_scaling PRIVATE axis halo)
    set_target_properties(axis_distributed_scaling PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()
