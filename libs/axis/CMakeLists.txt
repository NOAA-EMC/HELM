# ##############################################################################
# AXIS - Arbitrary eXgrid Interpolation Solver
#
# Tier 1 C++20 micro-library providing stateless spatial interpolation
# (regridding) for Earth-system fields via Kokkos.
#
# AXIS opens no files, links no file-format libraries, and performs no MPI
# communication. All byte-level I/O is delegated to AMIO via the plain-data
# GridDescriptor ingest contract, and all distributed communication is delegated
# to HALO via the neutral HaloPattern exchange-plan descriptor.
#
# Dependencies:
#   REQUIRED : Kokkos (parallelism + memory spaces)
#   OPTIONAL : PROJ   (coordinate projection transforms, guarded by AXIS_ENABLE_PROJ)
#
# Explicitly NOT linked (HELM Law #4 / Requirement 22.7):
#   NetCDF, HDF5, ecCodes, yaml-cpp, TensorStore, g2c, eckit, AMIO, HALO
# ##############################################################################

cmake_minimum_required(VERSION 3.21)

project(
    AXIS
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Arbitrary eXgrid Interpolation Solver — stateless spatial regridding for HELM"
)

# ##############################################################################
# Global build settings
# ##############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ##############################################################################
# Options
# ##############################################################################

option(AXIS_ENABLE_PROJ      "Enable PROJ (libproj) for coordinate projection transforms" ON)
option(AXIS_ENABLE_NETCDF    "Enable NetCDF I/O and ESMF weight egress support"           ON)
option(AXIS_BUILD_TESTING    "Build the AXIS test suite (GTest + RapidCheck)"             OFF)
option(AXIS_BUILD_FORTRAN    "Build the Fortran iso_c_binding interop layer"              OFF)
option(AXIS_BUILD_PYTHON     "Build the nanobind Python bindings"                         OFF)
option(AXIS_BUILD_EXAMPLES   "Build runnable documentation examples"                      OFF)
option(AXIS_BUILD_BENCHMARKS "Build the AXIS performance benchmark harness"               OFF)

# ##############################################################################
# Dependency discovery
# ##############################################################################

# Kokkos — the ONLY required third-party dependency (HELM Law #2).
option(AXIS_FETCH_KOKKOS "Fetch and build Kokkos automatically if not found on system" ON)

find_package(Kokkos QUIET)

if(NOT TARGET Kokkos::kokkos AND AXIS_FETCH_KOKKOS)
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        5.1.1
    )

    # Ensure Kokkos builds with position independent code and suitable default backend
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend" FORCE)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_MakeAvailable(kokkos)

    # Generate a dummy KokkosConfig.cmake for subprojects (ArborX, HALO, etc.)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cmake/Kokkos")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake/Kokkos/KokkosConfig.cmake" "
        set(Kokkos_FOUND TRUE)
        set(Kokkos_VERSION \"5.1.1\")
    ")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake/Kokkos/KokkosConfigVersion.cmake" "
        set(PACKAGE_VERSION \"5.1.1\")
        set(PACKAGE_VERSION_COMPATIBLE TRUE)
        set(PACKAGE_VERSION_EXACT FALSE)
    ")
    set(Kokkos_DIR "${CMAKE_CURRENT_BINARY_DIR}/cmake/Kokkos" CACHE PATH "Kokkos stub config (FetchContent)" FORCE)
endif()

if(NOT TARGET Kokkos::kokkos)
    message(FATAL_ERROR "Kokkos REQUIRED but not found on system, and AXIS_FETCH_KOKKOS is OFF.")
endif()

# MPI — REQUIRED for distributed weights generation
find_package(MPI REQUIRED COMPONENTS CXX)

# KokkosKernels — REQUIRED accelerator for sparse linear algebra (SpMV, sort).
# AXIS uses KokkosSparse::spmv for CSR apply. It is a hard dependency so every
# build (container, HPC, CI) uses the SAME SpMV path — the former optional
# fallback to hand-rolled TeamPolicy loops silently produced a different, slower
# code path on systems where find_package(KokkosKernels) missed (e.g. some
# spack-stack module sets). Mirror the Kokkos handling: prefer a system package,
# otherwise fetch a version matching the pinned Kokkos (5.1.1). Header-only
# usage (no ETI) keeps the fetched build light.
option(AXIS_FETCH_KOKKOSKERNELS "Fetch and build KokkosKernels automatically if not found on system" ON)

find_package(KokkosKernels QUIET)

if(NOT TARGET Kokkos::kokkoskernels AND AXIS_FETCH_KOKKOSKERNELS)
    include(FetchContent)
    FetchContent_Declare(
        kokkos-kernels
        GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
        GIT_TAG        5.1.1
    )
    # Header-only consumption: disable ETI and the test/example builds so the
    # fetched KokkosKernels stays a light, fast configure/build.
    set(KokkosKernels_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ETI_ONLY OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_BLAS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_LAPACK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kokkos-kernels)
endif()

if(NOT TARGET Kokkos::kokkoskernels)
    message(FATAL_ERROR "KokkosKernels REQUIRED but not found on system, and AXIS_FETCH_KOKKOSKERNELS is OFF.")
endif()

message(STATUS "AXIS: KokkosKernels available — using KokkosSparse::spmv for CSR apply")

# PROJ — the single OPTIONAL third-party dependency for projection-string
# coordinate transforms (pure math, no equivalent in AMIO's stack).
if(AXIS_ENABLE_PROJ)
    find_package(PROJ QUIET)
    if(NOT PROJ_FOUND)
        # Fallback: try pkg-config for older PROJ installations.
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PROJ QUIET IMPORTED_TARGET proj)
        endif()
    endif()
    if(PROJ_FOUND)
        message(STATUS "AXIS: PROJ found — projection_builder enabled")
    else()
        message(WARNING "AXIS: AXIS_ENABLE_PROJ=ON but PROJ not found — projection_builder disabled")
        set_property(CACHE AXIS_ENABLE_PROJ PROPERTY VALUE OFF)
    endif()
endif()

# ##############################################################################
# Vendored third-party: ArborX (BSD-3-Clause, header-only BVH spatial index)
# Provides GPU-portable bounding-volume-hierarchy for spatial queries used by
# WeightGenerator (nearest, bilinear point-location, conservative overlap).
#
# Prefer the vendored copy at third_party/ArborX (e.g. populated via git
# submodule). Fall back to FetchContent when the directory is absent (CI,
# fresh clones without --recurse-submodules).
# ##############################################################################

# Resolution order: system package (HPC/spack installs ship ArborXConfig.cmake),
# then the vendored copy, then FetchContent. AXIS uses the ArborX v2 API
# (dimension-templated ArborX::Box), so require version >= 2.
find_package(ArborX 2.1 QUIET)

if(TARGET ArborX::ArborX)
    message(STATUS "AXIS: ArborX ${ArborX_VERSION} found via find_package (${ArborX_DIR})")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ArborX/CMakeLists.txt")
    add_subdirectory(third_party/ArborX EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(
        ArborX
        GIT_REPOSITORY https://github.com/arborx/ArborX.git
        GIT_TAG        v2.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(ArborX)
endif()

# ##############################################################################
# Library sources
# ##############################################################################

# projection_builder.cpp is always compiled; the AXIS_ENABLE_PROJ preprocessor
# guard inside selects the real PROJ implementation or a stub that throws.
set(AXIS_SOURCES
    src/detail/raii_handles.cpp
    src/detail/tspack.cpp
    src/topology/unstructured_mesh.cpp
    src/topology/structured_grid.cpp
    src/topology/mesh_factory.cpp
    src/topology/named_grid_registry.cpp
    src/topology/rule_generator.cpp
    src/topology/projection_builder.cpp
    src/topology/gmsh_writer.cpp
    src/solver/weight_generator.cpp
    src/solver/weight_generator_conservative_rect.cpp
    src/solver/weight_generator_conservative_rect_nonuniform.cpp
    src/solver/weight_generator_bilinear_rect.cpp
    src/solver/weight_generator_bilinear_rect_nonuniform.cpp
    src/solver/interpolation_matrix.cpp
    src/solver/halo_pattern.cpp
    src/solver/conservation.cpp
    src/solver/gradient_reconstructor.cpp
    src/solver/vertical_regridder.cpp
    src/solver/vector_regridder.cpp
    src/distributed/distributed_weight_generator.cpp
)

# ##############################################################################
# Target: axis (STATIC library)
# ##############################################################################

add_library(axis STATIC ${AXIS_SOURCES})
add_library(HELM::AXIS ALIAS axis)

set_target_properties(axis PROPERTIES
    EXPORT_NAME AXIS
    OUTPUT_NAME axis
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(axis
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(axis PUBLIC cxx_std_20)

# ##############################################################################
# Link dependencies
# ##############################################################################

# Kokkos — PUBLIC so downstream consumers get Kokkos transitively.
target_link_libraries(axis PUBLIC Kokkos::kokkos)

# MPI — PUBLIC so downstream consumers get MPI include paths and libraries transitively.
target_link_libraries(axis PUBLIC MPI::MPI_CXX)

# KokkosKernels — PUBLIC for KokkosSparse::spmv in apply.hpp (header-only).
# Hard dependency (found or fetched above), so link unconditionally. No feature
# macro: the code paths it once guarded were unreachable once the dependency
# became required, so they were removed rather than left behind a dead #ifdef.
target_link_libraries(axis PUBLIC Kokkos::kokkoskernels)

# ArborX — PRIVATE (vendored, header-only BVH for spatial search in WeightGenerator).
target_link_libraries(axis PRIVATE ArborX::ArborX)

# HALO — PRIVATE boundary exchange library for MPI-distributed weight generation
if(NOT TARGET halo)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../halo/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../halo" "${CMAKE_CURRENT_BINARY_DIR}/halo_build")
    endif()
endif()
if(TARGET halo)
    target_link_libraries(axis PRIVATE halo)
    target_compile_definitions(axis PUBLIC AXIS_HAVE_HALO=1)
endif()

# PROJ — PRIVATE so downstream consumers do NOT link PROJ.
if(AXIS_ENABLE_PROJ)
    if(TARGET PROJ::proj)
        target_link_libraries(axis PRIVATE PROJ::proj)
    elseif(TARGET PkgConfig::PROJ)
        target_link_libraries(axis PRIVATE PkgConfig::PROJ)
    endif()
    target_compile_definitions(axis PUBLIC AXIS_ENABLE_PROJ=1)
endif()

# NetCDF — OPTIONAL target for ESMF weight files reading and writing (AXIS-IO)
if(AXIS_ENABLE_NETCDF)
    find_package(netCDF COMPONENTS C QUIET)
    if(NOT netCDF_FOUND)
        find_package(NetCDF COMPONENTS C QUIET)
    endif()

    if(netCDF_FOUND OR NetCDF_FOUND)
        set(AXIS_HAVE_NETCDF 1)
        add_library(axis_io STATIC
            src/io/esmf_weight_io.cpp
        )
        add_library(HELM::AXIS_IO ALIAS axis_io)

        target_link_libraries(axis_io PUBLIC axis)
        if(TARGET netCDF::netcdf)
            target_link_libraries(axis_io PRIVATE netCDF::netcdf)
        elseif(TARGET NetCDF::NetCDF_C)
            target_link_libraries(axis_io PRIVATE NetCDF::NetCDF_C)
        else()
            target_link_libraries(axis_io PRIVATE ${netCDF_LIBRARIES} ${NetCDF_LIBRARIES})
        endif()

        target_compile_definitions(axis_io PUBLIC AXIS_HAVE_NETCDF=1)
        target_compile_definitions(axis PUBLIC AXIS_HAVE_NETCDF=1)

        message(STATUS "AXIS: NetCDF found — compiling optional axis_io target")
    else()
        message(WARNING "AXIS: AXIS_ENABLE_NETCDF=ON but NetCDF NOT found — optional axis_io target disabled")
        set_property(CACHE AXIS_ENABLE_NETCDF PROPERTY VALUE OFF)
    endif()
endif()

# ##############################################################################
# Fortran interop (optional)
# ##############################################################################

if(AXIS_BUILD_FORTRAN)
    enable_language(Fortran)

    add_library(axis_fortran STATIC
        src/fortran/axis_c_interop.cpp
        fortran/axis_mod.f90
    )
    add_library(HELM::AXIS_Fortran ALIAS axis_fortran)

    set_target_properties(axis_fortran PROPERTIES
        EXPORT_NAME AXIS_Fortran
        POSITION_INDEPENDENT_CODE ON
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/modules
    )

    target_include_directories(axis_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran/modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/axis/fortran>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(axis_fortran PUBLIC axis)
endif()

# ##############################################################################
# Python bindings (nanobind)
# ##############################################################################

if(AXIS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ##############################################################################
# Documentation (optional — requires Doxygen)
# ##############################################################################

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating AXIS API documentation with Doxygen..."
        VERBATIM
    )
    message(STATUS "AXIS: Doxygen found — 'docs' target available (cmake --build build --target docs)")
else()
    message(STATUS "AXIS: Doxygen not found — 'docs' target disabled (documentation build skipped)")
endif()

# ##############################################################################
# Tier 1 Isolation Scan (HELM Law #4 — static verification)
# ##############################################################################

# Custom target: run the isolation scan on demand via `cmake --build . --target check_axis_isolation`
add_custom_target(check_axis_isolation
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_tier1_isolation.sh ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running AXIS Tier 1 isolation scan (HELM Law #4 verification)..."
    VERBATIM
)

# Register the isolation scan as a CTest so `ctest -R TierIsolation` picks it up.
# This is unconditional — the isolation scan is a static check, not a unit test.
enable_testing()
add_test(
    NAME TierIsolationScan
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_tier1_isolation.sh ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(TierIsolationScan PROPERTIES
    LABELS "isolation;static-analysis"
    TIMEOUT 60
)

# ##############################################################################
# Examples (runnable documentation)
# ##############################################################################

if(AXIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ##############################################################################
# Testing
# ##############################################################################

if(AXIS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(AXIS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ##############################################################################
# Install + export
# ##############################################################################

install(
    DIRECTORY include/axis
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

set(AXIS_INSTALL_TARGETS axis)
if(AXIS_BUILD_FORTRAN)
    list(APPEND AXIS_INSTALL_TARGETS axis_fortran)
endif()

install(
    TARGETS ${AXIS_INSTALL_TARGETS}
    EXPORT AXISTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT AXISTargets
    FILE AXISTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/AXISConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/AXISConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/AXISConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/AXISConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/AXISConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

if(AXIS_BUILD_FORTRAN)
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/modules/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/axis/fortran
        FILES_MATCHING PATTERN "*.mod"
    )
endif()

# ##############################################################################
# Configure-time summary
# ##############################################################################

message(STATUS "")
message(STATUS "AXIS ${PROJECT_VERSION} configuration summary")
message(STATUS "  C++ standard    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  PROJ enabled    : ${AXIS_ENABLE_PROJ}")
message(STATUS "  NetCDF enabled  : ${AXIS_ENABLE_NETCDF}")
message(STATUS "  Build testing   : ${AXIS_BUILD_TESTING}")
message(STATUS "  Build benchmarks: ${AXIS_BUILD_BENCHMARKS}")
message(STATUS "  Build examples  : ${AXIS_BUILD_EXAMPLES}")
message(STATUS "  Build Fortran   : ${AXIS_BUILD_FORTRAN}")
message(STATUS "  Build Python    : ${AXIS_BUILD_PYTHON}")
message(STATUS "  Doxygen docs    : ${DOXYGEN_FOUND}")
message(STATUS "  Install prefix  : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
