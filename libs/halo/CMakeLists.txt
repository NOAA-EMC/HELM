cmake_minimum_required(VERSION 3.21)
project(HALO
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Hardware-Abstracted Link Operations — RAII MPI with GPU-aware halo exchange"
)

# ─── Version Header Generation ───────────────────────────────────────────────
set(HALO_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(HALO_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(HALO_VERSION_PATCH ${PROJECT_VERSION_PATCH})

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/halo/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/halo/version.hpp"
    @ONLY
)

# ─── C++20 Requirement ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ─────────────────────────────────────────────────────────────────
option(HALO_GPU_AWARE_MPI "Enable GPU-aware MPI (pass device pointers directly)" OFF)
option(HALO_BUILD_TESTING "Build the HALO test suite (GTest + RapidCheck)" OFF)
option(HALO_BUILD_FORTRAN "Build the Fortran C-interop layer (halo_mod)" OFF)

# ─── Required Dependencies ───────────────────────────────────────────────────
find_package(MPI REQUIRED COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "HALO requires MPI with CXX component. Please install an MPI implementation (e.g., OpenMPI, MPICH) with C++ support.")
endif()

find_package(Kokkos REQUIRED)
if(NOT TARGET Kokkos::kokkos)
    message(FATAL_ERROR "HALO requires Kokkos. Please install Kokkos or set Kokkos_ROOT to the Kokkos installation prefix.")
endif()

# ─── Library Target ──────────────────────────────────────────────────────────
add_library(halo)
add_library(HELM::HALO ALIAS halo)

target_sources(halo PRIVATE
    src/collectives.cpp
    src/communicator.cpp
    src/environment.cpp
    src/error_policy.cpp
    src/halo_handle.cpp
    src/halo_plan.cpp
    src/request_guard.cpp
    src/window_guard.cpp
    src/detail/staging.cpp
)

target_include_directories(halo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(halo
    PUBLIC
        MPI::MPI_CXX
        Kokkos::kokkos
)

target_compile_features(halo PUBLIC cxx_std_20)

# ─── GPU-Aware MPI Compile Definition ───────────────────────────────────────
if(HALO_GPU_AWARE_MPI)
    target_compile_definitions(halo PUBLIC HALO_GPU_AWARE_MPI)
endif()

# ─── Fortran Interop Layer ───────────────────────────────────────────────────
if(HALO_BUILD_FORTRAN)
    enable_language(Fortran)

    # C interop layer (C++ compiled with extern "C"). Links the core halo
    # library PUBLIC so it inherits MPI, Kokkos, and the public include paths.
    add_library(halo_c_interop STATIC
        src/fortran/halo_c_interop.cpp
    )

    target_include_directories(halo_c_interop PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fortran
    )

    target_link_libraries(halo_c_interop
        PUBLIC
            halo
    )

    target_compile_features(halo_c_interop PUBLIC cxx_std_20)

    set_target_properties(halo_c_interop PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # Fortran module library wrapping the extern "C" symbols. Linking
    # halo_c_interop PUBLIC resolves those symbols at link time.
    add_library(halo_fortran
        fortran/halo_mod.f90
    )
    add_library(HELM::HALO_Fortran ALIAS halo_fortran)

    target_link_libraries(halo_fortran
        PUBLIC
            halo_c_interop
    )

    # Emit the generated halo_mod.mod into a dedicated module directory and
    # expose it on the target's interface so downstream Fortran consumers can
    # locate the module file.
    set_target_properties(halo_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules
    )

    target_include_directories(halo_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran>
    )
endif()

# ─── Install and Export Configuration ────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library binary
install(TARGETS halo
    EXPORT HALOTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install generated headers (version.hpp)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the Fortran interop targets and generated module files. These targets
# join the same HALOTargets export set so they are exported alongside halo.
if(HALO_BUILD_FORTRAN)
    install(TARGETS halo_c_interop halo_fortran
        EXPORT HALOTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fortran
        FILES_MATCHING PATTERN "*.mod"
    )
endif()

# Export targets with HELM:: namespace
install(EXPORT HALOTargets
    FILE HALOTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# Generate and install the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HALOConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# Generate and install the version file with SameMajorVersion compatibility
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# ─── Tier 1 Isolation Compliance (Requirement 13) ────────────────────────────
# HALO must have ZERO compile-time dependencies on any other HELM component
# (TICK, LOGS, AXIS, AMIO, SPAN, DAGR). The static scan below fails if any
# forbidden #include / Fortran `use` reference is found in the shipping sources.
set(HALO_ISOLATION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_tier1_isolation.sh")
find_program(HALO_SH_EXECUTABLE NAMES sh)

if(HALO_SH_EXECUTABLE)
    # Always-available custom target: `cmake --build . --target check_halo_isolation`.
    add_custom_target(check_halo_isolation
        COMMAND ${HALO_SH_EXECUTABLE} ${HALO_ISOLATION_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Scanning HALO sources for forbidden HELM component includes (Tier 1 isolation)"
        VERBATIM
    )
else()
    message(WARNING "POSIX 'sh' not found; the Tier 1 isolation check (check_isolation) will be unavailable.")
endif()

# ─── Testing ─────────────────────────────────────────────────────────────────
if(HALO_BUILD_TESTING)
    enable_testing()

    # Register the Tier 1 isolation scan as a CTest so it runs with the suite.
    if(HALO_SH_EXECUTABLE)
        add_test(
            NAME tier1_isolation
            COMMAND ${HALO_SH_EXECUTABLE} ${HALO_ISOLATION_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(tier1_isolation PROPERTIES
            TIMEOUT 30
            LABELS "static;isolation"
        )
    endif()

    add_subdirectory(tests)

    if(HALO_BUILD_FORTRAN)
        add_subdirectory(tests_fortran)
    endif()
endif()
