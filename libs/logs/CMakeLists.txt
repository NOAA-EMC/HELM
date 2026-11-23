# ##############################################################################
# LOGS — Lightweight Operational Global Status
#
# Tier 1 C++20 micro-library providing thread-safe, MPI-rank-aware operational
# status, logging, and error-handling for distributed HPC jobs. LOGS
# consolidates redundant cross-rank messages into single representative entries
# to prevent filesystem I/O storms at scale.
#
# LOGS has NO Kokkos dependency and NO Fortran interface. It uses MPI solely
# for rank identification, consolidation gather, and MPI_Abort.
# ##############################################################################

cmake_minimum_required(VERSION 3.21)
project(LOGS
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Lightweight Operational Global Status — thread-safe MPI-rank-aware logging for HELM"
)

# ─── C++20 Requirement ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Verify C++20 support at configure time.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" LOGS_HAS_CXX20_FLAG)
if(NOT LOGS_HAS_CXX20_FLAG)
    # Also try the feature-test approach via compile_features
    get_property(_cxx_known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
    if(NOT "cxx_std_20" IN_LIST _cxx_known_features)
        message(FATAL_ERROR
            "LOGS requires a C++20-capable compiler. "
            "The current compiler (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) "
            "does not appear to support C++20.")
    endif()
endif()

# ─── Options ─────────────────────────────────────────────────────────────────
option(LOGS_BUILD_TESTING "Build the LOGS test suite (GTest + RapidCheck)" OFF)

# ─── Required Dependencies ───────────────────────────────────────────────────
find_package(MPI REQUIRED COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
        "LOGS requires MPI with CXX component. "
        "Please install an MPI implementation (e.g., OpenMPI, MPICH) with C++ support.")
endif()

# ─── Library Target ──────────────────────────────────────────────────────────
add_library(logs SHARED
    src/severity.cpp
    src/log_record.cpp
    src/logger.cpp
    src/sink.cpp
    src/scoped_context.cpp
    src/stack_trace.cpp
    src/detail/context_stack.cpp
    src/detail/mpi_environment.cpp
    src/detail/consolidation.cpp
)
add_library(HELM::LOGS ALIAS logs)

target_include_directories(logs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(logs
    PUBLIC
        MPI::MPI_CXX
)

target_compile_features(logs PUBLIC cxx_std_20)

set_target_properties(logs PROPERTIES
    OUTPUT_NAME logs
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    # LOGS is a shared HELM library. Preserve its C++ API exports even when
    # the consuming workspace globally enables hidden symbol visibility.
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    CXX_EXTENSIONS OFF
)

# ─── Install & Export ─────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library binary.
install(TARGETS logs
    EXPORT LOGSTargets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install public headers.
install(DIRECTORY include/logs
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export the target set with HELM:: namespace.
install(EXPORT LOGSTargets
    FILE        LOGSTargets.cmake
    NAMESPACE   HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LOGS
)

# Generate the package config file from the template.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LOGSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LOGSConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LOGS
)

# Generate the version file (SameMajorVersion compatibility).
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LOGSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the generated config and version files.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LOGSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LOGSConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LOGS
)

# ─── Testing ─────────────────────────────────────────────────────────────────
if(LOGS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
