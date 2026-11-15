set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compiler commands for provenance")

# Native evidence tools are opt-in and do not change AXIS linkage.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPER_NETCDF REQUIRED IMPORTED_TARGET netcdf)
add_library(axis_paper_support STATIC native/case_io.cpp native/result_writer.cpp)
target_link_libraries(axis_paper_support PUBLIC PkgConfig::PAPER_NETCDF)
target_include_directories(axis_paper_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/native")
target_compile_features(axis_paper_support PUBLIC cxx_std_20)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE PAPER_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT PAPER_COMMIT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../AXIS_PAPER_SOURCE.json")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../AXIS_PAPER_SOURCE.json" _paper_source_snapshot)
  string(JSON PAPER_COMMIT GET "${_paper_source_snapshot}" source_commit)
endif()
configure_file(native/build_info.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp" @ONLY)
target_include_directories(axis_paper_support PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
foreach(_worker axis reference metrics)
  if(_worker STREQUAL "axis")
    set(_source native/axis_worker.cpp)
  else()
    set(_source native/${_worker}.cpp)
  endif()
  add_executable(axis_paper_${_worker} ${_source})
  target_link_libraries(axis_paper_${_worker} PRIVATE axis_paper_support)
  set_target_properties(axis_paper_${_worker} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
target_link_libraries(axis_paper_axis PRIVATE axis ArborX::ArborX)
if(AXIS_PAPER_ENABLE_ESMF)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(PaperESMF REQUIRED)
  # Validate installed headers and exact symbols at configure time. This is a
  # compile/link probe, so it also works when the executable cannot run locally.
  try_compile(PAPER_ESMC_LINKS
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/esmc_probe.cpp"
    LINK_LIBRARIES PaperESMF MPI::MPI_CXX
    CXX_STANDARD 20
    OUTPUT_VARIABLE PAPER_ESMC_PROBE_LOG)
  if(NOT PAPER_ESMC_LINKS)
    message(FATAL_ERROR "Installed ESMC headers/libraries failed the paper API probe; ${PAPER_ESMC_PROBE_LOG}")
  endif()
  add_executable(axis_paper_esmc native/esmc_worker.cpp)
  target_compile_definitions(axis_paper_esmc PRIVATE PAPER_ESMF_VERSION="${PAPER_ESMF_VERSION}")
  target_link_libraries(axis_paper_esmc PRIVATE axis_paper_support PaperESMF MPI::MPI_CXX)
  set_target_properties(axis_paper_esmc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  add_executable(axis_paper_esmc_probe cmake/esmc_probe.cpp)
  target_link_libraries(axis_paper_esmc_probe PRIVATE PaperESMF MPI::MPI_CXX)
endif()
