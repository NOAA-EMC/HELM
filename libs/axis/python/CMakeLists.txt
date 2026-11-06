# Python bindings for AXIS via nanobind
cmake_minimum_required(VERSION 3.21)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(axis_py axis_py.cpp)

target_link_libraries(axis_py PRIVATE axis)
if(AXIS_HAVE_NETCDF)
    target_link_libraries(axis_py PRIVATE axis_io)
endif()
target_include_directories(axis_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/ArborX/src
)

# On macOS, retain native dependency search paths in both build and installed
# extensions, and align the OpenMP runtime with shared Kokkos when applicable.
if(APPLE)
    # Use the actual linked library locations. During pip/uv build isolation,
    # Python's sys.prefix points at a temporary build environment, which does
    # not contain native libraries discovered in Conda or a system prefix.
    set(_axis_py_rpaths "@loader_path")
    if(AXIS_ENABLE_PROJ)
        if(TARGET PROJ::proj)
            list(APPEND _axis_py_rpaths "$<TARGET_FILE_DIR:PROJ::proj>")
        elseif(TARGET PkgConfig::PROJ)
            list(APPEND _axis_py_rpaths ${PROJ_LIBRARY_DIRS})
        endif()
    endif()

    # Directory of whatever OpenMP runtime CMake actually discovered.
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        # OpenMP_omp_LIBRARY is the resolved path to the runtime (libomp/libgomp).
        if(OpenMP_omp_LIBRARY)
            get_filename_component(_omp_dir "${OpenMP_omp_LIBRARY}" DIRECTORY)
            list(APPEND _axis_py_rpaths "${_omp_dir}")
        endif()
        foreach(_omp_lib ${OpenMP_CXX_LIBRARIES})
            if(EXISTS "${_omp_lib}")
                get_filename_component(_omp_dir "${_omp_lib}" DIRECTORY)
                list(APPEND _axis_py_rpaths "${_omp_dir}")
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _axis_py_rpaths)
    set_target_properties(axis_py PROPERTIES
        INSTALL_RPATH "${_axis_py_rpaths}"
        INSTALL_RPATH_USE_LINK_PATH ON
        BUILD_WITH_INSTALL_RPATH ON
    )

    # Rewrite the libomp install-name recorded in the binary to @rpath/... by
    # reading the ACTUAL dependency from the built file (see relocate_openmp.cmake),
    # rather than enumerating known absolute paths.
    find_program(OTOOL_EXECUTABLE otool)
    if(OTOOL_EXECUTABLE AND CMAKE_INSTALL_NAME_TOOL)
        # A shared Kokkos library may already load a different OpenMP runtime
        # than the compiler discovers. Keep the extension on that same runtime.
        set(_axis_kokkos_runtime_arg "")
        if(TARGET Kokkos::kokkoscore)
            get_target_property(_axis_kokkos_type Kokkos::kokkoscore TYPE)
            if(_axis_kokkos_type STREQUAL "SHARED_LIBRARY")
                set(_axis_kokkos_runtime_arg "-DKOKKOS_LIBRARY=$<TARGET_FILE:Kokkos::kokkoscore>")
            endif()
        endif()
        add_custom_command(TARGET axis_py POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DTARGET_FILE=$<TARGET_FILE:axis_py>
                    -DINSTALL_NAME_TOOL=${CMAKE_INSTALL_NAME_TOOL}
                    -DOTOOL=${OTOOL_EXECUTABLE}
                    ${_axis_kokkos_runtime_arg}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/relocate_openmp.cmake
            COMMENT "AXIS C++: normalizing OpenMP runtime reference to @rpath"
            VERBATIM
        )
    else()
        message(STATUS "AXIS C++: otool/install_name_tool unavailable — skipping OpenMP install-name relocation")
    endif()
endif()

# Install the pure-python axis package directory
install(DIRECTORY axis DESTINATION .)

# Install the compiled nanobind binary extension in the axis package folder
install(TARGETS axis_py LIBRARY DESTINATION axis)

# Generate a type stub (.pyi) for the compiled extension at build time so that
# downstream type checkers (mypy, pyright) can resolve axis_py symbols. The stub
# is produced by importing the freshly built module, so it always matches the
# bindings. Requires nanobind >= 2.0 (provides nanobind_add_stub).
nanobind_add_stub(
    axis_py_stub
    MODULE axis_py
    OUTPUT axis_py.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:axis_py>
    DEPENDS axis_py
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/axis_py.pyi DESTINATION axis)
