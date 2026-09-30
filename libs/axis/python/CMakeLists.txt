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

# On macOS, make the OpenMP runtime resolve against RPATHs at runtime instead of
# a build-host-specific absolute path. This is done generically (no hardcoded
# library locations) so the extension is portable across machines and avoids the
# duplicate OpenMP runtime crash (OMP: Error #15) when the host process (NumPy,
# SciPy, PyTorch, ...) has already loaded its own libomp.
if(APPLE)
    # ── RPATH search order (all discovered, none hardcoded) ───────────────────
    #   1. @loader_path            — libomp co-installed next to the extension
    #   2. active Python env /lib  — where NumPy/SciPy/PyTorch ship their libomp
    #   3. the libomp CMake found  — the OpenMP install used at configure time
    set(_axis_py_rpaths "@loader_path")

    # Active Python environment prefix. Works even inside PEP 517 build isolation.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE ACTIVE_PYTHON_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(ACTIVE_PYTHON_PREFIX)
        message(STATUS "AXIS C++: prioritizing active Python environment lib in RPATH: ${ACTIVE_PYTHON_PREFIX}/lib")
        list(APPEND _axis_py_rpaths "${ACTIVE_PYTHON_PREFIX}/lib")
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
        BUILD_WITH_INSTALL_RPATH ON
    )

    # Rewrite the libomp install-name recorded in the binary to @rpath/... by
    # reading the ACTUAL dependency from the built file (see relocate_openmp.cmake),
    # rather than enumerating known absolute paths.
    find_program(OTOOL_EXECUTABLE otool)
    if(OTOOL_EXECUTABLE AND CMAKE_INSTALL_NAME_TOOL)
        add_custom_command(TARGET axis_py POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DTARGET_FILE=$<TARGET_FILE:axis_py>
                    -DINSTALL_NAME_TOOL=${CMAKE_INSTALL_NAME_TOOL}
                    -DOTOOL=${OTOOL_EXECUTABLE}
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
