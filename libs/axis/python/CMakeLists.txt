# Python bindings for AXIS via nanobind
cmake_minimum_required(VERSION 3.21)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core _core.cpp)

target_link_libraries(_core PRIVATE axis)
if(AXIS_HAVE_NETCDF)
    target_link_libraries(_core PRIVATE axis_io)
endif()
target_include_directories(_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/ArborX/src
)

# Install the pure-python axis package directory
install(DIRECTORY axis DESTINATION .)

# Install the compiled nanobind binary extension in the axis package folder
install(TARGETS _core LIBRARY DESTINATION axis)

# Generate a type stub (.pyi) for the compiled extension at build time so that
# downstream type checkers (mypy, pyright) can resolve _core symbols. The stub
# is produced by importing the freshly built module, so it always matches the
# bindings. Requires nanobind >= 2.0 (provides nanobind_add_stub).
nanobind_add_stub(
    _core_stub
    MODULE _core
    OUTPUT _core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_core>
    DEPENDS _core
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi DESTINATION axis)
