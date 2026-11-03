add_executable(
    amio
    amio.cpp
    result_writer.cpp
    result_helpers.cpp
    selection_parse.cpp
    inspect.cpp
    validate.cpp
    conversion_plan.cpp
    copy.cpp
    diagnostics.cpp
    ../src/catalog/catalog_format.cpp
    ../src/catalog/collection_reader.cpp
    ../src/schema/schema.cpp)
target_compile_features(amio PRIVATE cxx_std_20)
target_include_directories(amio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(amio PRIVATE AMIO::amio_core)
if(MPI_CXX_FOUND)
    target_compile_definitions(amio PRIVATE AMIO_TOOLS_HAS_MPI=1)
    target_link_libraries(amio PRIVATE MPI::MPI_CXX)
endif()
install(TARGETS amio RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(AMIO_BUILD_TESTING)
    add_test(NAME integration.cli_capabilities COMMAND amio capabilities --json)
    add_test(NAME integration.cli_invalid_inspect COMMAND amio inspect /path/that/does/not/exist --json)
    set_tests_properties(integration.cli_invalid_inspect PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(integration.cli_capabilities integration.cli_invalid_inspect PROPERTIES LABELS "integration;utilities" TIMEOUT 30)
endif()
