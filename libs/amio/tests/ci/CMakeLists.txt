# ####################################################################################################################################################
# tests/ci/CMakeLists.txt
#
# CI gate tests that verify the eckit removal is complete and no eckit symbols leak into the shared library.
#
# ci.no_eckit_symbols  -- R12.8: `nm -D libamio.so | c++filt` must contain zero `eckit::` symbols.
#
# ci.no_eckit_recipe   -- R12.1: no eckit dependency declaration in the spack recipe or core CMake.
# ####################################################################################################################################################

add_test(NAME ci.no_eckit_symbols COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_eckit_symbols.sh $<TARGET_FILE:amio_core>)
set_tests_properties(ci.no_eckit_symbols PROPERTIES LABELS "ci;regression" TIMEOUT 30 SKIP_REGULAR_EXPRESSION "SKIP:")

add_test(NAME ci.no_eckit_recipe COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_eckit_recipe.sh)
set_tests_properties(ci.no_eckit_recipe PROPERTIES LABELS "ci;regression" TIMEOUT 30)
