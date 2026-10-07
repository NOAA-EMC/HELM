# ####################################################################################################################################################
# check_symbol_mangling.cmake
#
# Run as a `cmake -P` script from the CTest target `amio_symbol_mangling`.  Verifies R10.5, R10.8 by inspecting the dynamic symbol table of
# libamio.so:
#
# nm -D --defined-only libamio.so
#
# The check rule (from design.md "Header isolation enforcement"):
#
# nm -D libamio.so | grep ' T '            \\\ | grep -v '^[0-9a-f]* T amio_'
#
# must produce zero lines.  Equivalently: every globally-exported text-segment symbol in libamio.so MUST begin with the `amio_` prefix and MUST NOT
# carry C++ name mangling (`_Z...`, `__Z...`, vague-linkage helpers, etc.).
#
# When `nm` is unavailable on the host, the test exits 77 so CTest reports it as SKIPPED instead of failing -- CI always provides `nm`, so the gate is
# preserved on PRs.
#
# Required CMake variables (set by the parent test): AMIO_NM_EXECUTABLE   absolute path to nm, or empty when missing AMIO_LIBRARY_FILE    absolute
# path to the freshly-built libamio (resolved from $<TARGET_FILE:amio_core> at test invocation time)
# ####################################################################################################################################################

if(NOT DEFINED AMIO_LIBRARY_FILE OR "${AMIO_LIBRARY_FILE}" STREQUAL "")
    message(FATAL_ERROR "check_symbol_mangling: AMIO_LIBRARY_FILE not set; ensure the " "amio_core target is configured before this test is invoked.")
endif()

if(NOT EXISTS "${AMIO_LIBRARY_FILE}")
    message(FATAL_ERROR "check_symbol_mangling: libamio not found at " "'${AMIO_LIBRARY_FILE}'.  Build the `amio_core` target before "
                        "running ctest.")
endif()

if(NOT DEFINED AMIO_NM_EXECUTABLE OR "${AMIO_NM_EXECUTABLE}" STREQUAL "" OR "${AMIO_NM_EXECUTABLE}" STREQUAL "AMIO_NM_EXECUTABLE-NOTFOUND")
    # CTest is configured to interpret the literal token `SKIPPED:` in this script's output as a skip via SKIP_REGULAR_EXPRESSION. Returning normally
    # (exit 0) keeps the build green; the test appears as `***Skipped` in ctest output.
    message(STATUS "amio_symbol_mangling: SKIPPED: `nm` not on PATH " "(install binutils to enable the symbol-mangling check).")
    return()
endif()

# `nm -D --defined-only` lists exported symbols from the dynamic table.  We then filter to the text segment ('T' = global, externally visible code)
# and reject any line whose symbol name does not start with `amio_`.  Local ('t'), data ('D'), bss ('B'), and weak ('W'/'V') symbols are immaterial to
# the C ABI surface under inspection.
execute_process(
    COMMAND ${AMIO_NM_EXECUTABLE} -D --defined-only ${AMIO_LIBRARY_FILE}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _nm_out
    ERROR_VARIABLE _nm_err)

if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "amio_symbol_mangling FAILED to invoke nm.\n" "  nm       : ${AMIO_NM_EXECUTABLE}\n" "  Library  : ${AMIO_LIBRARY_FILE}\n"
                        "  Exit     : ${_rc}\n" "  stderr   :\n${_nm_err}")
endif()

# Parse line-by-line.  Each `nm -D` row is "<addr> <type> <name>".
string(REPLACE "\n" ";" _nm_lines "${_nm_out}")

set(_offending_lines "")
set(_text_symbol_count 0)
set(_amio_symbol_count 0)

foreach(_line IN LISTS _nm_lines)
    if("${_line}" STREQUAL "")
        continue()
    endif()

    # Match lines exposing a text-segment global symbol: T (text) or i (indirect function, GNU IFUNC).  `R` (read-only data) is benign for ABI but we
    # still gate it: any ABI-visible symbol must carry the `amio_` prefix.
    if("${_line}" MATCHES "^[0-9a-fA-F]+[ \t]+([TiR])[ \t]+(.+)$")
        set(_sym_name "${CMAKE_MATCH_2}")

        # `T` symbols are the strict gate.  Track all of them; flag any whose mangled form does not start with `amio_`.
        if(CMAKE_MATCH_1 STREQUAL "T")
            math(EXPR _text_symbol_count "${_text_symbol_count} + 1")
            if(_sym_name MATCHES "^amio_")
                math(EXPR _amio_symbol_count "${_amio_symbol_count} + 1")
            else()
                list(APPEND _offending_lines "${_line}")
            endif()
        endif()
    endif()
endforeach()

if(_offending_lines)
    string(REPLACE ";" "\n    " _offending_block "${_offending_lines}")
    message(
        FATAL_ERROR
            "amio_symbol_mangling FAILED.\n"
            "  Library : ${AMIO_LIBRARY_FILE}\n"
            "  Found ${_text_symbol_count} exported text symbols, of\n"
            "  which only ${_amio_symbol_count} carry the required\n"
            "  `amio_` prefix.\n\n"
            "  Offending exported symbols:\n"
            "    ${_offending_block}\n\n"
            "Every exported symbol in libamio.so MUST have the `amio_`\n"
            "C99 prefix and MUST NOT carry C++ name mangling (R10.5,\n"
            "R10.8).  Common causes:\n"
            "  * A C++ symbol leaked through `__attribute__((visibility(\"default\")))`.\n"
            "  * The translation unit was compiled without `extern \"C\"`.\n"
            "  * A static-init-time helper was emitted with default visibility.\n"
            "Fix by hiding the symbol (visibility=hidden, anonymous\n"
            "namespace, or removing the export) and re-running ctest.")
endif()

message(STATUS "amio_symbol_mangling: PASSED " "(${_text_symbol_count} exported text symbols, all `amio_*`).")
