/*
 * consumer.c
 *
 * Compiles a translation unit that includes only the AMIO public
 * header.  Verifies (statically, at consumer build time) that:
 *
 *   * `<amio/amio.h>` is reachable through the imported target's
 *     INTERFACE_INCLUDE_DIRECTORIES with no manual -I flags
 *     (R13.1 / R13.6).
 *   * The header compiles under C99 (R10.1, R10.2).
 *   * The opaque `amio_core_handle` typedef is visible.
 *
 * No call into AMIO is made -- the goal is to exercise the
 * find_package + imported target wiring, not the runtime behaviour.
 */

#include <amio/amio.h>

int main(void) {
    /* Reference the declared opaque handle so the linker is forced
     * to resolve against libamio.so's exported symbol set, which is
     * what R13.2 ultimately turns into a "no Rust/Cargo/Go in the
     * dependency closure" assertion.
     */
    amio_core_handle h = (amio_core_handle)0;
    return (h == 0) ? 0 : 1;
}
