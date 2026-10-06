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
 * A public function reference requires the linker to resolve an AMIO
 * symbol. The driver builds this executable but need not run it.
 */

#include <amio/amio.h>

int main(void) {
    /* The opaque handle checks the C99 header surface. Unlike the
     * handle typedef alone, amio_strerror also checks symbol linkage. */
    amio_core_handle h = (amio_core_handle)0;
    return (h == 0 && amio_strerror(AMIO_OK) != (const char *)0) ? 0 : 1;
}
