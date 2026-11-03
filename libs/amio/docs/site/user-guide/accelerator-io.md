# Accelerator memory submission

AMIO currently accepts owned snapshots from contiguous host buffers. A caller
must finish all writes to the buffer before submission; after a successful
`amio_put_field` or `amio_put_field_memory` return, AMIO owns its snapshot and
the caller may immediately reuse the source buffer.

The versioned `amio_memory_descriptor_t` interface makes the memory kind,
layout, ownership, element size and producer synchronization explicit.
Contiguous host memory is supported. Strided layouts, borrowed ownership,
unknown kinds and device pointers are rejected before AMIO reads the payload.
Device submission is not enabled by finding Kokkos alone. The current build
has Kokkos OPENMP and SERIAL host spaces but no accelerator execution space, so
device capability remains `unsupported` and no hardware device source-release
guarantee is advertised.

The optional Kokkos package is discovered privately and does not add Kokkos
headers to the installed C or Fortran interfaces. With a supported accelerator
Kokkos backend, AMIO reserves step capacity, copies into AMIO-owned host memory,
and fences before successful admission. Hardware-specific qualification is
still required before a device-enabled build can advertise support. Borrowed
buffers and direct device-to-storage transfers remain unsupported.
