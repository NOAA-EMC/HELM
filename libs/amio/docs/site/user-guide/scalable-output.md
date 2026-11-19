# Scalable output and progress

AMIO supports direct parallel NetCDF-4 output when built with parallel NetCDF.
Aggregated execution contexts can describe producer and writer rank groups, but
the public persistent dataset lifecycle is not yet connected to aggregation.
`amio_run_service` therefore returns `AMIO_ERR_UNSUPPORTED` on a writer instead
of waiting for producer work that cannot yet be submitted.

The C and Fortran examples `example_aggregated_steps` show the current role
configuration and the explicit service result. Treat them as API and topology
examples, not as a successful aggregated-output recipe. Do not publish data or
claim an aggregated storage guarantee from these examples.

With `AMIO_PROGRESS_CALLER`, call `amio_progress(context, max_callbacks,
&completed)` periodically from the thread that owns MPI. Keep `max_callbacks`
bounded so model work remains responsive. `AMIO_PROGRESS_BACKGROUND` requires
the MPI thread level documented by the context API; initialize MPI with
`MPI_THREAD_MULTIPLE` and verify the provided level before selecting it.
Progress and diagnostics queries are rank-local unless the explicit diagnostics
reduction operation is used.

The local staging budget bounds owned snapshots and retained payload capacity.
Size it for the largest local step contributions that can be live at once,
including retained free capacity and descriptors, and leave the configured
control reserve available for metadata and protocol traffic. Queue backpressure
can reject admission or wait only until its configured deadline. A wait cannot
free memory held by an unsealed step, so applications must seal/cancel steps and
drain accepted work before expecting capacity to return.

For a functioning persistent parallel workflow today, use direct mode and its
backend-specific collective requirements. Revisit this guide when the public
aggregated dataset lifecycle and independent multi-rank qualification land.
