# Live streaming

AMIO's stream mode connects statically configured producer and consumer ranks inside one MPI job. It does not open a persistent dataset, attach external processes, or make a live stream and a persistent collection atomically visible together.

Set `execution.mode` to `mpi_stream`, configure producer and consumer rank lists, then initialize every rank with the same context descriptor and canonical configuration. `stream.retention_policy` selects `reliable` (the default) or `latest`. The C and Fortran interfaces expose producer submission, consumer receive/acquire/read/release, release acknowledgment, and terminal status operations. See [`example_stream.c`](../../../examples/c/example_stream.c) and [`example_stream.f90`](../../../examples/fortran/example_stream.f90).

The receive descriptor supplies the expected step id, schema digest, and number of field contributions per producer. AMIO validates those values and the static producer-to-consumer mapping before accepting payload allocation. Each field carries its shape and global origin. A consumer can then read a bounded region from the lease without copying the full field into a separate AMIO buffer. Optional absence is explicit in the receive descriptor.

Reliable mode retains each producer's owned snapshot until the mapped consumer releases the step. If a release wait times out, it returns `AMIO_ERR_OPERATION_PENDING`; the retained ownership remains and the wait can be retried. Consumer leases pin their step until release. The finite stream receive budget is 64 MiB, the retained queue is limited to eight steps, and each MPI fragment is limited to 1 MiB. Capacity exhaustion returns a bounded backpressure error.

Latest mode bounds the queue to the same limits and evicts only completed, unleased steps. Acquisition reports the exact skipped ID range; it does not retain an unbounded list of skipped IDs. A lease still pins its payload. Both modes distinguish timeout, normal EOS, cancellation failure, and graceful producer disconnect. EOS and cancellation are sent only after reliable-mode release obligations drain.

The supported transport is same-job MPI with a static role map. The implementation does not promise recovery after a rank disappears or after MPI itself fails. A graceful disconnect is reported as `AMIO_ERR_STREAM_DISCONNECTED`; cancellation is reported as a failed terminal stream. Stream data has no persistent-storage visibility or durability guarantee.

Run the qualification matrix in cece-dev or an MPI-equipped build:

```sh
mpiexec -n 6 build/amio/bin/amio_qualify stream --policy reliable
mpiexec -n 6 build/amio/bin/amio_qualify stream --policy latest
mpiexec -n 6 build/amio/bin/amio_qualify stream --policy reliable --slow-consumer-ms 1
ctest --test-dir build/amio -L amio_stream --output-on-failure
```

The qualification runs four producers and two consumers for 1,000 steps and compares values and schema identity. CTest also runs retention/read-queue unit tests, cancellation/EOS/disconnect protocol cases, and C and Fortran API examples.
