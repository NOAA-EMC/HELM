#include <amio/amio_evolution.h>

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Run with at least two ranks: rank 0 is a producer, the final rank a writer. */
int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 2;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { MPI_Finalize(); return 2; }

    int32_t *producers = calloc((size_t)(size - 1), sizeof(*producers));
    int32_t writer = size - 1;
    static const char configuration[] = "{}";
    if (producers == NULL) { MPI_Finalize(); return 2; }
    for (int r = 0; r < size - 1; ++r) producers[r] = r;
    amio_context_descriptor_t descriptor = {0};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    descriptor.execution_mode = AMIO_EXECUTION_PERSISTENT_AGGREGATED;
    descriptor.progress_mode = AMIO_PROGRESS_CALLER;
    descriptor.rank_count = (uint32_t)size;
    descriptor.producer_count = (uint32_t)(size - 1);
    descriptor.writer_count = 1;
    descriptor.producers = producers;
    descriptor.writers = &writer;
    descriptor.fortran_communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
    descriptor.canonical_configuration = configuration;
    descriptor.configuration_bytes = sizeof(configuration) - 1;
    amio_context_handle context = NULL;
    amio_status_t status = amio_init_ex(&descriptor, &context);
    if (status == AMIO_OK && rank == writer) {
        /* The service entry point is wired, but the persistent aggregated
           dataset lifecycle is not connected yet; fail explicitly and cleanly. */
        status = amio_run_service(context);
        if (status != AMIO_ERR_UNSUPPORTED) fprintf(stderr, "expected unsupported service status, got %d\n", status);
        else fprintf(stderr, "writer service correctly reports unsupported until aggregation lifecycle integration\n");
    } else if (status == AMIO_OK && rank < writer) {
        uint32_t completed = 0;
        status = amio_progress(context, 8, &completed);
        if (status != AMIO_OK) fprintf(stderr, "producer progress failed: %d\n", status);
    }
    if (context) (void)amio_release_context(context);
    free(producers);
    MPI_Finalize();
    return status == AMIO_OK || (rank == writer && status == AMIO_ERR_UNSUPPORTED) ? 0 : 1;
}
