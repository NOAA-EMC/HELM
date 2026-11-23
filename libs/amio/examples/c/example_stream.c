#include <amio/amio_evolution.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Six ranks: four producers send one field, two consumers read/release their mapped streams. */
int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 2;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 6) {
        MPI_Finalize();
        return 2;
    }

    int32_t producers[4] = {0, 1, 2, 3};
    int32_t consumers[2] = {4, 5};
    static const char configuration[] = "diagnostics:\n  enabled: false\n";
    amio_context_descriptor_t context_descriptor;
    memset(&context_descriptor, 0, sizeof(context_descriptor));
    context_descriptor.struct_size = sizeof(context_descriptor);
    context_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_descriptor.execution_mode = AMIO_EXECUTION_MPI_STREAM;
    context_descriptor.progress_mode = AMIO_PROGRESS_CALLER;
    context_descriptor.rank_count = 6;
    context_descriptor.producer_count = 4;
    context_descriptor.consumer_count = 2;
    context_descriptor.producers = producers;
    context_descriptor.consumers = consumers;
    context_descriptor.fortran_communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
    context_descriptor.canonical_configuration = configuration;
    context_descriptor.configuration_bytes = sizeof(configuration) - 1;
    amio_context_handle context = NULL;
    amio_status_t status = amio_init_ex(&context_descriptor, &context);

    uint8_t digest[32];
    memset(digest, 0x6c, sizeof(digest));
    if (status == AMIO_OK && rank < 4) {
        float value = (float)(5000 + rank);
        amio_stream_contribution_t contribution;
        memset(&contribution, 0, sizeof(contribution));
        contribution.struct_size = sizeof(contribution);
        contribution.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        contribution.step_id = 1;
        contribution.sequence = 1;
        contribution.field_id = 7;
        contribution.rank = 1;
        contribution.element_bytes = sizeof(value);
        memcpy(contribution.schema_digest, digest, sizeof(digest));
        contribution.offsets[0] = (uint64_t)rank;
        contribution.extents[0] = 1;
        status = amio_stream_submit(context, &contribution, &value, sizeof(value));
        if (status == AMIO_OK) status = amio_stream_wait_release(context, 1);
        if (status == AMIO_OK) status = amio_stream_finish(context, 1);
    } else if (status == AMIO_OK) {
        amio_stream_acquire_result_t timed;
        memset(&timed, 0, sizeof(timed));
        status = amio_stream_acquire(context, 0, &timed);
        if (status == AMIO_OK && timed.state != AMIO_STREAM_TIMEOUT) status = AMIO_ERR_STEP_FAILED;
        amio_stream_receive_descriptor_t receive;
        memset(&receive, 0, sizeof(receive));
        receive.struct_size = sizeof(receive);
        receive.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        receive.step_id = 1;
        receive.fields_per_producer = 1;
        memcpy(receive.schema_digest, digest, sizeof(digest));
        if (status == AMIO_OK) status = amio_stream_receive_step(context, &receive);
        amio_stream_acquire_result_t acquired;
        memset(&acquired, 0, sizeof(acquired));
        if (status == AMIO_OK) status = amio_stream_acquire(context, 0, &acquired);
        if (status == AMIO_OK && acquired.state != AMIO_STREAM_AVAILABLE) status = AMIO_ERR_STEP_FAILED;
        for (int producer = 0; status == AMIO_OK && producer < 4; ++producer) {
            if (4 + producer % 2 != rank) continue;
            amio_stream_read_descriptor_t selection;
            memset(&selection, 0, sizeof(selection));
            selection.struct_size = sizeof(selection);
            selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            selection.producer_rank = (uint32_t)producer;
            selection.field_id = 7;
            selection.rank = 1;
            selection.element_bytes = sizeof(float);
            selection.offsets[0] = (uint64_t)producer;
            selection.extents[0] = 1;
            float value = 0;
            size_t bytes = 0;
            status = amio_stream_read(acquired.lease, &selection, &value, sizeof(value), &bytes);
            if (status == AMIO_OK && (bytes != sizeof(value) || value != (float)(5000 + producer))) status = AMIO_ERR_SCHEMA_MISMATCH;
        }
        if (status == AMIO_OK) status = amio_stream_release(acquired.lease);
        if (status == AMIO_OK) status = amio_stream_finish(context, 1);
        if (status == AMIO_OK) {
            memset(&acquired, 0, sizeof(acquired));
            status = amio_stream_acquire(context, 0, &acquired);
            if (status == AMIO_OK && acquired.state != AMIO_STREAM_END_OF_STREAM) status = AMIO_ERR_STEP_FAILED;
        }
    }
    if (status != AMIO_OK) fprintf(stderr, "rank %d stream example failed: %s (%d)\n", rank, amio_strerror(status), status);
    if (context != NULL) {
        const amio_status_t close_status = amio_release_context(context);
        if (status == AMIO_OK) status = close_status;
    }
    MPI_Finalize();
    return status == AMIO_OK ? 0 : 1;
}
