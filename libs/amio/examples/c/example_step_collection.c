#define _XOPEN_SOURCE 700
#include <amio/amio_evolution.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef AMIO_EXAMPLE_HAS_MPI
#include <mpi.h>
#endif

static int fail(amio_status_t status, const char *where) {
    if (status == AMIO_OK) return 0;
    fprintf(stderr, "%s failed with AMIO status %d\n", where, (int)status);
    return 1;
}

int main(void) {
    amio_context_handle context = NULL;
    amio_dataset_handle dataset = NULL;
    amio_schema_handle schema = NULL;
    amio_step_handle step = NULL;
    amio_operation_handle operation = NULL;
    amio_step_lease_handle lease = NULL;
    char temporary_dir[] = "/tmp/amio-step-collection-XXXXXX";
    char collection[160];
    amio_status_t status = AMIO_OK;
    int result = EXIT_FAILURE;
#ifdef AMIO_EXAMPLE_HAS_MPI
    int argc = 0;
    char **argv = NULL;
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED) return EXIT_FAILURE;
#endif
    if (mkdtemp(temporary_dir) == NULL) {
        perror("mkdtemp");
        goto done;
    }
    (void)snprintf(collection, sizeof(collection), "%s/collection", temporary_dir);

    const char configuration[] = "{}";
    amio_context_descriptor_t context_desc = {0};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
    context_desc.rank_count = 1;
    context_desc.canonical_configuration = configuration;
    context_desc.configuration_bytes = sizeof(configuration) - 1;
#ifdef AMIO_EXAMPLE_HAS_MPI
    context_desc.fortran_communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_SELF);
#endif
    status = amio_init_ex(&context_desc, &context);
    if (fail(status, "amio_init_ex")) goto done;

    amio_dataset_descriptor_t dataset_desc = {0};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = collection;
    dataset_desc.location_bytes = strlen(collection);
    dataset_desc.format = AMIO_FORMAT_NETCDF4;
    dataset_desc.budget_bytes = 1024 * 1024;
    dataset_desc.budget_descriptors = 64;
    dataset_desc.control_budget_bytes = 4096;
    dataset_desc.control_budget_descriptors = 8;
    status = amio_open_dataset_ex(context, &dataset_desc, &dataset);
    if (fail(status, "amio_open_dataset_ex")) goto done;

    const char dimension_name[] = "x";
    const char field_name[] = "temperature";
    amio_dimension_descriptor_t dimension = {0};
    dimension.struct_size = sizeof(dimension);
    dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name = dimension_name;
    dimension.name_bytes = sizeof(dimension_name) - 1;
    dimension.extent = 4;
    amio_variable_descriptor_t variable = {0};
    variable.struct_size = sizeof(variable);
    variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    variable.name = field_name;
    variable.name_bytes = sizeof(field_name) - 1;
    variable.source_dtype = AMIO_DTYPE_F32;
    variable.stored_dtype = AMIO_DTYPE_F32;
    variable.required = 1;
    variable.dimension_count = 1;
    variable.dimensions = &dimension;
    amio_schema_descriptor_t schema_desc = {0};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = 1;
    schema_desc.variables = &variable;
    status = amio_define_schema(dataset, &schema_desc, &schema);
    if (fail(status, "amio_define_schema")) goto done;

    amio_step_descriptor_t step_desc = {0};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 0;
    step_desc.model_time = 12.5;
    step_desc.schema = schema;
    status = amio_begin_step(dataset, &step_desc, &step);
    if (fail(status, "amio_begin_step")) goto done;

    const float temperature[4] = {280.0f, 281.0f, 282.0f, 283.0f};
    status = amio_put_field(step, field_name, sizeof(field_name) - 1, temperature, sizeof(temperature), &operation);
    if (fail(status, "amio_put_field")) goto done;
    status = amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 0);
    if (fail(status, "wait for source release")) goto done;
    status = amio_end_step(step);
    if (fail(status, "amio_end_step")) goto done;

    amio_step_summary_t summary = {0};
    status = amio_describe_step(dataset, 0, &summary);
    if (fail(status, "amio_describe_step")) goto done;
    status = amio_acquire_step_lease(dataset, summary.step_id, &lease);
    if (fail(status, "amio_acquire_step_lease")) goto done;
    if (amio_close_dataset_ex(dataset) != AMIO_ERR_INVALID_POLICY) {
        fprintf(stderr, "dataset close must reject an outstanding step lease\n");
        goto done;
    }
    status = amio_release_step_lease(lease);
    if (fail(status, "amio_release_step_lease")) goto done;
    lease = NULL;
    printf("Published step %" PRIu64 " with %u field(s) at %s\n", summary.step_id, summary.present_field_count, collection);
    result = EXIT_SUCCESS;

done:
    if (lease != NULL) (void)amio_release_step_lease(lease);
    if (operation != NULL) (void)amio_release_operation(operation);
    if (step != NULL) (void)amio_release_step_handle(step);
    if (schema != NULL) (void)amio_release_schema(schema);
    if (dataset != NULL) (void)amio_close_dataset_ex(dataset);
    if (context != NULL) (void)amio_release_context(context);
#ifdef AMIO_EXAMPLE_HAS_MPI
    MPI_Finalize();
#endif
    return result;
}
