#include <stddef.h>
#include <stdint.h>

#include "amio/amio_evolution.h"

/* The caller synchronizes producer writes before this call. On success AMIO
 * owns a snapshot and the source may immediately be reused. Device submission
 * requires a qualified private runtime adapter; this profile rejects it. */
amio_status_t amio_example_submit_host_snapshot(amio_step_handle step, const char *field, const float *values, uint64_t count,
                                                amio_operation_handle *operation) {
    amio_memory_descriptor_t memory = {0};
    memory.struct_size = (uint32_t)sizeof(memory);
    memory.version = 1;
    memory.memory_kind = 0; /* host */
    memory.ownership = 0;   /* copy into AMIO-owned storage */
    memory.rank = 1;
    memory.producer_writes_synchronized = 1;
    memory.element_bytes = sizeof(*values);
    memory.extents[0] = count;
    memory.strides[0] = sizeof(*values);
    return amio_put_field_memory(step, field, 5, &memory, values, (size_t)(count * sizeof(*values)), operation);
}

amio_status_t amio_example_submit_device_snapshot(amio_step_handle step, const char *field, const void *device_values, uint64_t count,
                                                  amio_operation_handle *operation) {
    amio_memory_descriptor_t memory = {0};
    memory.struct_size = (uint32_t)sizeof(memory);
    memory.version = 1;
    memory.memory_kind = 1; /* device */
    memory.ownership = 0;
    memory.device_id = 0;
    memory.rank = 1;
    memory.producer_writes_synchronized = 1; /* caller must synchronize its producer */
    memory.element_bytes = sizeof(float);
    memory.extents[0] = count;
    memory.strides[0] = sizeof(float);
    return amio_put_field_memory(step, field, 5, &memory, device_values, (size_t)(count * sizeof(float)), operation);
}
