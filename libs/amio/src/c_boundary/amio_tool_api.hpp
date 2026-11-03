#pragma once
#include "amio/amio_evolution.h"
#ifdef __cplusplus
extern "C" {
#endif
/** Internal utility entry point: attach a read-only handle to an existing POSIX step collection. */
AMIO_API amio_status_t amio_open_dataset_reader(amio_context_handle context, const char *location, size_t location_bytes,
                                                amio_dataset_handle *out_dataset);
AMIO_API amio_status_t amio_get_step_model_time(amio_dataset_handle dataset, uint64_t step_id, double *out_model_time);
#ifdef __cplusplus
}
#endif
