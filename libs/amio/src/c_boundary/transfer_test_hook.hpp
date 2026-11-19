#pragma once

// Private test-build injection point. Install or remove only when no transfer
// can be entering the hook; the callback and context must outlive its use.
#include "amio/amio_evolution.h"

#ifdef AMIO_TEST_TRANSFER_CONTROL
extern "C" {
typedef amio_status_t (*amio_test_transfer_hook)(void *context);
AMIO_API void amio_set_transfer_test_hook(amio_test_transfer_hook hook, void *context);
}
#endif
