#include "catalog/tensorstore_kvstore.hpp"

// Keep this translation unit in the feature-enabled AMIO core target so CI
// compiles the concrete adapter whenever TensorStore is available. The class
// implementation is header-only to keep TensorStore types private to AMIO.
