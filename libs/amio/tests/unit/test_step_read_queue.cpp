#include "prefetch/step_read_queue.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace amio::detail;

int main() {
    StepReadQueue queue(16);
    StepReadKey key{"dataset-a", 8, "schema-a", "temperature", std::nullopt};
    StepReadMetadata metadata{"dataset-a", 8, "schema-a", "temperature", true, 8, {8}};
    const std::array<std::uint64_t, 1> values{0x123456789abcdef0ULL};
    assert(queue.store(key, metadata, values.data(), sizeof(values)) == AMIO_OK);
    assert(queue.bytes_used() == sizeof(values));
    std::vector<std::byte> output;
    assert(queue.load(key, metadata, output) == AMIO_OK);
    assert(output.size() == sizeof(values));

    // Dataset UUID and schema digest are part of cache identity, not merely
    // validation metadata. A stale entry for another logical dataset/schema
    // must not satisfy this lookup.
    auto other_dataset = key;
    other_dataset.dataset_uuid = "dataset-b";
    auto other_dataset_meta = metadata;
    other_dataset_meta.dataset_uuid = "dataset-b";
    assert(queue.load(other_dataset, other_dataset_meta, output) == AMIO_ERR_BACKEND_FAILURE);
    auto other_schema = key;
    other_schema.schema_digest = "schema-b";
    auto other_schema_meta = metadata;
    other_schema_meta.schema_digest = "schema-b";
    assert(queue.load(other_schema, other_schema_meta, output) == AMIO_ERR_BACKEND_FAILURE);

    auto other_step = key;
    other_step.step_id++;
    auto other_metadata = metadata;
    other_metadata.step_id++;
    assert(queue.load(other_step, other_metadata, output) == AMIO_ERR_BACKEND_FAILURE);

    auto absent = metadata;
    absent.field_present = false;
    assert(queue.load(key, absent, output) == AMIO_ERR_FIELD_NOT_PRESENT);

    auto selected = key;
    amio_bbox_t box{};
    box.rank = 1;
    box.offsets[0] = 0;
    box.extents[0] = 2;
    box.strides[0] = 1;
    selected.selection = box;
    assert(queue.load(selected, metadata, output) == AMIO_ERR_BACKEND_FAILURE);

    auto invalid_selection = selected;
    invalid_selection.selection->offsets[0] = 7;
    invalid_selection.selection->extents[0] = 2;
    assert(queue.load(invalid_selection, metadata, output) == AMIO_ERR_SCHEMA_MISMATCH);
    invalid_selection.selection = box;
    invalid_selection.selection->strides[0] = INT64_MAX;
    assert(queue.store(invalid_selection, metadata, values.data(), sizeof(values)) == AMIO_ERR_SCHEMA_MISMATCH);

    auto wrong_field = metadata;
    wrong_field.field = "pressure";
    assert(queue.store(key, wrong_field, values.data(), sizeof(values)) == AMIO_ERR_SCHEMA_MISMATCH);

    StepReadKey oversized{"dataset-a", 9, "schema-a", "large", std::nullopt};
    StepReadMetadata large_meta{"dataset-a", 9, "schema-a", "large", true, 17, {17}};
    std::array<std::uint8_t, 17> large{};
    assert(queue.store(oversized, large_meta, large.data(), large.size()) == AMIO_ERR_BUDGET_EXCEEDED);
    assert(queue.entry_count() == 1);
    queue.clear();
    assert(queue.bytes_used() == 0 && queue.entry_count() == 0);

    // A hit refreshes recency, so insertion evicts the least recently used
    // value while keeping the byte ceiling exact.
    StepReadQueue lru(16);
    StepReadKey first{"dataset-a", 1, "schema-a", "a", std::nullopt};
    StepReadMetadata first_meta{"dataset-a", 1, "schema-a", "a", true, 8, {8}};
    StepReadKey second{"dataset-a", 2, "schema-a", "b", std::nullopt};
    StepReadMetadata second_meta{"dataset-a", 2, "schema-a", "b", true, 8, {8}};
    StepReadKey third{"dataset-a", 3, "schema-a", "c", std::nullopt};
    StepReadMetadata third_meta{"dataset-a", 3, "schema-a", "c", true, 8, {8}};
    assert(lru.store(first, first_meta, values.data(), sizeof(values)) == AMIO_OK);
    assert(lru.store(second, second_meta, values.data(), sizeof(values)) == AMIO_OK);
    assert(lru.load(first, first_meta, output) == AMIO_OK);
    assert(lru.store(third, third_meta, values.data(), sizeof(values)) == AMIO_OK);
    assert(lru.bytes_used() == 16 && lru.entry_count() == 2);
    assert(lru.load(first, first_meta, output) == AMIO_OK);
    assert(lru.load(second, second_meta, output) == AMIO_ERR_BACKEND_FAILURE);
}
