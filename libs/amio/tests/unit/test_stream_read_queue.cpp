#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "prefetch/stream_read_queue.hpp"

using namespace amio::detail;

int main() {
    StreamRetention retention(StreamRetentionPolicy::Reliable, 1, 2, 1024);
    RetainedStreamStep step;
    step.step_id = 7;
    step.schema_digest = "schema-v2";
    auto &bytes = step.fields["temperature"];
    const std::array<std::uint32_t, 12> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bytes.resize(sizeof(values));
    std::memcpy(bytes.data(), values.data(), sizeof(values));
    step.payload_bytes = bytes.size();
    step.field_shapes["temperature"] = {3, 4};
    step.field_offsets["temperature"] = {10, 20};
    step.absent_optional_fields.insert("rain");
    assert(retention.publish(std::move(step)) == AMIO_OK);

    StreamReadQueue queue(retention, 0);
    const auto acquired = queue.acquire(std::chrono::milliseconds(0));
    assert(acquired.state == StreamAcquireState::Available && acquired.lease);
    const RegionSelection selection{{11, 21}, {2, 2}, false};
    std::array<std::uint32_t, 4> selected{};
    std::size_t written = 0;
    assert(queue.read(*acquired.lease, "temperature", selection, sizeof(std::uint32_t), std::as_writable_bytes(std::span(selected)), written) ==
           AMIO_OK);
    assert(written == sizeof(selected));
    assert((selected == std::array<std::uint32_t, 4>{5, 6, 9, 10}));

    std::array<std::byte, 1> short_buffer{};
    assert(queue.read(*acquired.lease, "temperature", selection, sizeof(std::uint32_t), short_buffer, written) == AMIO_ERR_BUFFER_TOO_SMALL);
    assert(written == 0);
    const RegionSelection outside{{9, 20}, {1, 1}, false};
    assert(queue.read(*acquired.lease, "temperature", outside, sizeof(std::uint32_t), short_buffer, written) == AMIO_ERR_PARTITION_INVALID);
    const RegionSelection wrong_rank{{10}, {1}, false};
    assert(queue.read(*acquired.lease, "temperature", wrong_rank, sizeof(std::uint32_t), short_buffer, written) == AMIO_ERR_INVALID_INPUT);
    assert(queue.read(*acquired.lease, "rain", selection, sizeof(std::uint32_t), short_buffer, written) == AMIO_ERR_FIELD_NOT_PRESENT);
    assert(queue.read(*acquired.lease, "field_99_producer_0", selection, sizeof(std::uint32_t), short_buffer, written, "rain") ==
           AMIO_ERR_FIELD_NOT_PRESENT);
    assert(queue.release(*acquired.lease) == AMIO_OK);
    assert(queue.read(*acquired.lease, "temperature", selection, sizeof(std::uint32_t), std::as_writable_bytes(std::span(selected)), written) ==
           AMIO_ERR_INVALID_HANDLE);
    return 0;
}
