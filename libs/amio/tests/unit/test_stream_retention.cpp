#include "execution/stream_retention.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace amio::detail;

namespace {
RetainedStreamStep step(std::uint64_t id, std::byte value) {
    RetainedStreamStep result;
    result.step_id = id;
    result.schema_digest = "schema-" + std::to_string(id / 2);
    result.fields.emplace("temperature", std::vector<std::byte>(4, value));
    result.payload_bytes = 4;
    return result;
}
}

int main() {
    {
        StreamRetention retained(StreamRetentionPolicy::Reliable, 2, 1, 8);
        assert(retained.publish(step(0, std::byte{1})) == AMIO_OK);
        auto c0 = retained.acquire_next(0, std::chrono::milliseconds(0));
        auto c1 = retained.acquire_next(1, std::chrono::milliseconds(0));
        assert(c0.state == StreamAcquireState::Available && c0.lease);
        assert(c1.state == StreamAcquireState::Available && c1.lease);
        assert(c0.lease->step->schema_digest == "schema-0");
        assert(retained.snapshot().bytes == 4 && retained.snapshot().leases == 2);
        assert(retained.close() == AMIO_ERR_VIEWS_OUTSTANDING);
        assert(retained.release(c0.lease->lease_id) == AMIO_OK);
        assert(retained.snapshot().bytes == 4);
        assert(retained.publish(step(1, std::byte{2})) == AMIO_ERR_QUEUE_FULL);
        assert(retained.release(c1.lease->lease_id) == AMIO_OK);
        assert(retained.snapshot().bytes == 0);
        assert(retained.publish(step(1, std::byte{2})) == AMIO_OK);
        retained.finish();
        auto done0 = retained.acquire_next(0, std::chrono::milliseconds(0));
        auto done1 = retained.acquire_next(1, std::chrono::milliseconds(0));
        assert(done0.state == StreamAcquireState::Available && done0.lease);
        assert(done1.state == StreamAcquireState::Available && done1.lease);
        assert(retained.release(done0.lease->lease_id) == AMIO_OK);
        assert(retained.release(done1.lease->lease_id) == AMIO_OK);
        assert(retained.acquire_next(0, std::chrono::milliseconds(0)).state == StreamAcquireState::EndOfStream);
        assert(retained.close() == AMIO_OK);
    }
    {
        StreamRetention retained(StreamRetentionPolicy::Latest, 2, 2, 8);
        assert(retained.publish(step(0, std::byte{1})) == AMIO_OK);
        assert(retained.publish(step(1, std::byte{2})) == AMIO_OK);
        auto c0 = retained.acquire_next(0, std::chrono::milliseconds(0));
        assert(c0.state == StreamAcquireState::Available && c0.lease->step->step_id == 0);
        const auto c0_step0_lease = c0.lease->lease_id;
        assert(retained.publish(step(2, std::byte{3})) == AMIO_OK); // evicts unleased step 1
        assert(retained.snapshot().bytes == 8 && retained.snapshot().steps == 2);
        auto c1_first = retained.acquire_next(1, std::chrono::milliseconds(0));
        assert(c1_first.state == StreamAcquireState::Available && c1_first.lease->step->step_id == 0);
        assert(retained.release(c1_first.lease->lease_id) == AMIO_OK);
        auto c1_skip = retained.acquire_next(1, std::chrono::milliseconds(0));
        assert(c1_skip.state == StreamAcquireState::Skipped && c1_skip.skipped);
        assert(c1_skip.skipped->first == 1 && c1_skip.skipped->last == 1);
        auto c0_next = retained.acquire_next(0, std::chrono::milliseconds(0));
        assert(c0_next.state == StreamAcquireState::Skipped && c0_next.skipped);
        assert(c0_next.skipped->first == 1 && c0_next.skipped->last == 1);
        assert(retained.release(c0_step0_lease) == AMIO_OK);
        c0 = retained.acquire_next(0, std::chrono::milliseconds(0));
        c1_first = retained.acquire_next(1, std::chrono::milliseconds(0));
        assert(c0.state == StreamAcquireState::Available && c0.lease->step->step_id == 2);
        assert(c1_first.state == StreamAcquireState::Available && c1_first.lease->step->step_id == 2);
        assert(retained.release(c0.lease->lease_id) == AMIO_OK);
        assert(retained.release(c1_first.lease->lease_id) == AMIO_OK);
        assert(retained.snapshot().skipped_steps == 2);
        retained.finish();
        assert(retained.close() == AMIO_OK);
    }
    {
        StreamRetention retained(StreamRetentionPolicy::Latest, 1, 1, 4);
        assert(retained.acquire_next(0, std::chrono::milliseconds(0)).state == StreamAcquireState::Timeout);
        assert(retained.publish(step(0, std::byte{1})) == AMIO_OK);
        const auto lease = retained.acquire_next(0, std::chrono::milliseconds(0)).lease;
        assert(lease);
        assert(retained.publish(step(1, std::byte{2})) == AMIO_ERR_QUEUE_FULL); // leased bytes cannot be evicted
        retained.finish(AMIO_ERR_BACKEND_FAILURE);
        assert(retained.release(lease->lease_id) == AMIO_OK);
        assert(retained.close(true) == AMIO_OK);
    }
    {
        StreamRetention retained(StreamRetentionPolicy::Reliable, 1, 1, 4, 2);
        assert(retained.publish(step(0, std::byte{1})) == AMIO_OK);
        const auto acquire = retained.acquire_next(0, std::chrono::milliseconds(0));
        assert(acquire.state == StreamAcquireState::Failed && acquire.failure == AMIO_ERR_BUDGET_EXCEEDED);
        assert(retained.snapshot().leases == 0);
        assert(retained.close(true) == AMIO_OK);
    }
    {
        StreamRetention publisher(StreamRetentionPolicy::Reliable, 1, 2, 8);
        assert(publisher.publish(step(0, std::byte{1})) == AMIO_OK);
        assert(publisher.snapshot().bytes == 4);
        assert(publisher.acknowledge(0, 0) == AMIO_OK);
        assert(publisher.snapshot().bytes == 0);
        assert(publisher.acknowledge(0, 0) == AMIO_ERR_INVALID_HANDLE);
    }
}
