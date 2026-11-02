#include "diagnostics/diagnostic_store.hpp"

#include <cassert>
#include <cstdint>
#include <string>

using namespace amio::detail;

namespace {

DiagnosticScope scope(std::uint64_t id) {
    DiagnosticScope key;
    key.context_epoch = 9;
    key.dataset_id = id;
    key.step_id = id + 100;
    key.field_id = 2;
    key.role = DiagnosticRole::Producer;
    key.participant = static_cast<std::uint32_t>(id);
    return key;
}

void test_scope_limit_evicts_detail_into_cumulative_totals() {
    DiagnosticStore store({2, 512, 2, 32});
    store.add_counter(scope(1), DiagnosticCounter::Accepted, 3);
    store.observe_duration(scope(1), DiagnosticHistogram::PublicationNs, 1'000'000);
    store.set_resource(scope(1), DiagnosticResource::StagingRetainedBytes, 96, 128);
    store.add_counter(scope(2), DiagnosticCounter::Completed, 4);
    store.add_counter(scope(3), DiagnosticCounter::Failed, 5);
    auto snapshot = store.snapshot();
    assert(snapshot.scope_count == 2);
    assert(snapshot.evicted_scope_count == 1);
    assert(snapshot.cumulative[static_cast<std::size_t>(DiagnosticCounter::Accepted)] == 3);
    assert(snapshot.cumulative[static_cast<std::size_t>(DiagnosticCounter::Completed)] == 4);
    assert(snapshot.cumulative[static_cast<std::size_t>(DiagnosticCounter::Failed)] == 5);
    assert(snapshot.cumulative_histograms[static_cast<std::size_t>(DiagnosticHistogram::PublicationNs)][20] == 1);
    assert(snapshot.cumulative_resource_high_water[static_cast<std::size_t>(DiagnosticResource::StagingRetainedBytes)] == 96);
    assert(!store.scope_snapshot(scope(1)).has_value());
}

void test_first_causal_error_and_bounded_event_ring() {
    DiagnosticStore store({4, 256, 2, 24});
    store.record_error(scope(1), -17, "first cause");
    store.record_error(scope(1), -18, "later cause");
    store.record_event(scope(1), DiagnosticEventKind::Payload, "a payload transfer event");
    store.record_event(scope(1), DiagnosticEventKind::Publication, "another event");
    store.record_event(scope(1), DiagnosticEventKind::Failure, "third event must evict");
    const auto snapshot = store.snapshot();
    assert(snapshot.first_error.status == -17);
    assert(snapshot.first_error.message == "first cause");
    assert(snapshot.events.size() <= 2);
    assert(snapshot.dropped_sample_count >= 1);
    assert(snapshot.event_bytes <= 256);
    const auto event_resource = static_cast<std::size_t>(DiagnosticResource::EventHistoryBytes);
    assert(snapshot.cumulative_resource_high_water[event_resource] >= snapshot.event_bytes);
    assert(snapshot.cumulative_resource_high_water[event_resource] <= 256);
    const auto item = store.scope_snapshot(scope(1));
    assert(item.has_value());
    assert(item->resource_high_water[event_resource] >= snapshot.event_bytes);
    assert(item->resource_ceilings[event_resource] == 256);
}

void test_histograms_and_retained_resource_high_water_are_fixed_and_bounded() {
    DiagnosticStore store({1, 128, 3, 16});
    store.observe_duration(scope(1), DiagnosticHistogram::PublicationNs, 1);
    store.observe_duration(scope(1), DiagnosticHistogram::PublicationNs, 1'000'000);
    store.set_resource(scope(1), DiagnosticResource::StagingRetainedBytes, 96, 128);
    store.set_resource(scope(1), DiagnosticResource::StagingRetainedBytes, 64, 128);
    const auto item = store.scope_snapshot(scope(1));
    assert(item.has_value());
    const auto &hist = item->histograms[static_cast<std::size_t>(DiagnosticHistogram::PublicationNs)];
    assert(hist.size() == kDiagnosticHistogramBins);
    assert(item->resource_high_water[static_cast<std::size_t>(DiagnosticResource::StagingRetainedBytes)] == 96);
    assert(item->resource_ceilings[static_cast<std::size_t>(DiagnosticResource::StagingRetainedBytes)] == 128);
}

} // namespace

int main() {
    test_scope_limit_evicts_detail_into_cumulative_totals();
    test_first_causal_error_and_bounded_event_ring();
    test_histograms_and_retained_resource_high_water_are_fixed_and_bounded();
}
