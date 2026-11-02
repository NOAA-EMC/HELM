#include "diagnostics/diagnostic_store.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace amio::detail {
namespace {
void saturating_add(std::uint64_t &target, std::uint64_t value) noexcept {
    target = value > std::numeric_limits<std::uint64_t>::max() - target ? std::numeric_limits<std::uint64_t>::max() : target + value;
}

template <typename Enum, std::size_t N>
bool valid_index(Enum value) noexcept {
    return static_cast<std::size_t>(value) < N;
}
} // namespace

bool DiagnosticScope::operator<(const DiagnosticScope &other) const noexcept {
    return std::tie(context_epoch, dataset_id, step_id, field_id, role, participant) <
           std::tie(other.context_epoch, other.dataset_id, other.step_id, other.field_id, other.role, other.participant);
}

DiagnosticStore::DiagnosticStore(Limits limits) : limits_(limits) {
    if (limits_.max_scopes == 0 || limits_.max_events == 0 || limits_.event_bytes < sizeof(DiagnosticEvent) || limits_.max_event_message_bytes == 0) {
        throw std::invalid_argument("diagnostic limits must provide finite scope and event capacity");
    }
}

DiagnosticScopeSnapshot &DiagnosticStore::touch_scope(const DiagnosticScope &scope) {
    auto found = scopes_.find(scope);
    if (found == scopes_.end()) {
        if (scopes_.size() >= limits_.max_scopes) evict_oldest();
        found = scopes_.emplace(scope, Entry{}).first;
    }
    found->second.last_touch = ++clock_;
    return found->second.value;
}

void DiagnosticStore::evict_oldest() {
    if (scopes_.empty()) return;
    auto oldest = std::min_element(scopes_.begin(), scopes_.end(), [](const auto &left, const auto &right) {
        return left.second.last_touch < right.second.last_touch;
    });
    scopes_.erase(oldest);
    saturating_add(evicted_scope_count_, 1);
}

std::size_t DiagnosticStore::event_cost(std::size_t message_bytes) noexcept { return sizeof(DiagnosticEvent) + message_bytes; }

void DiagnosticStore::update_event_resource(const DiagnosticScope &scope) {
    const auto index = static_cast<std::size_t>(DiagnosticResource::EventHistoryBytes);
    auto &resource = touch_scope(scope);
    const auto current = static_cast<std::uint64_t>(event_bytes_);
    resource.resource_current[index] = current;
    resource.resource_high_water[index] = std::max(resource.resource_high_water[index], current);
    resource.resource_ceilings[index] = static_cast<std::uint64_t>(limits_.event_bytes);
    cumulative_resource_high_water_[index] = std::max(cumulative_resource_high_water_[index], current);
}

void DiagnosticStore::add_counter(const DiagnosticScope &scope, DiagnosticCounter counter, std::uint64_t amount) {
    if (!valid_index<DiagnosticCounter, kDiagnosticCounterCount>(counter)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto index = static_cast<std::size_t>(counter);
    saturating_add(touch_scope(scope).counters[index], amount);
    saturating_add(cumulative_[index], amount);
}

void DiagnosticStore::observe_duration(const DiagnosticScope &scope, DiagnosticHistogram histogram, std::uint64_t nanoseconds) {
    if (!valid_index<DiagnosticHistogram, kDiagnosticHistogramCount>(histogram)) return;
    const std::size_t bin = nanoseconds == 0 ? 0 : std::min<std::size_t>(std::bit_width(nanoseconds), kDiagnosticHistogramBins - 1);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto index = static_cast<std::size_t>(histogram);
    saturating_add(touch_scope(scope).histograms[index][bin], 1);
    saturating_add(cumulative_histograms_[index][bin], 1);
}

void DiagnosticStore::set_resource(const DiagnosticScope &scope, DiagnosticResource resource, std::uint64_t current, std::uint64_t ceiling) {
    if (!valid_index<DiagnosticResource, kDiagnosticResourceCount>(resource)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &snapshot = touch_scope(scope);
    const auto i = static_cast<std::size_t>(resource);
    snapshot.resource_current[i] = current;
    snapshot.resource_high_water[i] = std::max(snapshot.resource_high_water[i], current);
    snapshot.resource_ceilings[i] = ceiling;
    cumulative_resource_high_water_[i] = std::max(cumulative_resource_high_water_[i], current);
}

void DiagnosticStore::record_error(const DiagnosticScope &scope, std::int32_t status, std::string_view message) {
    if (status == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &snapshot = touch_scope(scope);
    if (snapshot.first_error.status == 0) {
        snapshot.first_error.status = status;
        snapshot.first_error.message.assign(message.substr(0, limits_.max_event_message_bytes));
    }
    if (first_error_.status == 0) {
        first_error_.status = status;
        first_error_.message.assign(message.substr(0, limits_.max_event_message_bytes));
    }
    saturating_add(snapshot.counters[static_cast<std::size_t>(DiagnosticCounter::Failed)], 1);
    saturating_add(cumulative_[static_cast<std::size_t>(DiagnosticCounter::Failed)], 1);
    const std::size_t length = std::min(message.size(), limits_.max_event_message_bytes);
    const std::size_t cost = event_cost(length);
    if (cost <= limits_.event_bytes && limits_.max_events > 0) {
        while (!events_.empty() && (events_.size() >= limits_.max_events || event_bytes_ > limits_.event_bytes - cost)) {
            event_bytes_ -= event_cost(events_.front().message.size());
            events_.pop_front();
            saturating_add(dropped_sample_count_, 1);
        }
        events_.push_back({scope, DiagnosticEventKind::Failure, std::string(message.substr(0, length))});
        event_bytes_ += cost;
    } else {
        saturating_add(dropped_sample_count_, 1);
    }
    update_event_resource(scope);
}

void DiagnosticStore::record_event(const DiagnosticScope &scope, DiagnosticEventKind kind, std::string_view message) {
    std::lock_guard<std::mutex> lock(mutex_);
    (void)touch_scope(scope);
    const std::size_t length = std::min(message.size(), limits_.max_event_message_bytes);
    const std::size_t cost = event_cost(length);
    if (cost > limits_.event_bytes) {
        saturating_add(dropped_sample_count_, 1);
        return;
    }
    while (!events_.empty() && (events_.size() >= limits_.max_events || event_bytes_ > limits_.event_bytes - cost)) {
        event_bytes_ -= event_cost(events_.front().message.size());
        events_.pop_front();
        saturating_add(dropped_sample_count_, 1);
    }
    events_.push_back({scope, kind, std::string(message.substr(0, length))});
    event_bytes_ += cost;
    update_event_resource(scope);
}

DiagnosticSnapshot DiagnosticStore::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DiagnosticSnapshot result;
    result.scope_count = scopes_.size();
    result.evicted_scope_count = evicted_scope_count_;
    result.dropped_sample_count = dropped_sample_count_;
    result.event_bytes = event_bytes_;
    result.cumulative = cumulative_;
    result.cumulative_histograms = cumulative_histograms_;
    result.cumulative_resource_high_water = cumulative_resource_high_water_;
    result.first_error = first_error_;
    for (const auto &[scope, entry] : scopes_) result.scopes.emplace(scope, entry.value);
    result.events = events_;
    return result;
}

std::optional<DiagnosticScopeSnapshot> DiagnosticStore::scope_snapshot(const DiagnosticScope &scope) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = scopes_.find(scope);
    if (found == scopes_.end()) return std::nullopt;
    return found->second.value;
}

} // namespace amio::detail
