#ifndef AMIO_SRC_DIAGNOSTICS_DIAGNOSTIC_STORE_HPP
#define AMIO_SRC_DIAGNOSTICS_DIAGNOSTIC_STORE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace amio::detail {

inline constexpr std::size_t kDiagnosticCounterCount = 17;
inline constexpr std::size_t kDiagnosticHistogramCount = 9;
inline constexpr std::size_t kDiagnosticHistogramBins = 32;
inline constexpr std::size_t kDiagnosticResourceCount = 12;

enum class DiagnosticCounter : std::size_t {
    Accepted,
    Rejected,
    Completed,
    Failed,
    BytesAccepted,
    BytesWritten,
    BytesPublished,
    SourceReleasedBytes,
    TransportReceivedBytes,
    TransportPublishedBytes,
    ViewBytes,
    StreamSkips,
    StreamDeliveredBytes,
    StreamReleasedBytes,
    StreamDisconnects,
    OperatorInputBytes,
    OperatorOutputBytes
};
enum class DiagnosticHistogram : std::size_t {
    AdmissionNs,
    SnapshotNs,
    OperatorNs,
    TransportNs,
    BackendNs,
    PublicationNs,
    EndToEndNs,
    WaitNs,
    SourceReleaseNs
};
enum class DiagnosticResource : std::size_t {
    QueueDescriptors,
    StagingRetainedBytes,
    StagingDescriptors,
    TransportBytes,
    TransportDescriptors,
    MetadataBytes,
    PrefetchBytes,
    ViewBytes,
    OperatorScratchBytes,
    ExceptionHistoryBytes,
    EventHistoryBytes,
    ActiveOperations
};
enum class DiagnosticRole : std::uint8_t { Unknown, Producer, Writer, Consumer, Service };
enum class DiagnosticEventKind : std::uint8_t { Admission, Snapshot, Payload, Publication, Failure, Progress, Resource };

struct DiagnosticScope {
    std::uint64_t context_epoch = 0;
    std::uint64_t dataset_id = 0;
    std::uint64_t step_id = 0;
    std::uint32_t field_id = 0;
    DiagnosticRole role = DiagnosticRole::Unknown;
    std::uint32_t participant = 0;
    bool operator<(const DiagnosticScope &other) const noexcept;
};

struct DiagnosticError {
    std::int32_t status = 0;
    std::string message;
};
struct DiagnosticEvent {
    DiagnosticScope scope;
    DiagnosticEventKind kind = DiagnosticEventKind::Progress;
    std::string message;
};

struct DiagnosticScopeSnapshot {
    std::array<std::uint64_t, kDiagnosticCounterCount> counters{};
    std::array<std::array<std::uint64_t, kDiagnosticHistogramBins>, kDiagnosticHistogramCount> histograms{};
    std::array<std::uint64_t, kDiagnosticResourceCount> resource_current{};
    std::array<std::uint64_t, kDiagnosticResourceCount> resource_high_water{};
    std::array<std::uint64_t, kDiagnosticResourceCount> resource_ceilings{};
    DiagnosticError first_error;
};

struct DiagnosticSnapshot {
    std::size_t scope_count = 0;
    std::uint64_t evicted_scope_count = 0;
    std::uint64_t dropped_sample_count = 0;
    std::size_t event_bytes = 0;
    std::array<std::uint64_t, kDiagnosticCounterCount> cumulative{};
    /** Histograms across all scopes, including scopes whose detail was evicted. */
    std::array<std::array<std::uint64_t, kDiagnosticHistogramBins>, kDiagnosticHistogramCount> cumulative_histograms{};
    /** Peak resource values across all scopes, including evicted scopes. */
    std::array<std::uint64_t, kDiagnosticResourceCount> cumulative_resource_high_water{};
    DiagnosticError first_error;
    std::map<DiagnosticScope, DiagnosticScopeSnapshot> scopes;
    std::deque<DiagnosticEvent> events;
};

class DiagnosticStore {
   public:
    /** Hard limits for retained scopes, event bytes, event count, and message size. */
    struct Limits {
        std::size_t max_scopes;
        std::size_t event_bytes;
        std::size_t max_events;
        std::size_t max_event_message_bytes;
    };
    /** Create a bounded local diagnostics store; every limit must be finite and positive. */
    explicit DiagnosticStore(Limits limits);
    /** Add a counter sample to its retained scope and lifetime cumulative totals. */
    void add_counter(const DiagnosticScope &scope, DiagnosticCounter counter, std::uint64_t amount = 1);
    /** Record a duration in the fixed histogram for the selected metric. */
    void observe_duration(const DiagnosticScope &scope, DiagnosticHistogram histogram, std::uint64_t nanoseconds);
    /** Update current and peak resource usage and its declared ceiling. */
    void set_resource(const DiagnosticScope &scope, DiagnosticResource resource, std::uint64_t current, std::uint64_t ceiling);
    /** Retain the first causal error for a scope and globally, within bounded storage. */
    void record_error(const DiagnosticScope &scope, std::int32_t status, std::string_view message);
    /** Append a bounded event sample, accounting for dropped samples on eviction. */
    void record_event(const DiagnosticScope &scope, DiagnosticEventKind kind, std::string_view message);
    /** Copy a bounded local snapshot; this operation performs no MPI collective. */
    [[nodiscard]] DiagnosticSnapshot snapshot() const;
    /** Return one retained scope snapshot, or empty if it was never retained/was evicted. */
    [[nodiscard]] std::optional<DiagnosticScopeSnapshot> scope_snapshot(const DiagnosticScope &scope) const;

   private:
    struct Entry {
        DiagnosticScopeSnapshot value;
        std::uint64_t last_touch = 0;
    };
    DiagnosticScopeSnapshot &touch_scope(const DiagnosticScope &scope);
    void evict_oldest();
    void update_event_resource(const DiagnosticScope &scope);
    static std::size_t event_cost(std::size_t message_bytes) noexcept;
    Limits limits_;
    mutable std::mutex mutex_;
    std::map<DiagnosticScope, Entry> scopes_;
    std::deque<DiagnosticEvent> events_;
    std::size_t event_bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t evicted_scope_count_ = 0;
    std::uint64_t dropped_sample_count_ = 0;
    std::array<std::uint64_t, kDiagnosticCounterCount> cumulative_{};
    std::array<std::array<std::uint64_t, kDiagnosticHistogramBins>, kDiagnosticHistogramCount> cumulative_histograms_{};
    std::array<std::uint64_t, kDiagnosticResourceCount> cumulative_resource_high_water_{};
    DiagnosticError first_error_;
};

}  // namespace amio::detail

#endif
