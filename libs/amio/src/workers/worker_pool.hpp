// worker_pool.hpp -- AMIO Worker_Pool background thread manager.
//
// This header is PRIVATE to the AMIO_Core build (`src/workers/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// The Worker_Pool manages a configurable number of background threads
// (1..256, default 1) that dequeue and execute write and prefetch
// tasks.  It provides:
//
//   * Two task queues:
//     - WriteQueue: FIFO globally, with per-(dataset, variable)
//       ordering mutex to preserve submission order at the storage
//       layer (R6.3).
//     - PrefetchQueue: priority-ordered by timestep distance from
//       the current read position.
//
//   * Per-(dataset, variable) sequence counter assigned at enqueue
//     time for order preservation.
//
//   * Per-(dataset, variable) mutex held only across the backend
//     serialize call, dropped before MPI-IO collectives (R3.7).
//
//   * Per-thread CPU/NUMA pinning at thread start (R3.2, R3.3).
//
//   * MPI communicator split integration: holds the IOCommunicator
//     so that all Backend_Driver MPI calls are routed through the
//     I/O communicator only (R3.5, R3.6).
//
//   * Graceful shutdown: drain queues on destruction, join all
//     threads.
//
// Thread safety
// -------------
// All public methods are safe to call concurrently from any thread.
// The implementation uses std::mutex + std::condition_variable for
// queue synchronization.
//
// Validates: R3.1, R3.2, R3.3, R3.4, R3.5, R3.6, R3.7, R3.8, R6.1, R6.2, R6.3, R6.8, R6.9

#ifndef AMIO_SRC_WORKERS_WORKER_POOL_HPP
#define AMIO_SRC_WORKERS_WORKER_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "workers/comm_split.hpp"
#include "workers/exception_bridge.hpp"
#include "workers/thread_pinning.hpp"

namespace amio::detail {

// DatasetVariableKey -- identifies a unique (dataset, variable) pair
// for ordering purposes.
struct DatasetVariableKey {
    std::uint64_t dataset_id = 0;
    std::uint64_t variable_id = 0;

    bool operator==(const DatasetVariableKey &other) const noexcept {
        return dataset_id == other.dataset_id && variable_id == other.variable_id;
    }

    bool operator<(const DatasetVariableKey &other) const noexcept {
        if (dataset_id != other.dataset_id) return dataset_id < other.dataset_id;
        return variable_id < other.variable_id;
    }
};

// WriteTask -- descriptor for a write task on the worker queue.
//
// The `callback` is invoked by the worker thread to perform the
// actual backend serialization.  The per-(dataset, variable) mutex
// ensures that writes to the same pair execute in submission order.
//
// The `handle_id` identifies the originating opaque handle so that
// exceptions can be recorded against it (R12.1, R12.2).
struct WriteTask {
    DatasetVariableKey dv_key;       // (dataset, variable) pair
    std::uint64_t dv_seq = 0;        // per-(dataset, variable) sequence number
    std::uint64_t handle_id = 0;     // originating opaque handle for outcome recording
    std::function<void()> callback;  // backend serialize functor
};

// PrefetchTask -- descriptor for a prefetch task on the worker queue.
//
// Priority is determined by `distance`: smaller distance means
// higher priority (closer to the current read position).
//
// The `handle_id` identifies the originating opaque handle so that
// exceptions can be recorded against it (R12.1, R12.2).
struct PrefetchTask {
    std::int64_t timestep = 0;       // target timestep
    std::int64_t distance = 0;       // distance from current read position
    std::uint64_t dataset_id = 0;    // owning dataset
    std::uint64_t handle_id = 0;     // originating opaque handle for outcome recording
    std::function<void()> callback;  // backend read functor

    // Higher priority = smaller distance (min-heap).
    bool operator>(const PrefetchTask &other) const noexcept {
        return distance > other.distance;
    }
};

// BackpressureConfig -- optional backpressure watermark configuration.
//
// When enabled, the Worker_Pool blocks write submissions when the
// queue depth reaches the high watermark (H) and unblocks when the
// depth falls below the low watermark (L).
//
// Invariant: 0 <= low_watermark < high_watermark <= queue_capacity
//
// When not enabled (enabled == false), the pool uses a simple
// queue capacity limit: if the queue is full, submit_write returns
// AMIO_ERR_QUEUE_FULL immediately without blocking (R6.9).
struct BackpressureConfig {
    bool enabled = false;
    std::size_t low_watermark = 0;
    std::size_t high_watermark = 0;
    std::size_t queue_capacity = 1024;  // default queue capacity
};

// WorkerPoolConfig -- configuration for constructing a WorkerPool.
//
// Bundles thread count, per-thread pinning configuration, the
// MPI communicator split result, and backpressure watermark
// settings so that the pool can apply pinning at thread start,
// route Backend_Driver MPI calls through the I/O communicator,
// and enforce queue admission control (R6.8, R6.9).
struct WorkerPoolConfig {
    std::size_t thread_count = 1;

    // Per-thread pinning configs.  If non-empty, element [i] is
    // applied to worker thread i.  If the vector is shorter than
    // thread_count, remaining threads use default (no pinning).
    // If empty, no pinning is applied to any thread.
    std::vector<ThreadConfig> thread_configs;

    // MPI communicator split result.  If valid, Backend_Driver MPI
    // calls are routed through io_comm only (R3.5).
    IOCommunicator io_comm;

    // Backpressure watermark configuration (R6.8, R6.9).
    // When enabled: queue depth >= H blocks until depth < L.
    // When not enabled: queue depth >= capacity returns
    // AMIO_ERR_QUEUE_FULL.
    BackpressureConfig backpressure;
};

// WorkerPool -- background thread pool with write and prefetch queues.
//
// Construction starts N worker threads.  Destruction signals shutdown,
// drains remaining tasks, and joins all threads.
class WorkerPool {
   public:
    // Configuration limits (from design.md / requirements).
    static constexpr std::size_t kMinThreadCount = 0;
    static constexpr std::size_t kMaxThreadCount = 256;
    static constexpr std::size_t kDefaultThreadCount = 1;

    // Construct a pool with `thread_count` worker threads.
    //
    // Preconditions (enforced by assertions in debug builds;
    // callers are expected to validate via Config_Loader before
    // constructing):
    //   thread_count in [kMinThreadCount, kMaxThreadCount]
    explicit WorkerPool(std::size_t thread_count = kDefaultThreadCount);

    // Construct a pool with full configuration including thread
    // pinning and MPI communicator split.
    //
    // Each worker thread applies its ThreadConfig at start (R3.2).
    // If any pinning fails, the thread continues without pinning
    // (the error is recorded but does not abort the pool -- the
    // caller should have validated bindings before constructing).
    //
    // The IOCommunicator is stored and exposed via io_communicator()
    // so that Backend_Driver callbacks can route MPI calls through
    // the I/O communicator only (R3.5).
    explicit WorkerPool(const WorkerPoolConfig &config);

    // Non-copyable, non-movable (owns threads).
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;
    WorkerPool(WorkerPool &&) = delete;
    WorkerPool &operator=(WorkerPool &&) = delete;

    // Destructor: signals shutdown, drains queues, joins threads.
    ~WorkerPool();

    // submit_write -- enqueue a write task.
    //
    // The task is assigned a per-(dataset, variable) sequence number
    // to preserve submission order.  The callback will be invoked
    // on a worker thread with the per-(dataset, variable) ordering
    // mutex held (dropped before any MPI-IO collective).
    //
    // Backpressure behavior (R6.8, R6.9):
    //   * If backpressure is configured and queue depth >= H:
    //     blocks until depth < L, then enqueues.
    //   * If backpressure is NOT configured and queue depth >=
    //     capacity: returns AMIO_ERR_QUEUE_FULL without enqueuing.
    //
    // Returns the assigned sequence number for the task (via out
    // parameter `seq_out`).  Returns AMIO_OK on success, or
    // AMIO_ERR_QUEUE_FULL if the queue is at capacity and no
    // backpressure is configured.
    amio_err_t submit_write(DatasetVariableKey dv_key, std::function<void()> callback, std::uint64_t *seq_out);

    // submit_write -- convenience overload (legacy interface).
    //
    // Returns the assigned sequence number on success, or 0 if
    // the pool is shut down or the queue is full.  Prefer the
    // error-returning overload for new code.
    std::uint64_t submit_write(DatasetVariableKey dv_key, std::function<void()> callback);

    // submit_write with handle_id -- enqueue a write task with an
    // associated opaque handle for exception cordon outcome recording.
    //
    // Same as submit_write above, but exceptions caught by the
    // exception cordon (R12.1, R12.2) are recorded against the
    // specified handle_id in the outcome registry.
    //
    // Returns the assigned sequence number on success, or 0 if
    // the pool is shut down or the queue is full.
    std::uint64_t submit_write(DatasetVariableKey dv_key, std::uint64_t handle_id, std::function<void()> callback);

    // submit_prefetch -- enqueue a prefetch task.
    //
    // The task is prioritized by `distance` from the current read
    // position (smaller distance = higher priority).
    void submit_prefetch(std::int64_t timestep, std::int64_t distance, std::uint64_t dataset_id, std::function<void()> callback);

    // submit_prefetch with handle_id -- enqueue a prefetch task with
    // an associated opaque handle for exception cordon outcome recording.
    void submit_prefetch(std::int64_t timestep, std::int64_t distance, std::uint64_t dataset_id, std::uint64_t handle_id,
                         std::function<void()> callback);

    // drain -- block until all currently enqueued tasks complete.
    //
    // New tasks submitted after drain() is called are NOT waited on.
    void drain();

    // shutdown -- signal all workers to stop after draining.
    //
    // Called automatically by the destructor.  After shutdown(),
    // submit_write and submit_prefetch are no-ops.
    void shutdown();

    // ----- Diagnostics / test helpers -----

    // thread_count -- the number of worker threads.
    std::size_t thread_count() const noexcept;

    // write_queue_depth -- current number of pending write tasks.
    std::size_t write_queue_depth() const noexcept;

    // prefetch_queue_depth -- current number of pending prefetch tasks.
    std::size_t prefetch_queue_depth() const noexcept;

    // total_writes_completed -- total write tasks completed since
    // construction.
    std::uint64_t total_writes_completed() const noexcept;

    // total_prefetches_completed -- total prefetch tasks completed
    // since construction.
    std::uint64_t total_prefetches_completed() const noexcept;

    // is_shutdown -- whether shutdown has been signaled.
    bool is_shutdown() const noexcept;

    // queue_capacity -- the configured queue capacity.
    std::size_t queue_capacity() const noexcept;

    // backpressure_enabled -- whether backpressure watermarks are
    // configured.
    bool backpressure_enabled() const noexcept;

    // low_watermark -- the configured low watermark (0 if not enabled).
    std::size_t low_watermark() const noexcept;

    // high_watermark -- the configured high watermark (0 if not enabled).
    std::size_t high_watermark() const noexcept;

    // io_communicator -- the I/O communicator for Backend_Driver MPI
    // calls.  All Backend_Driver MPI operations MUST use this
    // communicator rather than MPI_COMM_WORLD (R3.5).
    //
    // If no communicator split was configured (default mode), the
    // returned IOCommunicator has valid=true and is_io_rank=true,
    // meaning all ranks participate in I/O on the world communicator.
    const IOCommunicator &io_communicator() const noexcept;

    // pinning_errors -- returns the number of threads that failed
    // to apply their CPU/NUMA pinning configuration at start.
    // A non-zero value indicates degraded affinity isolation.
    std::size_t pinning_errors() const noexcept;

    // outcome_registry -- access the exception cordon outcome registry.
    //
    // Worker threads record task outcomes here after exception cordon
    // processing (R12.1, R12.2).  The C-boundary layer queries
    // outcomes on flush/close/wait to surface failures (R12.9).
    OutcomeRegistry &outcome_registry() noexcept;
    const OutcomeRegistry &outcome_registry() const noexcept;

   private:
    // Worker thread main loop (no pinning).
    void worker_loop();

    // Worker thread main loop with pinning config for thread index.
    void worker_loop_pinned(std::size_t thread_index);

    // Try to dequeue and execute one task.  Returns true if a task
    // was executed, false if no task was available (caller should
    // wait on CV).
    bool try_execute_one(std::unique_lock<std::mutex> &lock);

    // Get or create the per-(dataset, variable) ordering state.
    struct DvOrderState {
        std::mutex mu;               // ordering mutex
        std::uint64_t next_seq = 0;  // next sequence number to assign
        std::uint64_t exec_seq = 0;  // next sequence number to execute
    };

    DvOrderState &get_dv_state(const DatasetVariableKey &key);

    // ---- Members ----

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable drain_cv_;
    std::condition_variable backpressure_cv_;  // for blocking writers (R6.8)

    // Write queue: FIFO order.
    std::queue<WriteTask> write_queue_;

    // Prefetch queue: min-heap by distance.
    std::priority_queue<PrefetchTask, std::vector<PrefetchTask>, std::greater<PrefetchTask>> prefetch_queue_;

    // Per-(dataset, variable) ordering state.
    // Protected by mu_ for map access; individual DvOrderState::mu
    // is independent.
    std::map<DatasetVariableKey, std::unique_ptr<DvOrderState>> dv_states_;

    // Worker threads.
    std::vector<std::thread> workers_;

    // Shutdown flag.
    std::atomic<bool> shutdown_{false};

    // In-flight task count (for drain).
    std::atomic<std::uint64_t> in_flight_{0};

    // Completion counters.
    std::atomic<std::uint64_t> writes_completed_{0};
    std::atomic<std::uint64_t> prefetches_completed_{0};

    // Configuration.
    std::size_t thread_count_;

    // Per-thread pinning configurations (R3.2).
    std::vector<ThreadConfig> thread_configs_;

    // I/O communicator for Backend_Driver MPI calls (R3.5).
    IOCommunicator io_comm_;

    // Count of threads that failed to apply pinning at start.
    std::atomic<std::size_t> pinning_errors_{0};

    // Backpressure configuration (R6.8, R6.9).
    BackpressureConfig backpressure_;

    // Exception cordon outcome registry (R12.1, R12.2, R12.9, R12.10).
    // Records task outcomes against originating opaque handles.
    OutcomeRegistry outcome_registry_;
};

}  // namespace amio::detail

#endif  // AMIO_SRC_WORKERS_WORKER_POOL_HPP
