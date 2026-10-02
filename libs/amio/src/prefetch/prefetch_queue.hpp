// prefetch_queue.hpp -- AMIO Prefetch_Queue for read look-ahead.
//
// This header is PRIVATE to the AMIO_Core build (`src/prefetch/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// The Prefetch_Queue manages background read look-ahead for the
// read path.  It:
//
//   * Holds a configurable look-ahead depth N (from manifest,
//     default 4, range [1, 1024]).
//   * Tracks completed buffers by timestep (map<int64_t, StagingBuffer*>).
//   * Tracks pending fetches (set of timesteps being fetched).
//   * Tracks failed fetches (map<int64_t, amio_err_t>).
//   * Supports scheduling background fetches via the Worker_Pool
//     or synchronous fallback when no Worker_Pool is available.
//   * Supports checking if a timestep is ready.
//
// Thread safety
// -------------
// All public methods are safe to call concurrently from any thread.
// The implementation uses std::mutex + std::condition_variable.
//
// Validates: R5.1, R5.2, R5.3, R5.4, R5.5, R5.7, R5.8

#ifndef AMIO_SRC_PREFETCH_PREFETCH_QUEUE_HPP
#define AMIO_SRC_PREFETCH_PREFETCH_QUEUE_HPP

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#include "amio/amio_errors.h"
#include "amio/amio_types.h"
#include "factory/backend_driver.hpp"  // VariableInfo, VarMeta, element_size

namespace amio::detail {

// Forward declarations.
struct StagingBuffer;
class StagingPool;
class WorkerPool;
class Backend_Driver;
struct BoundingBox;
struct VarMeta;

// PrefetchQueue -- look-ahead prefetch manager for read datasets.
//
// On open_read, the queue schedules min(N, M) background fetches.
// On each successful amio_read(T), if T+N is within bounds, the
// queue schedules a fetch for T+N to maintain the look-ahead depth.
//
// When a Worker_Pool is available, fetches are dispatched as
// PrefetchTasks.  When no Worker_Pool is available (stub mode),
// fetches are performed synchronously through the Backend_Driver.
class PrefetchQueue {
   public:
    // Configuration limits.
    static constexpr std::size_t kMinDepth = 1;
    static constexpr std::size_t kMaxDepth = 1024;
    static constexpr std::size_t kDefaultDepth = 4;

    static constexpr std::int64_t kMinReadTimeoutS = 1;
    static constexpr std::int64_t kMaxReadTimeoutS = 3600;
    static constexpr std::int64_t kDefaultReadTimeoutS = 60;

    // Construct a PrefetchQueue with the given look-ahead depth,
    // read timeout, and references to the staging pool, worker pool,
    // and backend driver.
    //
    // Parameters:
    //   depth          - look-ahead depth N in [1, 1024]
    //   read_timeout_s - read timeout in seconds [1, 3600]
    //   pool           - staging pool for buffer acquisition
    //   workers        - worker pool for background fetches (may be null)
    //   driver         - backend driver for read operations
    //   dataset_id     - owning dataset identifier
    //   var_name       - variable name for reads
    //   info           - variable metadata (dtype + shape) used to
    //                    build VarMeta and size the staging buffer
    //   total_timesteps - total number of timesteps in the dataset
    PrefetchQueue(std::size_t depth, std::int64_t read_timeout_s, StagingPool *pool, WorkerPool *workers, Backend_Driver *driver,
                  std::uint64_t dataset_id, const std::string &var_name, const VariableInfo &info, std::int64_t total_timesteps);

    ~PrefetchQueue();

    // Non-copyable, non-movable.
    PrefetchQueue(const PrefetchQueue &) = delete;
    PrefetchQueue &operator=(const PrefetchQueue &) = delete;
    PrefetchQueue(PrefetchQueue &&) = delete;
    PrefetchQueue &operator=(PrefetchQueue &&) = delete;

    // schedule_initial -- schedule the initial min(N, M) fetches.
    //
    // Schedules fetches for timesteps [0, min(N, M)) using the currently
    // latched selection (see get_buffer).  Exposed for tests and for callers
    // that want to warm the window from record 0; the normal read path does
    // NOT need it, because get_buffer self-kicks-off the look-ahead window
    // at the first timestep the host actually requests.
    void schedule_initial();

    // get_buffer -- retrieve the completed buffer for timestep T.
    //
    // If the buffer is ready, returns it immediately (no I/O on
    // calling thread).  If not ready, blocks until the worker
    // completes the fetch or the read timeout expires.
    //
    // Selection (bounding box) handling:
    //   * The returned buffer ALWAYS matches `bbox` exactly.  A completed
    //     fetch made with a different selection is released and re-fetched,
    //     so varying the selection between calls stays correct.
    //   * `bbox` is also latched as the queue's current selection, so every
    //     look-ahead fetch (the initial window and schedule_next
    //     replenishment) reads the same sub-region the host is consuming
    //     rather than a full record.  This is what makes band-scoped reads
    //     on an MPI rank actually reduce disk traffic: without the latch,
    //     prefetch would keep issuing full-record reads behind the caller's
    //     back.
    //   * On the first call for this variable the look-ahead window is
    //     kicked off starting at `timestep` (not at 0), so records the host
    //     never reads are never staged.
    //
    // Returns:
    //   AMIO_OK with *out_buf set on success.
    //   AMIO_ERR_TIMEOUT if the read timeout expires.
    //   AMIO_ERR_BACKEND_FAILURE (or other AMIO_ERR_*) if the
    //     background fetch failed.
    amio_status_t get_buffer(std::int64_t timestep, const amio_bbox_t *bbox, StagingBuffer **out_buf);

    // schedule_next -- schedule fetch for timestep T+N if within bounds.
    //
    // Called after a successful read of timestep T to maintain the
    // look-ahead depth.
    void schedule_next(std::int64_t current_timestep);

    // mark_complete -- mark a timestep as successfully fetched.
    //
    // Called by the worker thread (or synchronous fallback) when
    // the fetch completes.  `bbox` records the selection the buffer was
    // actually filled with, so get_buffer can tell an exact match from a
    // stale window.  It defaults to nullopt (full record) for callers that
    // do not care.
    void mark_complete(std::int64_t timestep, StagingBuffer *buf, const std::optional<amio_bbox_t> &bbox = std::nullopt);

    // mark_failed -- record a fetch failure for a timestep.
    //
    // Called by the worker thread (or synchronous fallback) when
    // the fetch fails.
    void mark_failed(std::int64_t timestep, amio_err_t error);

    // cancel_pending -- cancel all pending fetches and release
    // completed buffers.
    //
    // Called on dataset close.
    void cancel_pending();

    // ----- Diagnostics / test helpers -----

    std::size_t depth() const noexcept {
        return depth_;
    }
    std::int64_t total_timesteps() const noexcept {
        return total_timesteps_;
    }
    std::size_t completed_count() const noexcept;
    std::size_t pending_count() const noexcept;
    std::size_t failed_count() const noexcept;

   private:
    // Schedule a single fetch for the given timestep, using the currently
    // latched selection.  Caller must NOT hold mu_.
    void schedule_fetch(std::int64_t timestep);

    // Perform a synchronous fetch (used when worker_pool is null).
    // Caller must NOT hold mu_.
    void sync_fetch(std::int64_t timestep, const amio_bbox_t *bbox);

    // bbox_same -- true when two optional selections address exactly the
    // same sub-region (rank, offsets, extents and strides all equal).
    // Two nullopts are "the same" (both full-record).
    static bool bbox_same(const std::optional<amio_bbox_t> &a, const std::optional<amio_bbox_t> &b) noexcept;

    // ---- Members ----

    mutable std::mutex mu_;
    std::condition_variable cv_;

    // Configuration.
    std::size_t depth_;
    std::int64_t read_timeout_s_;
    std::int64_t total_timesteps_;
    std::uint64_t dataset_id_;
    std::string var_name_;
    VariableInfo info_;  // dtype + shape for VarMeta / buffer sizing

    // External references (not owned).
    StagingPool *pool_;
    WorkerPool *workers_;
    Backend_Driver *driver_;

    // State tracking.
    std::map<std::int64_t, StagingBuffer *> completed_;  // timestep -> buffer
    std::map<std::int64_t, amio_bbox_t> fetched_bbox_;   // timestep -> selection that filled completed_[t]
    std::set<std::int64_t> pending_;                     // timesteps being fetched
    std::map<std::int64_t, amio_err_t> failed_;          // timestep -> error code

    // Latched selection: the sub-region the host is consuming, taken from
    // the most recent get_buffer call.  Look-ahead fetches use it so
    // prefetch traffic matches the caller's window.  nullopt == full record.
    std::optional<amio_bbox_t> latched_bbox_;

    // True once the look-ahead window has been opened (by the first
    // get_buffer, or by an explicit schedule_initial call).
    bool kicked_off_ = false;

    // Cancellation flag.
    bool cancelled_ = false;
};

}  // namespace amio::detail

#endif  // AMIO_SRC_PREFETCH_PREFETCH_QUEUE_HPP
