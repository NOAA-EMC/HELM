// worker_pool.cpp -- AMIO Worker_Pool background thread manager implementation.
//
// Implements the background thread pool described in design.md §3
// (Staging Pool & Worker Pool).  Workers dequeue tasks from two
// queues (WriteQueue and PrefetchQueue) and execute them with
// per-(dataset, variable) ordering guarantees.
//
// Key design decisions:
//
//   * Write tasks are dequeued FIFO.  Before executing a write
//     callback, the worker acquires the per-(dataset, variable)
//     ordering mutex and verifies that the task's sequence number
//     matches the expected execution sequence.  If not, the task
//     is re-enqueued (another write to the same pair must complete
//     first).  This ensures writes to the same (dataset, variable)
//     pair execute in submission order regardless of thread count.
//
//   * Prefetch tasks are dequeued by priority (smallest distance
//     from current read position first).
//
//   * Write tasks take priority over prefetch tasks to minimize
//     write latency and avoid backpressure.
//
//   * An ordering-domain reservation prevents concurrent execution of
//     equal keys. No WorkerPool lock is held while callbacks issue MPI-IO.
//
//   * Per-thread CPU/NUMA pinning is applied at thread start using
//     the ThreadConfig for that thread index (R3.2).  Invalid
//     bindings are recorded but do not abort the pool (the caller
//     should have validated bindings during init -- R3.3).
//
//   * The IOCommunicator from the MPI communicator split is stored
//     and exposed via io_communicator() so Backend_Driver callbacks
//     can route all MPI calls through the I/O communicator (R3.5).
//
// Validates: R3.1, R3.2, R3.3, R3.4, R3.5, R3.6, R3.7, R3.8, R6.1, R6.2, R6.3, R6.8, R6.9, R12.1, R12.2, R12.3, R12.4, R12.9, R12.10

#include "workers/worker_pool.hpp"

#include <cassert>

namespace amio::detail {

WorkerPool::WorkerPool(std::size_t thread_count) : thread_count_(thread_count) {
    assert(thread_count >= kMinThreadCount && thread_count <= kMaxThreadCount);

    // Default IOCommunicator: all ranks do I/O on world communicator.
    io_comm_.valid = true;
    io_comm_.is_io_rank = true;

    workers_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back(&WorkerPool::worker_loop, this);
    }
}

WorkerPool::WorkerPool(const WorkerPoolConfig &config)
    : thread_count_(config.thread_count), thread_configs_(config.thread_configs), io_comm_(config.io_comm), backpressure_(config.backpressure) {
    assert(config.thread_count >= kMinThreadCount && config.thread_count <= kMaxThreadCount);

    // If no IOCommunicator was provided (default-constructed), set
    // up the default "all ranks do I/O" mode.
    if (!io_comm_.valid) {
        io_comm_.valid = true;
        io_comm_.is_io_rank = true;
    }

    workers_.reserve(config.thread_count);
    for (std::size_t i = 0; i < config.thread_count; ++i) {
        workers_.emplace_back(&WorkerPool::worker_loop_pinned, this, i);
    }
}

WorkerPool::~WorkerPool() {
    shutdown();
}

void WorkerPool::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) {
        // Already shut down -- just ensure threads are joined.
        for (auto &t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        return;
    }

    // Wake all workers so they can observe the shutdown flag.
    cv_.notify_all();
    backpressure_cv_.notify_all();  // Unblock any writers waiting on backpressure.

    for (auto &t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

std::uint64_t WorkerPool::submit_write(DatasetVariableKey dv_key, std::function<void()> callback) {
    std::uint64_t seq = 0;
    amio_err_t rc = submit_write(dv_key, std::move(callback), &seq);
    if (rc != AMIO_OK) {
        return 0;  // Queue full or shutdown.
    }
    return seq;
}

std::uint64_t WorkerPool::submit_write(DatasetVariableKey dv_key, std::uint64_t handle_id, std::function<void()> callback) {
    std::uint64_t seq = 0;
    if (submit_write(dv_key, handle_id, std::move(callback), &seq) != AMIO_OK) return 0;
    return seq;
}

amio_err_t WorkerPool::submit_write(DatasetVariableKey dv_key, std::function<void()> callback, std::uint64_t *seq_out) {
    return submit_write(dv_key, 0, std::move(callback), seq_out);
}

amio_err_t WorkerPool::submit_write(DatasetVariableKey dv_key, std::uint64_t handle_id, std::function<void()> callback, std::uint64_t *seq_out) {
    if (seq_out) *seq_out = 0;
    std::unique_lock<std::mutex> lock(mu_);
    if (shutdown_.load(std::memory_order_acquire)) return AMIO_ERR_QUEUE_FULL;

    if (backpressure_.enabled && write_queue_.size() >= backpressure_.high_watermark) {
        backpressure_cv_.wait(lock,
                              [this]() { return write_queue_.size() <= backpressure_.low_watermark || shutdown_.load(std::memory_order_acquire); });
    }
    if (shutdown_.load(std::memory_order_acquire) || write_queue_.size() >= backpressure_.queue_capacity) {
        return AMIO_ERR_QUEUE_FULL;
    }

    DvOrderState &state = get_dv_state(dv_key);
    const std::uint64_t seq = state.next_seq;
    WriteTask task;
    task.dv_key = dv_key;
    task.dv_seq = seq;
    task.handle_id = handle_id;
    task.callback = std::move(callback);
    write_queue_.push(std::move(task));
    ++state.next_seq;
    if (seq_out) *seq_out = seq;
    lock.unlock();
    cv_.notify_one();
    return AMIO_OK;
}

void WorkerPool::submit_prefetch(std::int64_t timestep, std::int64_t distance, std::uint64_t dataset_id, std::function<void()> callback) {
    if (shutdown_.load(std::memory_order_acquire)) {
        return;  // No-op after shutdown.
    }

    {
        std::lock_guard<std::mutex> lock(mu_);

        PrefetchTask task;
        task.timestep = timestep;
        task.distance = distance;
        task.dataset_id = dataset_id;
        task.callback = std::move(callback);
        prefetch_queue_.push(std::move(task));
    }

    cv_.notify_one();
}

void WorkerPool::submit_prefetch(std::int64_t timestep, std::int64_t distance, std::uint64_t dataset_id, std::uint64_t handle_id,
                                 std::function<void()> callback) {
    if (shutdown_.load(std::memory_order_acquire)) {
        return;  // No-op after shutdown.
    }

    {
        std::lock_guard<std::mutex> lock(mu_);

        PrefetchTask task;
        task.timestep = timestep;
        task.distance = distance;
        task.dataset_id = dataset_id;
        task.handle_id = handle_id;
        task.callback = std::move(callback);
        prefetch_queue_.push(std::move(task));
    }

    cv_.notify_one();
}

void WorkerPool::drain() {
    std::unique_lock<std::mutex> lock(mu_);
    drain_cv_.wait(lock, [this]() { return write_queue_.empty() && prefetch_queue_.empty() && in_flight_.load(std::memory_order_acquire) == 0; });
}

std::size_t WorkerPool::thread_count() const noexcept {
    return thread_count_;
}

std::size_t WorkerPool::write_queue_depth() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return write_queue_.size();
}

std::size_t WorkerPool::prefetch_queue_depth() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return prefetch_queue_.size();
}

std::uint64_t WorkerPool::total_writes_completed() const noexcept {
    return writes_completed_.load(std::memory_order_acquire);
}

std::uint64_t WorkerPool::total_prefetches_completed() const noexcept {
    return prefetches_completed_.load(std::memory_order_acquire);
}

bool WorkerPool::is_shutdown() const noexcept {
    return shutdown_.load(std::memory_order_acquire);
}

const IOCommunicator &WorkerPool::io_communicator() const noexcept {
    return io_comm_;
}

std::size_t WorkerPool::pinning_errors() const noexcept {
    return pinning_errors_.load(std::memory_order_acquire);
}

std::size_t WorkerPool::queue_capacity() const noexcept {
    return backpressure_.queue_capacity;
}

bool WorkerPool::backpressure_enabled() const noexcept {
    return backpressure_.enabled;
}

std::size_t WorkerPool::low_watermark() const noexcept {
    return backpressure_.low_watermark;
}

std::size_t WorkerPool::high_watermark() const noexcept {
    return backpressure_.high_watermark;
}

OutcomeRegistry &WorkerPool::outcome_registry() noexcept {
    return outcome_registry_;
}

const OutcomeRegistry &WorkerPool::outcome_registry() const noexcept {
    return outcome_registry_;
}

// ---- Private implementation ----

void WorkerPool::worker_loop_pinned(std::size_t thread_index) {
    // Apply per-thread CPU/NUMA pinning at thread start (R3.2).
    // If the thread_configs_ vector has a config for this index,
    // apply it.  Otherwise, use default (no pinning).
    if (thread_index < thread_configs_.size()) {
        const ThreadConfig &tc = thread_configs_[thread_index];
        amio_err_t pin_rc = apply_thread_pinning(tc);
        if (pin_rc != AMIO_OK) {
            // Record the pinning failure.  The thread continues
            // without pinning -- the caller (amio_init) should have
            // validated bindings before constructing the pool (R3.3).
            pinning_errors_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Enter the normal worker loop.
    worker_loop();
}

void WorkerPool::worker_loop() {
    while (true) {
        std::unique_lock<std::mutex> lock(mu_);

        // Wait until there's work or shutdown is signaled.
        // 100ms polling fallback prevents hangs if notify is missed.
        cv_.wait_for(lock, std::chrono::milliseconds(50),
                     [this]() { return shutdown_.load(std::memory_order_relaxed) || !write_queue_.empty() || !prefetch_queue_.empty(); });

        // Check shutdown after waking.
        if (shutdown_.load(std::memory_order_acquire) && write_queue_.empty() && prefetch_queue_.empty()) {
            break;
        }

        // Try to execute a task.
        if (!try_execute_one(lock)) {
            // No executable task found.  If shutting down and queues
            // are empty, exit the loop.
            if (shutdown_.load(std::memory_order_acquire) && write_queue_.empty() && prefetch_queue_.empty()) {
                return;
            }
            // A predecessor is in flight. Release the queue lock so its
            // completion can advance the sequence without a busy spin.
            cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
}

bool WorkerPool::try_execute_one(std::unique_lock<std::mutex> &lock) {
    // Priority: write tasks first, then prefetch tasks.

    // --- Try write queue ---
    if (!write_queue_.empty()) {
        WriteTask task = std::move(write_queue_.front());
        write_queue_.pop();

        DvOrderState &state = get_dv_state(task.dv_key);

        // Check if this task is the next one to execute for its
        // (dataset, variable) pair.
        if (state.running || task.dv_seq != state.exec_seq) {
            // Not ready yet -- re-enqueue and try something else.
            write_queue_.push(std::move(task));
            // Try a prefetch task instead.
            if (!prefetch_queue_.empty()) {
                PrefetchTask ptask = std::move(const_cast<PrefetchTask &>(prefetch_queue_.top()));
                prefetch_queue_.pop();

                in_flight_.fetch_add(1, std::memory_order_acq_rel);
                lock.unlock();

                // Execute prefetch callback with exception cordon
                // (R12.1, R12.2).
                if (ptask.handle_id != 0) {
                    execute_with_exception_cordon(ptask.callback, ptask.handle_id, io_comm_, outcome_registry_);
                } else {
                    try {
                        if (ptask.callback) {
                            ptask.callback();
                        }
                    } catch (const std::exception &e) {
                        emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, e.what());
                    } catch (...) {
                        emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, "Unknown exception (non-std)");
                    }
                }

                {
                    // drain() checks in_flight_ while holding mu_. Update it
                    // under the same mutex so its predicate cannot miss the
                    // transition to idle between checking and sleeping.
                    std::lock_guard<std::mutex> completion_lock(mu_);
                    prefetches_completed_.fetch_add(1, std::memory_order_release);
                    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                }
                drain_cv_.notify_all();
                return true;
            }
            return false;
        }

        in_flight_.fetch_add(1, std::memory_order_acq_rel);
        state.running = true;
        lock.unlock();
        backpressure_cv_.notify_all();

        // The running flag reserves this ordering domain until completion.
        // No WorkerPool mutex is held across the backend callback or MPI-IO.
        //
        // Exception cordon (R12.1, R12.2, R12.3, R12.4):
        // The callback is wrapped in try/catch.  On exception:
        //   1. emit_parallel_stacktrace (collective) BEFORE recording
        //   2. Record outcome against originating handle
        //   3. Buffer release happens after (caller's responsibility)
        {
            if (task.handle_id != 0) {
                // Use the full exception cordon with outcome recording.
                execute_with_exception_cordon(task.callback, task.handle_id, io_comm_, outcome_registry_);
            } else {
                // Legacy path: no handle_id, use basic exception cordon.
                try {
                    if (task.callback) {
                        task.callback();
                    }
                } catch (const std::exception &e) {
                    emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, e.what());
                } catch (...) {
                    emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, "Unknown exception (non-std)");
                }
            }
            // Advance the execution sequence counter.
            std::lock_guard<std::mutex> queue_lock(mu_);
            state.exec_seq++;
            state.running = false;
        }

        {
            std::lock_guard<std::mutex> completion_lock(mu_);
            writes_completed_.fetch_add(1, std::memory_order_release);
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        }

        // Notify: drain waiters and other workers that may have
        // re-enqueued tasks waiting on this sequence.
        // Also notify backpressure waiters (R6.8): queue depth
        // decreased, so blocked writers may proceed.
        drain_cv_.notify_all();
        cv_.notify_all();
        backpressure_cv_.notify_all();
        return true;
    }

    // --- Try prefetch queue ---
    if (!prefetch_queue_.empty()) {
        PrefetchTask task = std::move(const_cast<PrefetchTask &>(prefetch_queue_.top()));
        prefetch_queue_.pop();

        in_flight_.fetch_add(1, std::memory_order_acq_rel);
        lock.unlock();

        // Execute prefetch callback with exception cordon (R12.1, R12.2).
        if (task.handle_id != 0) {
            execute_with_exception_cordon(task.callback, task.handle_id, io_comm_, outcome_registry_);
        } else {
            try {
                if (task.callback) {
                    task.callback();
                }
            } catch (const std::exception &e) {
                emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, e.what());
            } catch (...) {
                emit_parallel_stacktrace(io_comm_, AMIO_ERR_BACKEND_FAILURE, "Unknown exception (non-std)");
            }
        }

        {
            std::lock_guard<std::mutex> completion_lock(mu_);
            prefetches_completed_.fetch_add(1, std::memory_order_release);
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        }
        drain_cv_.notify_all();
        return true;
    }

    return false;
}

WorkerPool::DvOrderState &WorkerPool::get_dv_state(const DatasetVariableKey &key) {
    // Caller must hold mu_.
    auto it = dv_states_.find(key);
    if (it == dv_states_.end()) {
        auto [inserted, _] = dv_states_.emplace(key, std::make_unique<DvOrderState>());
        return *inserted->second;
    }
    return *it->second;
}

}  // namespace amio::detail
