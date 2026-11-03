#include "execution/progress_engine.hpp"

#include <stdexcept>
#include <utility>

namespace amio::detail {

ProgressEngine::ProgressEngine(ProgressPolicy policy, std::size_t queue_capacity, int thread_level) : policy_(policy), capacity_(queue_capacity) {
    if (capacity_ == 0) throw std::invalid_argument("progress queue capacity must be positive");
    if (policy_ == ProgressPolicy::Background) {
        // MPI_THREAD_MULTIPLE is the standard value 3. Keep this small engine
        // independent of the MPI runtime; callers obtain the level via HALO.
        if (thread_level < 3) throw std::invalid_argument("background MPI progress requires MPI_THREAD_MULTIPLE");
        worker_ = std::thread(&ProgressEngine::background_loop, this);
    }
}

ProgressEngine::~ProgressEngine() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable()) worker_.join();
}

amio_status_t ProgressEngine::submit(Callback callback) {
    if (!callback) return AMIO_ERR_INVALID_INPUT;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_ || closed_) return AMIO_ERR_INVALID_POLICY;
        if (queue_.size() + active_ >= capacity_) return AMIO_ERR_QUEUE_FULL;
        queue_.push_back(std::move(callback));
    }
    ready_.notify_one();
    return AMIO_OK;
}

amio_status_t ProgressEngine::run_one(Callback callback) noexcept {
    amio_status_t status = AMIO_OK;
    try {
        status = callback();
    } catch (...) {
        status = AMIO_ERR_BACKEND_FAILURE;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status != AMIO_OK && first_error_ == AMIO_OK) first_error_ = status;
        --active_;
        if (active_ == 0 && queue_.empty()) idle_.notify_all();
    }
    return status;
}

amio_status_t ProgressEngine::progress(std::size_t max_callbacks, std::size_t &completed) {
    completed = 0;
    if (policy_ != ProgressPolicy::Caller) return AMIO_ERR_INVALID_POLICY;
    if (max_callbacks == 0) return AMIO_ERR_INVALID_INPUT;
    amio_status_t result = AMIO_OK;
    while (completed < max_callbacks) {
        Callback callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) break;
            callback = std::move(queue_.front());
            queue_.pop_front();
            ++active_;
        }
        const auto status = run_one(std::move(callback));
        ++completed;
        if (status != AMIO_OK && result == AMIO_OK) result = status;
    }
    if (result == AMIO_OK) {
        std::lock_guard<std::mutex> lock(mutex_);
        result = first_error_;
    }
    return result;
}

amio_status_t ProgressEngine::close(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    ready_.notify_all();
    if (!idle_.wait_for(lock, timeout, [this] { return queue_.empty() && active_ == 0; })) return AMIO_ERR_OPERATION_PENDING;
    closed_ = true;
    return first_error_;
}

std::size_t ProgressEngine::pending() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() + active_;
}

void ProgressEngine::background_loop() noexcept {
    for (;;) {
        Callback callback;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return closing_ || !queue_.empty(); });
            if (queue_.empty() && closing_) {
                idle_.notify_all();
                closed_ = true;
                return;
            }
            callback = std::move(queue_.front());
            queue_.pop_front();
            ++active_;
        }
        (void)run_one(std::move(callback));
    }
}

}  // namespace amio::detail
