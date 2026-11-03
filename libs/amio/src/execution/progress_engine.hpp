#ifndef AMIO_SRC_EXECUTION_PROGRESS_ENGINE_HPP
#define AMIO_SRC_EXECUTION_PROGRESS_ENGINE_HPP

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "execution/context.hpp"

namespace amio::detail {

/** Bounded progress queue: caller policy runs callbacks only from progress(); background
 * policy runs them on one worker after MPI_THREAD_MULTIPLE has been validated.
 */
class ProgressEngine {
   public:
    using Callback = std::function<amio_status_t()>;
    ProgressEngine(ProgressPolicy policy, std::size_t queue_capacity, int thread_level);
    ProgressEngine(const ProgressEngine &) = delete;
    ProgressEngine &operator=(const ProgressEngine &) = delete;
    ~ProgressEngine();

    [[nodiscard]] amio_status_t submit(Callback callback);
    [[nodiscard]] amio_status_t progress(std::size_t max_callbacks, std::size_t &completed);
    [[nodiscard]] amio_status_t close(std::chrono::milliseconds timeout);
    [[nodiscard]] std::size_t pending() const noexcept;

   private:
    void background_loop() noexcept;
    amio_status_t run_one(Callback callback) noexcept;
    ProgressPolicy policy_;
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable idle_;
    std::deque<Callback> queue_;
    std::thread worker_;
    std::size_t active_ = 0;
    bool closing_ = false;
    bool closed_ = false;
    amio_status_t first_error_ = AMIO_OK;
};

}  // namespace amio::detail

#endif
