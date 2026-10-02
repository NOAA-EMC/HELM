// exception_bridge.hpp -- AMIO exception cordon and parallel diagnostics.
//
// This header is PRIVATE to the AMIO_Core build (`src/workers/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// The exception bridge provides:
//
//   * A TaskOutcome struct to record success/failure with error code
//     and optional stack trace against an originating handle.
//
//   * An emit_parallel_stacktrace function that performs a collective
//     diagnostic emission on the I/O communicator BEFORE recording
//     the outcome (R12.3, R12.4).
//
//   * Exception translation logic (ordered most-specific first):
//       - conf::Conf_Error  → AMIO_ERR_MANIFEST_NOT_FOUND,
//                              AMIO_ERR_MANIFEST_INVALID, or
//                              AMIO_ERR_INVALID_INPUT (by Error_Code)
//       - std::invalid_argument → AMIO_ERR_INVALID_INPUT
//       - std::exception        → AMIO_ERR_BACKEND_FAILURE
//       - catch(...)            → AMIO_ERR_BACKEND_FAILURE
//
//   * An OutcomeRegistry that records task outcomes against opaque
//     handles for later retrieval by flush/close/wait.
//
// Conditional compilation:
//   - MPI collective stack trace emission is gated on AMIO_HAS_MPI.
//     Without MPI, emit_parallel_stacktrace logs locally only.
//
// Thread safety
// -------------
// All public functions and OutcomeRegistry methods are safe to call
// concurrently from any thread.
//
// Validates: R12.1, R12.2, R12.3, R12.4, R12.9, R12.10

#ifndef AMIO_SRC_WORKERS_EXCEPTION_BRIDGE_HPP
#define AMIO_SRC_WORKERS_EXCEPTION_BRIDGE_HPP

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "amio/amio_errors.h"
#include "workers/comm_split.hpp"

// Forward declaration -- avoids pulling in <logs/logger.hpp> from this
// internal header.  The full definition is only needed in the .cpp.
namespace logs {
class Logger;
}

namespace amio::detail {

// TaskOutcome -- records the result of a worker task execution.
//
// On success: error_code == AMIO_OK, message is empty.
// On failure: error_code is a non-zero AMIO_ERR_*, message contains
//             the exception what() string, and stack_trace contains
//             the parallel stack trace (if available).
struct TaskOutcome {
    amio_err_t error_code = AMIO_OK;
    std::string message;      // Exception what() or description
    std::string stack_trace;  // Parallel stack trace (R12.10)
    bool is_failure() const noexcept {
        return error_code != AMIO_OK;
    }
};

// translate_exception_to_error -- maps a caught exception to an
// AMIO_ERR_* code.
//
// This function is called from within a catch block.  It examines
// the current exception and returns the appropriate error code.
//
// The catch hierarchy is ordered from most-specific to least-specific:
//   1. conf::Conf_Error   → mapped by Error_Code to AMIO_ERR_MANIFEST_*
//                           or AMIO_ERR_INVALID_INPUT
//   2. std::invalid_argument → AMIO_ERR_INVALID_INPUT
//   3. std::exception        → AMIO_ERR_BACKEND_FAILURE
//   4. catch(...)            → AMIO_ERR_BACKEND_FAILURE
//
// Parameters:
//   out_message - if non-null, receives the exception's what() string
//
// Returns: the AMIO_ERR_* code for the current exception.
amio_err_t translate_exception_to_error(std::string *out_message = nullptr);

// emit_parallel_stacktrace -- collective stack trace emission on
// the I/O communicator.
//
// This function MUST be called BEFORE recording the task outcome
// (R12.3, R12.4).  It ensures that the parallel stack trace is
// fully emitted before any API call can surface the failure.
//
// When AMIO_HAS_MPI is defined, this performs a collective operation
// on the I/O communicator to gather stack traces from all ranks.
// Without MPI, it logs the local stack trace only.
//
// Diagnostics routing (Req 6.1, 6.4, 6.7):
//   If `logger` is non-null (meaning LOGS is initialized), the
//   diagnostic is emitted via logs::Logger::log at FATAL severity
//   for unrecoverable errors, ERROR severity otherwise.
//   If `logger` is null (pre-initialization), falls back to
//   fprintf(stderr, ...).
//
// Parameters:
//   io_comm     - the I/O communicator for collective operations
//   error_code  - the AMIO_ERR_* code being reported
//   message     - the exception message
//   logger      - optional pointer to the LOGS Logger instance;
//                 nullptr means LOGS is not yet initialized (Req 6.7)
//
// Returns: the formatted stack trace string (retained for R12.10).
std::string emit_parallel_stacktrace(const IOCommunicator &io_comm, amio_err_t error_code, const std::string &message,
                                     logs::Logger *logger = nullptr);

// OutcomeRegistry -- thread-safe registry of task outcomes keyed by
// opaque handle identifier.
//
// Worker threads record outcomes here after exception cordon
// processing.  The C-boundary layer queries outcomes on flush/close/
// wait to surface failures to the host application (R12.9).
//
// Outcomes are retained for the AMIO_Core lifetime (R12.10) and
// released on finalization.
class OutcomeRegistry {
   public:
    // Record a task outcome against a handle identifier.
    //
    // If the handle already has a recorded failure, the new outcome
    // is appended (multiple failures can accumulate against a single
    // dataset handle).
    void record(std::uint64_t handle_id, TaskOutcome outcome);

    // Query whether a handle has any recorded failures.
    bool has_failure(std::uint64_t handle_id) const;

    // Retrieve all outcomes for a handle (empty vector if none).
    std::vector<TaskOutcome> get_outcomes(std::uint64_t handle_id) const;

    // Retrieve the first (oldest) failure for a handle.
    // Returns a default TaskOutcome with AMIO_OK if no failure exists.
    TaskOutcome get_first_failure(std::uint64_t handle_id) const;

    // Clear all outcomes for a handle (called on finalization).
    void clear(std::uint64_t handle_id);

    // Clear all outcomes (called on AMIO_Core finalization).
    void clear_all();

    // Count of handles with at least one recorded failure.
    std::size_t failure_count() const;

   private:
    mutable std::mutex mu_;
    std::unordered_map<std::uint64_t, std::vector<TaskOutcome>> outcomes_;
};

// execute_with_exception_cordon -- wraps a task callback in the
// full exception cordon.
//
// This is the primary entry point for worker threads.  It:
//   1. Executes the callback inside try/catch
//   2. On exception: translates to AMIO_ERR_*
//   3. Calls emit_parallel_stacktrace (collective) BEFORE recording
//   4. Records the outcome in the registry
//   5. Returns the outcome (caller uses this to decide buffer release)
//
// Parameters:
//   callback   - the task body to execute
//   handle_id  - the originating opaque handle identifier
//   io_comm    - the I/O communicator for parallel stack trace
//   registry   - the outcome registry to record against
//   logger     - optional pointer to the LOGS Logger instance;
//                nullptr means LOGS is not yet initialized (Req 6.7)
//
// Returns: the TaskOutcome (AMIO_OK on success, error code on failure)
TaskOutcome execute_with_exception_cordon(const std::function<void()> &callback, std::uint64_t handle_id, const IOCommunicator &io_comm,
                                          OutcomeRegistry &registry, logs::Logger *logger = nullptr);

}  // namespace amio::detail

#endif  // AMIO_SRC_WORKERS_EXCEPTION_BRIDGE_HPP
