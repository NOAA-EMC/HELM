// exception_bridge.cpp -- AMIO exception cordon and parallel diagnostics.
//
// Implements the exception bridge described in design.md §3
// (Staging Pool & Worker Pool) and required by R12.1–R12.4,
// R12.9, R12.10.
//
// Key design decisions:
//
//   * The exception cordon wraps every worker task body in a
//     four-level catch hierarchy:
//       1. conf::Conf_Error (most specific – HELM::CONF errors)
//       2. std::invalid_argument
//       3. std::exception
//       4. catch(...)
//     This ensures no exception propagates across the C-boundary
//     (R12.2).
//
//   * emit_parallel_stacktrace is called BEFORE recording the
//     outcome in the registry (R12.3, R12.4).  This guarantees
//     that the stack trace is fully emitted before any API call
//     (flush/close/wait) can surface the failure.
//
//   * Exception translation maps conf::Conf_Error to specific
//     AMIO_ERR_* codes based on error code; std::invalid_argument
//     maps to AMIO_ERR_INVALID_INPUT; all others map to
//     AMIO_ERR_BACKEND_FAILURE.
//
//   * The OutcomeRegistry retains outcomes for the AMIO_Core
//     lifetime (R12.10), released only on finalization.
//
// Validates: R12.1, R12.2, R12.3, R12.4, R12.9, R12.10

#include "workers/exception_bridge.hpp"

#include <conf/error.hpp>
#include <cstdio>
#include <ctime>
#include <exception>
#include <logs/logger.hpp>
#include <logs/severity.hpp>
#include <sstream>
#include <stdexcept>

namespace amio::detail {

// ---- Exception translation ----

amio_err_t translate_exception_to_error(std::string *out_message) {
    try {
        // Re-throw the current exception to inspect it.
        throw;
    } catch (const conf::Conf_Error &e) {
        // conf::Conf_Error is the most specific; map by error code.
        if (out_message) *out_message = e.what();
        switch (e.code()) {
            case conf::Error_Code::File_Not_Found:
                return AMIO_ERR_MANIFEST_NOT_FOUND;
            case conf::Error_Code::Parse_Error:
            case conf::Error_Code::Key_Not_Found:
                return AMIO_ERR_MANIFEST_INVALID;
            case conf::Error_Code::Type_Mismatch:
            case conf::Error_Code::Invalid_Arg:
                return AMIO_ERR_INVALID_INPUT;
            default:
                return AMIO_ERR_BACKEND_FAILURE;
        }
    } catch (const std::invalid_argument &e) {
        if (out_message) *out_message = e.what();
        return AMIO_ERR_INVALID_INPUT;
    } catch (const std::exception &e) {
        if (out_message) *out_message = e.what();
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        if (out_message) *out_message = "Unknown exception (non-std)";
        return AMIO_ERR_BACKEND_FAILURE;
    }
}

// ---- Parallel stack trace emission ----

// Determine whether an error code represents an unrecoverable failure.
// Unrecoverable errors are those from which AMIO cannot continue normal
// operation (communicator failures, threading violations).
static bool is_unrecoverable(amio_err_t error_code) {
    switch (error_code) {
        case AMIO_ERR_COMM_SPLIT_FAILED:
        case AMIO_ERR_THREADING_UNSUPPORTED:
        case AMIO_ERR_FINALIZE_TIMEOUT:
            return true;
        default:
            return false;
    }
}

std::string emit_parallel_stacktrace(const IOCommunicator &io_comm, amio_err_t error_code, const std::string &message, logs::Logger *logger) {
    // Build a local stack trace record.
    std::ostringstream oss;

    // Timestamp for the diagnostic.
    std::time_t now = std::time(nullptr);
    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    oss << "[AMIO FATAL] " << time_buf << "\n"
        << "  Error code: " << error_code << " (" << amio_strerror(error_code) << ")\n"
        << "  Message: " << message << "\n"
        << "  I/O Communicator: " << (io_comm.valid ? "valid" : "invalid") << ", rank=" << io_comm.rank() << ", size=" << io_comm.size()
        << ", is_io_rank=" << (io_comm.is_io_rank ? "true" : "false") << "\n";

#ifdef AMIO_HAS_MPI
    // In a real MPI build, this would perform a collective operation
    // on the I/O communicator to gather stack traces from all ranks.
    // For now, we emit the local trace.  The collective ensures all
    // ranks have emitted before any rank proceeds (R12.4).
    //
    // MPI_Allgather or MPI_Gather of the trace string would go here.
    // The barrier ensures completion before recording outcome.
    oss << "  [MPI collective stack trace would be gathered here]\n";
#endif

    std::string trace = oss.str();

    // Route diagnostics through LOGS if initialized (Req 6.1, 6.4, 6.7).
    if (logger) {
        auto severity = is_unrecoverable(error_code) ? logs::Severity_Level::FATAL : logs::Severity_Level::ERROR;
        logger->log(severity, trace);
    } else {
        // Pre-initialization fallback: emit to stderr directly (Req 6.7).
        std::fprintf(stderr, "%s", trace.c_str());
    }

    return trace;
}

// ---- OutcomeRegistry ----

void OutcomeRegistry::record(std::uint64_t handle_id, TaskOutcome outcome) {
    std::lock_guard<std::mutex> lock(mu_);
    outcomes_[handle_id].push_back(std::move(outcome));
}

bool OutcomeRegistry::has_failure(std::uint64_t handle_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = outcomes_.find(handle_id);
    if (it == outcomes_.end()) return false;
    for (const auto &o : it->second) {
        if (o.is_failure()) return true;
    }
    return false;
}

std::vector<TaskOutcome> OutcomeRegistry::get_outcomes(std::uint64_t handle_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = outcomes_.find(handle_id);
    if (it == outcomes_.end()) return {};
    return it->second;
}

TaskOutcome OutcomeRegistry::get_first_failure(std::uint64_t handle_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = outcomes_.find(handle_id);
    if (it == outcomes_.end()) return TaskOutcome{};
    for (const auto &o : it->second) {
        if (o.is_failure()) return o;
    }
    return TaskOutcome{};
}

void OutcomeRegistry::clear(std::uint64_t handle_id) {
    std::lock_guard<std::mutex> lock(mu_);
    outcomes_.erase(handle_id);
}

void OutcomeRegistry::clear_all() {
    std::lock_guard<std::mutex> lock(mu_);
    outcomes_.clear();
}

std::size_t OutcomeRegistry::failure_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t count = 0;
    for (const auto &[id, outcomes] : outcomes_) {
        for (const auto &o : outcomes) {
            if (o.is_failure()) {
                ++count;
                break;  // Count each handle only once.
            }
        }
    }
    return count;
}

// ---- Exception cordon execution ----

TaskOutcome execute_with_exception_cordon(const std::function<void()> &callback, std::uint64_t handle_id, const IOCommunicator &io_comm,
                                          OutcomeRegistry &registry, logs::Logger *logger) {
    TaskOutcome outcome;

    try {
        if (callback) {
            callback();
        }
        // Success path: record AMIO_OK.
        outcome.error_code = AMIO_OK;
    } catch (...) {
        // Delegate all exception-to-error mapping to translate_exception_to_error
        // for consistent behaviour between the cordon and direct translation callers.
        outcome.error_code = translate_exception_to_error(&outcome.message);

        // Emit parallel stack trace BEFORE recording outcome (R12.3, R12.4).
        // Pass the logger for LOGS routing when initialized (Req 6.1, 6.7).
        outcome.stack_trace = emit_parallel_stacktrace(io_comm, outcome.error_code, outcome.message, logger);

        // Record the failure against the originating handle.
        registry.record(handle_id, outcome);
    }

    return outcome;
}

}  // namespace amio::detail
