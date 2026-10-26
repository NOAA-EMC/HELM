// SPDX-License-Identifier: Apache-2.0
#pragma once

/// @file axis/solver/paper_diagnostics.hpp
/// @brief Opt-in solver-path and overlap diagnostics for evidence collection.

#include <cmath>
#include <cstdint>
namespace axis::solver {

/// @brief Snapshot of one weight-generation path's dispatch and overlap counts.
///
/// Host-only paper diagnostics. Null by default; no allocations or atomics.
/// Scope is thread-local to avoid altering unrelated callers' state.
/// Counters use -1 for "unknown" (for example on parallel or device paths);
/// zero means "measured and genuinely zero", never "not attempted".
struct PaperDiagnostics {
    const char *path = "unknown";
    std::int64_t candidates = -1, invalid = -1, zero_overlap = -1, discarded = -1;
};
/// @brief Thread-local active diagnostics sink; nullptr disables instrumentation.
inline thread_local PaperDiagnostics *paper_diagnostics = nullptr;

/// @brief Record which solver path was selected into the active sink, if any.
inline void paper_path(const char *path) {
    if (paper_diagnostics) paper_diagnostics->path = path;
}
/// @brief Reset the overlap counters of the active sink to zero.
///
/// Call only on serial host paths. Unknown parallel/device counters remain -1.
inline void paper_overlap_begin() {
    if (paper_diagnostics) {
        paper_diagnostics->candidates = 0;
        paper_diagnostics->invalid = 0;
        paper_diagnostics->zero_overlap = 0;
        paper_diagnostics->discarded = 0;
    }
}
/// @brief Classify one computed candidate overlap area into the active sink.
///
/// Increments the candidate counter and classifies the area as invalid
/// (non-finite), a valid exact-zero overlap, discarded (a roundoff sliver at
/// or below threshold), or kept. A null sink makes this a no-op.
/// @param area      Overlap area produced by the clipper.
/// @param threshold Nonzero areas at or below this value count as discarded.
inline void paper_overlap_observation(double area, double threshold = 0.0) {
    if (!paper_diagnostics) return;
    ++paper_diagnostics->candidates;
    if (!std::isfinite(area))
        ++paper_diagnostics->invalid;
    else if (area == 0.0)
        ++paper_diagnostics->zero_overlap;
    else if (area <= threshold)
        ++paper_diagnostics->discarded;
}
}  // namespace axis::solver
