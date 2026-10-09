// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstdint>
namespace axis::solver {
// Host-only paper diagnostics. Null by default; no allocations or atomics.
// Scope is thread-local to avoid altering unrelated callers' state.
struct PaperDiagnostics {
    const char *path = "unknown";
    std::int64_t candidates = -1, invalid = -1, zero_overlap = -1, discarded = -1;
};
inline thread_local PaperDiagnostics *paper_diagnostics = nullptr;
inline void paper_path(const char *path) {
    if (paper_diagnostics) paper_diagnostics->path = path;
}
// Call only on serial host paths. Unknown parallel/device counters remain -1.
inline void paper_overlap_begin() {
    if (paper_diagnostics) {
        paper_diagnostics->candidates = 0;
        paper_diagnostics->invalid = 0;
        paper_diagnostics->zero_overlap = 0;
        paper_diagnostics->discarded = 0;
    }
}
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
