// SPDX-License-Identifier: Apache-2.0
#pragma once

/// @file axis/detail/parallel_host_rows.hpp
/// @brief Host-row parallel-for helper with deferred exception collection.

#include <Kokkos_Core.hpp>
#include <exception>
#include <vector>
namespace axis::detail {

/// @brief Run function(i) for each host row, in parallel when worthwhile.
///
/// Falls back to a serial loop below parallel_minimum rows or when the host
/// execution space has concurrency 1. Exceptions thrown inside function are
/// captured per row and rethrown after the parallel region completes, so a
/// failing row never leaves a detached Kokkos kernel running.
/// @tparam Function Callable accepting the row index (std::size_t).
/// @param label            Kokkos parallel region name.
/// @param count            Number of rows.
/// @param function         Per-row body; must be thread-safe.
/// @param parallel_minimum Row count below which execution stays serial.
template <class Function>
void parallel_host_rows(const char *label, std::size_t count, Function function, std::size_t parallel_minimum = 256) {
    if (count < parallel_minimum || Kokkos::DefaultHostExecutionSpace().concurrency() == 1) {
        for (std::size_t i = 0; i < count; ++i) function(i);
        return;
    }
    std::vector<std::exception_ptr> errors(count);
    Kokkos::parallel_for(label, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, count), [&](std::size_t i) {
        try {
            function(i);
        } catch (...) {
            errors[i] = std::current_exception();
        }
    });
    Kokkos::fence("axis::host_rows::complete");
    for (const auto &error : errors)
        if (error) std::rethrow_exception(error);
}
}  // namespace axis::detail
