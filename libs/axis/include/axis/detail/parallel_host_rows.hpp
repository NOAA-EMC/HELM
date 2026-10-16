// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <Kokkos_Core.hpp>
#include <exception>
#include <vector>
namespace axis::detail {
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
