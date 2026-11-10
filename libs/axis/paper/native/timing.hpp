// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>

#include "records.hpp"
#include "result_writer.hpp"
namespace paper {
using Clock = std::chrono::steady_clock;
// Preserve completed samples if an engine throws during a later repetition.
// The failure marker has no successful duration and is excluded from summaries.
class ObservationJournal {
   public:
    ObservationJournal(std::filesystem::path path, std::vector<Observation> &samples)
        : path_(std::move(path)), samples_(samples) {}
    ~ObservationJournal() noexcept {
        try {
            if (std::uncaught_exceptions() > 0)
                samples_.push_back({-1, "worker_failure", 0, 0.0, "failed"});
            observations(path_, samples_);
        } catch (const std::exception &error) {
            std::cerr << "Could not preserve timing observations: " << error.what() << '\n';
        }
    }

   private:
    std::filesystem::path path_;
    std::vector<Observation> &samples_;
};

template <class Fn, class Fence>
double elapsed(Fn fn, Fence fence) {
    fence();
    auto t = Clock::now();
    fn();
    fence();
    return std::chrono::duration<double>(Clock::now() - t).count();
}
// Reuse measurements exclude mesh preparation and weight generation. Fence both
// boundaries so asynchronous backends cannot move work outside the interval.
template <class Apply, class Fence>
void measure_reused_operator(Apply apply, Fence fence, bool timing, int warmups, int repetitions,
                             const std::vector<int> &reuse_sizes,
                             std::vector<Observation> &observations) {
    for (int iteration = 0; iteration < warmups; ++iteration)
        apply();
    if (!timing) {
        apply();
        return;
    }
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        observations.push_back({repetition, "apply", 1, elapsed(apply, fence)});
        for (int count : reuse_sizes) {
            const double seconds = elapsed(
                [&] {
                    for (int iteration = 0; iteration < count; ++iteration)
                        apply();
                },
                fence);
            observations.push_back({repetition, "repeated_workload", count, seconds});
        }
    }
}

inline std::vector<int> reuse_counts(const Args &a) {
    std::vector<int> n;
    std::istringstream ss(a.get("reuse", "1,100"));
    std::string token;
    while (std::getline(ss, token, ',')) {
        int k = std::stoi(token);
        if (k < 1)
            throw Error(2, "Invalid reuse count");
        n.push_back(k);
    }
    if (n.empty())
        throw Error(2, "Empty reuse");
    return n;
}
inline std::filesystem::path run_directory(const Args &a) {
    std::filesystem::path p = a.require("output");
    if (std::filesystem::exists(p))
        throw Error(2, "Output exists: choose new run ID");
    std::filesystem::create_directories(p);
    return p;
}
}  // namespace paper
