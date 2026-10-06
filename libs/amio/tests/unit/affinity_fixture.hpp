// Test-only Linux affinity snapshot/restore support.  CPU counts are not IDs.
#ifndef AMIO_TESTS_UNIT_AFFINITY_FIXTURE_HPP
#define AMIO_TESTS_UNIT_AFFINITY_FIXTURE_HPP

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace amio_test {

inline std::vector<int> cpu_ids(const cpu_set_t &mask) {
    std::vector<int> ids;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            ids.push_back(cpu);
        }
    }
    return ids;
}

// A failed query returns an empty vector, which must fail an assertion rather
// than turn an affinity test into a skip.  Linux cannot run with an empty mask.
inline std::vector<int> current_cpu_ids() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return {};
    }
    return cpu_ids(mask);
}

class AffinityGuard {
   public:
    using FailureReporter = void (*)(const std::string &);

    explicit AffinityGuard(FailureReporter report) : report_(report) {
        CPU_ZERO(&original_);
        if (sched_getaffinity(0, sizeof(original_), &original_) != 0) {
            report_(std::string("sched_getaffinity failed: ") + std::strerror(errno));
            return;
        }
        allowed_ = cpu_ids(original_);
        valid_ = !allowed_.empty();
        if (!valid_) {
            report_("sched_getaffinity returned an empty CPU mask");
        }
    }

    AffinityGuard(const AffinityGuard &) = delete;
    AffinityGuard &operator=(const AffinityGuard &) = delete;

    ~AffinityGuard() {
        if (!valid_) {
            return;
        }
        // Restore only the exact mask captured on entry, never 0..CPU-count-1.
        // In particular, do not add CPUs outside the inherited scheduler mask.
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_);
        if (rc != 0) {
            report_(std::string("restoring original affinity failed: ") + std::strerror(rc));
        } else if (current_cpu_ids() != allowed_) {
            report_("restored affinity does not match the original CPU mask");
        }
    }

    bool valid() const {
        return valid_;
    }
    const std::vector<int> &allowed() const {
        return allowed_;
    }

   private:
    FailureReporter report_;
    cpu_set_t original_;
    std::vector<int> allowed_;
    bool valid_ = false;
};

}  // namespace amio_test
#endif  // __linux__

#endif  // AMIO_TESTS_UNIT_AFFINITY_FIXTURE_HPP
