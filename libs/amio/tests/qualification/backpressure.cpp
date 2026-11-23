#include "backpressure.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "staging/resource_budget.hpp"
#include "workers/worker_pool.hpp"

namespace {
using namespace amio::detail;
constexpr std::size_t kPayloadBytes = 20u * 1024u * 1024u;
constexpr std::size_t kBudgetBytes = 64u * 1024u * 1024u;

bool parse_unsigned(const char *text, std::uint64_t &value) {
    try {
        std::size_t used = 0;
        const auto parsed = std::stoull(text, &used);
        if (used != std::string_view(text).size()) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

struct MeasuredHighWater {
    std::size_t data_bytes = 0, data_descriptors = 0, queued_writes = 0, control_bytes = 0;
    std::uint64_t completed_writes = 0;
};

bool run_policy(bool wait_policy, std::chrono::seconds delay, MeasuredHighWater &high) {
    ResourceBudget budget({kBudgetBytes, 8, 4096, 2});
    WorkerPoolConfig config;
    config.thread_count = 1;
    config.backpressure.enabled = wait_policy;
    config.backpressure.low_watermark = 0;
    config.backpressure.high_watermark = 1;
    config.backpressure.queue_capacity = 1;
    config.backpressure.admission_timeout = delay + std::chrono::seconds(10);
    config.backpressure.prefetch_capacity = 1;
    WorkerPool pool(config);
    std::atomic<bool> first_started{false};

    const auto submit_payload = [&](std::uint64_t id, std::chrono::milliseconds task_delay, bool block) {
        auto reservation = budget.reserve(kPayloadBytes, 1);
        if (!reservation) return AMIO_ERR_QUEUE_FULL;
        auto payload = std::make_shared<std::vector<std::byte>>(kPayloadBytes, std::byte{static_cast<unsigned char>(id)});
        std::uint64_t sequence = 0;
        const auto status = pool.submit_write(
            {1, id},
            [&, id, payload = std::move(payload), ticket = *reservation, task_delay, block]() mutable {
                if (id == 0) first_started.store(true, std::memory_order_release);
                if (block) std::this_thread::sleep_for(task_delay);
                (void)payload;
                budget.release(ticket);
            },
            &sequence);
        if (status != AMIO_OK) budget.release(*reservation);
        const auto peak = budget.high_water();
        high.data_bytes = std::max(high.data_bytes, peak.data_bytes);
        high.data_descriptors = std::max(high.data_descriptors, peak.data_descriptors);
        high.queued_writes = std::max(high.queued_writes, pool.write_queue_depth());
        high.control_bytes = std::max(high.control_bytes, peak.control_bytes);
        return status;
    };

    const auto held_for = std::max(std::chrono::milliseconds(200), std::chrono::duration_cast<std::chrono::milliseconds>(delay));
    if (submit_payload(0, held_for, true) != AMIO_OK) {
        std::cerr << "first admission failed\n";
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!first_started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    if (!first_started.load(std::memory_order_acquire)) {
        std::cerr << "worker did not start first task\n";
        return false;
    }
    if (submit_payload(1, std::chrono::milliseconds(0), false) != AMIO_OK) {
        std::cerr << "second admission failed\n";
        return false;
    }

    const auto third = submit_payload(2, std::chrono::milliseconds(0), false);
    if (wait_policy) {
        if (third != AMIO_OK) {
            std::cerr << "wait policy third admission status=" << third << '\n';
            return false;
        }
    } else if (third != AMIO_ERR_QUEUE_FULL) {
        std::cerr << "reject policy third admission status=" << third << '\n';
        return false;
    }
    pool.drain();
    high.completed_writes = pool.total_writes_completed();
    const std::uint64_t expected_completed = wait_policy ? 3 : 2;
    const auto usage = budget.snapshot();
    const auto peak = budget.high_water();
    if (usage.data_bytes != 0 || peak.data_bytes > kBudgetBytes || peak.data_descriptors > 3 || high.queued_writes > 1 ||
        high.completed_writes != expected_completed) {
        std::cerr << "budget final=" << usage.data_bytes << " peak=" << peak.data_bytes << " descriptors=" << peak.data_descriptors
                  << " queue=" << high.queued_writes << '\n';
        return false;
    }
    return true;
}
}  // namespace

int run_backpressure_fixture(int argc, char **argv) {
    std::uint64_t delay_seconds = 600;
    std::uint64_t budget_mib = 64;
    if (argc != 8 || std::string_view(argv[2]) != "--delay-seconds" || std::string_view(argv[4]) != "--budget-mib" ||
        std::string_view(argv[6]) != "--output" || !parse_unsigned(argv[3], delay_seconds) || !parse_unsigned(argv[5], budget_mib) ||
        budget_mib != 64 || delay_seconds > 86400 || argv[7][0] == '\0') {
        std::cerr << "Usage: amio_qualify backpressure --delay-seconds N --budget-mib 64 --output DIR\n";
        return 2;
    }
    const auto delay = std::chrono::seconds(delay_seconds);
    MeasuredHighWater reject_high, wait_high;
    bool reject_ok = false, wait_ok = false;
    std::thread reject_run([&] { reject_ok = run_policy(false, delay, reject_high); });
    std::thread wait_run([&] { wait_ok = run_policy(true, delay, wait_high); });
    reject_run.join();
    wait_run.join();
    if (!reject_ok || !wait_ok) {
        std::cerr << "backpressure fixture did not preserve queue or byte budgets\n";
        return 1;
    }
    std::cout << "bounded backpressure fixture passed; reject_bytes_high_water=" << reject_high.data_bytes
              << " wait_bytes_high_water=" << wait_high.data_bytes << " byte_limit=" << kBudgetBytes << " delay_seconds=" << delay_seconds
              << " policies=reject,wait\n";
    std::error_code ec;
    const std::filesystem::path output(argv[7]);
    std::filesystem::create_directories(output, ec);
    if (ec) return 5;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ofstream report(output / ("backpressure-" + std::to_string(stamp) + ".json"), std::ios::out | std::ios::binary);
    if (!report) return 5;
    report << "{\"schema_version\":1,\"status\":\"PASS\",\"delay_seconds\":" << delay_seconds << ",\"budget_bytes\":" << kBudgetBytes
           << ",\"policies\":{\"reject\":{\"data_bytes_high_water\":" << reject_high.data_bytes
           << ",\"data_descriptors_high_water\":" << reject_high.data_descriptors << ",\"control_bytes_high_water\":" << reject_high.control_bytes
           << ",\"write_queue_high_water\":" << reject_high.queued_writes << ",\"accepted_writes_drained\":" << reject_high.completed_writes
           << "},\"wait\":{\"data_bytes_high_water\":" << wait_high.data_bytes << ",\"data_descriptors_high_water\":" << wait_high.data_descriptors
           << ",\"control_bytes_high_water\":" << wait_high.control_bytes << ",\"write_queue_high_water\":" << wait_high.queued_writes
           << ",\"accepted_writes_drained\":" << wait_high.completed_writes << "}}}\n";
    if (!report) return 5;
    return 0;
}
