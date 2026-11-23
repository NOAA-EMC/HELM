#ifndef AMIO_BENCHMARKS_WORKLOAD_HPP
#define AMIO_BENCHMARKS_WORKLOAD_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace amio::benchmark {
struct Workload {
    std::uint64_t steps = 8;
    std::uint64_t fields = 2;
    std::uint64_t cells = 65536;
    std::uint64_t seed = 1729;
};

// Frozen, deliberately small key=value manifest format; unknown keys and
// duplicate values are rejected so runs cannot silently change semantics.
inline bool load_workload(const std::filesystem::path &path, Workload &workload, std::string &error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open workload manifest";
        return false;
    }
    Workload parsed{};
    bool steps = false, fields = false, cells = false, seed = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            error = "manifest rows must be key=value";
            return false;
        }
        const auto key = line.substr(0, equals);
        const auto value = line.substr(equals + 1);
        std::uint64_t number = 0;
        try {
            std::size_t used = 0;
            number = std::stoull(value, &used);
            if (used != value.size()) throw std::invalid_argument("tail");
        } catch (...) {
            error = "manifest values must be unsigned integers";
            return false;
        }
        bool *seen = nullptr;
        if (key == "steps") {
            parsed.steps = number;
            seen = &steps;
        } else if (key == "fields") {
            parsed.fields = number;
            seen = &fields;
        } else if (key == "cells") {
            parsed.cells = number;
            seen = &cells;
        } else if (key == "seed") {
            parsed.seed = number;
            seen = &seed;
        } else {
            error = "unknown manifest key: " + key;
            return false;
        }
        if (*seen) {
            error = "duplicate manifest key: " + key;
            return false;
        }
        *seen = true;
    }
    if (!steps || !fields || !cells || !seed || parsed.steps == 0 || parsed.fields == 0 || parsed.cells == 0 || parsed.steps > 100000 ||
        parsed.fields > 10000 || parsed.cells > 100000000 || parsed.cells > UINT64_MAX / parsed.fields / sizeof(float) / parsed.steps) {
        error = "manifest is incomplete or exceeds safe workload limits";
        return false;
    }
    workload = parsed;
    return true;
}

inline float value(std::uint64_t seed, std::uint64_t step, std::uint64_t field, std::uint64_t cell) {
    const std::uint64_t mixed = seed + step * 1000003ULL + field * 9176ULL + cell * 37ULL;
    return static_cast<float>(static_cast<std::int64_t>(mixed % 2000001ULL) - 1000000) / 100.0f;
}
}  // namespace amio::benchmark

#endif
