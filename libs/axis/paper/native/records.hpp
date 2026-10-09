// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace paper {
struct Error : std::runtime_error {
    int code;
    Error(int c, const std::string &s) : std::runtime_error(s), code(c) {}
};
struct Mesh {
    std::vector<double> lon, lat, center_lon, center_lat, area;
    std::vector<int> offsets, indices, mask, ids, parent;
    std::size_t size() const {
        return offsets.size() - 1;
    }
};
struct Case {
    Mesh src, dst;
    std::string id, geometry, field, norm;
    double tolerance = 1e-9, area_tolerance = 1e-10;
    double constant_tolerance = 1e-10, conservation_tolerance = 1e-10;
    double near_zero_tolerance = 1e-12, reproduction_tolerance = 1e-10;
    double smooth_rms_tolerance = 0.1;
    int depth = 20, budget = 1000000;
};
struct Reference {
    std::vector<double> src, dst, src_area, dst_area, src_covered, dst_covered, uncertainty;
    std::vector<double> precision_difference, area_uncertainty, roundoff_discarded_area,
        source_uncertainty;
};
struct Output {
    std::vector<double> src, dst, src_area, dst_area, src_frac, dst_frac, row_sum;
};
struct Observation {
    int repetition;
    std::string phase;
    int applications;
    double seconds;
    std::string status = "complete";
};
struct Args {
    std::map<std::string, std::string> values;
    Args(int argc, char **argv) {
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0)
                throw Error(2, "Expected --option value");
            if (!values.emplace(argv[i] + 2, argv[i + 1]).second)
                throw Error(2, "Duplicate option");
        }
    }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto i = values.find(k);
        return i == values.end() ? d : i->second;
    }
    std::string require(const std::string &k) const {
        auto s = get(k);
        if (s.empty())
            throw Error(2, "Missing --" + k);
        return s;
    }
    int count(const std::string &k, int d) const {
        const auto text = get(k, std::to_string(d));
        size_t consumed = 0;
        int value;
        try {
            value = std::stoi(text, &consumed);
        } catch (const std::exception &) {
            throw Error(2, "Invalid integer --" + k);
        }
        if (consumed != text.size() || value < 1)
            throw Error(2, "Expected positive integer --" + k);
        return value;
    }
};
}  // namespace paper
