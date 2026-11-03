#ifndef AMIO_TESTS_QUALIFICATION_REFERENCE_FIELDS_HPP
#define AMIO_TESTS_QUALIFICATION_REFERENCE_FIELDS_HPP

#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <string>

namespace amio::qualification {

struct ReferenceWorkload {
    std::uint64_t steps;
    std::uint64_t fields;
    std::uint64_t cells;
};

inline float reference_value(std::uint64_t step, std::uint64_t field, std::uint64_t cell) {
    // Keep the integer range small so every generated value is finite and
    // exactly reproducible when written with max_digits10 precision.
    const auto code = (step * 131U + field * 17U + cell * 7U) % 20001U;
    return static_cast<float>(static_cast<std::int64_t>(code) - 10000) / 100.0F;
}

inline bool valid(const ReferenceWorkload &workload) {
    return workload.steps > 0 && workload.fields > 0 && workload.cells > 0;
}

inline void write_reference(std::ostream &out, const ReferenceWorkload &workload) {
    out << "AMIO_REFERENCE_V1\t" << workload.steps << '\t' << workload.fields << '\t' << workload.cells << '\n';
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::uint64_t step = 0; step < workload.steps; ++step) {
        for (std::uint64_t field = 0; field < workload.fields; ++field) {
            for (std::uint64_t cell = 0; cell < workload.cells; ++cell) {
                out << step << '\t' << field << '\t' << cell << '\t' << reference_value(step, field, cell) << '\n';
            }
        }
    }
}

inline bool verify_reference(std::istream &in, const ReferenceWorkload &expected, std::string &error) {
    std::string magic;
    ReferenceWorkload actual{};
    if (!(in >> magic >> actual.steps >> actual.fields >> actual.cells) || magic != "AMIO_REFERENCE_V1") {
        error = "invalid reference header";
        return false;
    }
    if (actual.steps != expected.steps || actual.fields != expected.fields || actual.cells != expected.cells) {
        error = "reference workload dimensions do not match";
        return false;
    }

    for (std::uint64_t step = 0; step < expected.steps; ++step) {
        for (std::uint64_t field = 0; field < expected.fields; ++field) {
            for (std::uint64_t cell = 0; cell < expected.cells; ++cell) {
                std::uint64_t found_step = 0;
                std::uint64_t found_field = 0;
                std::uint64_t found_cell = 0;
                float value = 0.0F;
                if (!(in >> found_step >> found_field >> found_cell >> value)) {
                    error = "reference payload ended before all expected values";
                    return false;
                }
                if (found_step != step || found_field != field || found_cell != cell || value != reference_value(step, field, cell)) {
                    error = "reference payload differs at step " + std::to_string(step) + ", field " + std::to_string(field) + ", cell " +
                            std::to_string(cell);
                    return false;
                }
            }
        }
    }
    std::string trailing;
    if (in >> trailing) {
        error = "reference payload has trailing data";
        return false;
    }
    return true;
}

}  // namespace amio::qualification

#endif  // AMIO_TESTS_QUALIFICATION_REFERENCE_FIELDS_HPP
