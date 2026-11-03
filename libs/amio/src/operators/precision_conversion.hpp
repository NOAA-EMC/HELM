#ifndef AMIO_SRC_OPERATORS_PRECISION_CONVERSION_HPP
#define AMIO_SRC_OPERATORS_PRECISION_CONVERSION_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

struct PrecisionConversionPolicy {
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
    std::optional<double> missing_sentinel;
    bool canonicalize_nan = false;
};

struct PrecisionConversionResult {
    std::vector<float> values;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    double maximum_absolute_error = 0.0;
};

[[nodiscard]] amio_status_t convert_f64_to_f32(std::span<const double> input, const PrecisionConversionPolicy &policy,
                                               PrecisionConversionResult &result);

}  // namespace amio::detail

#endif
