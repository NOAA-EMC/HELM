#include "operators/precision_conversion.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <utility>

namespace amio::detail {

amio_status_t convert_f64_to_f32(std::span<const double> input, const PrecisionConversionPolicy &policy, PrecisionConversionResult &result) {
    if (!std::isfinite(policy.absolute_tolerance) || !std::isfinite(policy.relative_tolerance) || policy.absolute_tolerance < 0.0 ||
        policy.relative_tolerance < 0.0 || ((policy.absolute_tolerance == 0.0) == (policy.relative_tolerance == 0.0)) ||
        (policy.missing_sentinel && !std::isfinite(*policy.missing_sentinel)))
        return AMIO_ERR_INVALID_INPUT;
    PrecisionConversionResult converted;
    converted.input_bytes = static_cast<std::uint64_t>(input.size()) * sizeof(double);
    converted.output_bytes = static_cast<std::uint64_t>(input.size()) * sizeof(float);
    converted.values.reserve(input.size());
    for (const double original : input) {
        if (policy.missing_sentinel && original == *policy.missing_sentinel) {
            const float sentinel = static_cast<float>(original);
            if (!std::isfinite(sentinel) || static_cast<double>(sentinel) != original) return AMIO_ERR_INVALID_INPUT;
            converted.values.push_back(sentinel);
            continue;
        }
        if (std::isnan(original)) {
            if (!policy.canonicalize_nan) return AMIO_ERR_UNSUPPORTED;
            converted.values.push_back(std::bit_cast<float>(std::uint32_t{0x7fc00000u}));
            continue;
        }
        const float narrowed = static_cast<float>(original);
        if (std::isinf(original)) {
            if (!std::isinf(narrowed) || std::signbit(original) != std::signbit(narrowed)) return AMIO_ERR_INVALID_INPUT;
            converted.values.push_back(narrowed);
            continue;
        }
        if (!std::isfinite(narrowed)) return AMIO_ERR_INVALID_INPUT;
        if (original == 0.0 && std::signbit(original) != std::signbit(narrowed)) return AMIO_ERR_INVALID_INPUT;
        const double error = std::abs(static_cast<double>(narrowed) - original);
        const double allowed = policy.absolute_tolerance > 0.0 ? policy.absolute_tolerance : policy.relative_tolerance * std::abs(original);
        if (error > allowed) return AMIO_ERR_INVALID_INPUT;
        if (original != 0.0 && narrowed == 0.0f && policy.absolute_tolerance == 0.0) return AMIO_ERR_INVALID_INPUT;
        converted.maximum_absolute_error = std::max(converted.maximum_absolute_error, error);
        converted.values.push_back(narrowed);
    }
    result = std::move(converted);
    return AMIO_OK;
}

}  // namespace amio::detail
