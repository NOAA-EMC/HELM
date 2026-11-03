#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#include "operators/precision_conversion.hpp"

using namespace amio::detail;

int main() {
    const std::vector<double> input{1.0, -0.0, std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::quiet_NaN()};
    PrecisionConversionPolicy policy;
    policy.absolute_tolerance = 1e-6;
    policy.canonicalize_nan = true;
    PrecisionConversionResult result;
    assert(convert_f64_to_f32(input, policy, result) == AMIO_OK);
    assert(result.values.size() == input.size());
    assert(std::signbit(result.values[1]));
    assert(std::isinf(result.values[3]));
    assert(std::isnan(result.values[4]));
    assert(result.values[4] == std::numeric_limits<float>::quiet_NaN() || std::isnan(result.values[4]));

    policy.absolute_tolerance = 0.0;
    assert(convert_f64_to_f32(std::vector<double>{1.0 + 1e-8}, policy, result) == AMIO_ERR_INVALID_INPUT);
    policy.absolute_tolerance = 0.0;
    policy.relative_tolerance = 0.1;
    assert(convert_f64_to_f32(std::vector<double>{0.0}, policy, result) == AMIO_OK);
    assert(convert_f64_to_f32(std::vector<double>{1e-320}, policy, result) == AMIO_ERR_INVALID_INPUT);

    policy.relative_tolerance = 0.0;
    policy.absolute_tolerance = 1e-3;
    policy.missing_sentinel = -9999.0;
    assert(convert_f64_to_f32(std::vector<double>{-9999.0}, policy, result) == AMIO_OK);
    policy.missing_sentinel = 1.0 / 3.0;
    assert(convert_f64_to_f32(std::vector<double>{*policy.missing_sentinel}, policy, result) == AMIO_ERR_INVALID_INPUT);
    assert(convert_f64_to_f32(std::vector<double>{std::numeric_limits<double>::max()}, policy, result) == AMIO_ERR_INVALID_INPUT);
}
