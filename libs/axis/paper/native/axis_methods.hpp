// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <axis/solver/regrid_config.hpp>

#include "methods.hpp"
namespace paper {
inline void select_axis_method(axis::solver::RegridConfig &config, const std::string &method) {
    validate_method(method);
    using M = axis::solver::InterpolationMethod;
    config.method = method == "nearest"           ? M::NearestNeighbor
                    : method == "bilinear"        ? M::Bilinear
                    : method == "bicubic"         ? M::Bicubic
                    : method == "patch"           ? M::Patch
                    : second_order_method(method) ? M::Conservative2ndOrder
                                                  : M::Conservative1stOrder;
    config.use_limiter = method == "conservative_second_order_limited";
}
}  // namespace paper
