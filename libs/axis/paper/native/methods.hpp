// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "records.hpp"
namespace paper {
inline void validate_method(const std::string &m) {
    if (m != "nearest" && m != "bilinear" && m != "bicubic" && m != "patch" &&
        m != "conservative_first_order" && m != "conservative_second_order" &&
        m != "conservative_second_order_limited")
        throw Error(2, "Unknown paper method: " + m);
}
inline bool conservative_method(const std::string &m) {
    validate_method(m);
    return m.rfind("conservative_", 0) == 0;
}
inline bool second_order_method(const std::string &m) {
    return m.rfind("conservative_second_order", 0) == 0;
}
}  // namespace paper
