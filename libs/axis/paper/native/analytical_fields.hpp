// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>

#include "records.hpp"

namespace paper {
// Mathematical field definitions transcribed from Appendix A, Figures A1–A4,
// Valcke, Piacentini and Jonville (2022), doi:10.3390/mca27020031 (CC BY 4.0).
// Coordinates are unit-sphere Cartesian vectors; angles below are radians.
inline bool valcke_field(const std::string &name) {
    return name == "sinusoid" || name == "harmonic" || name == "vortex" || name == "gulfstream";
}
inline double field_bound(const std::string &name) {
    return valcke_field(name) ? 5.0 : name == "smooth" ? 3.0 : 1.0;
}
inline double field_lower_bound(const std::string &name) {
    return name == "vortex"     ? 0.0
           : valcke_field(name) ? 1.0
           : name == "zero"     ? -1.0
           : name == "sharp"    ? 0.0
                                : 1.0;
}
template <typename Real>
Real analytical_value(const std::array<Real, 3> &point, const std::string &name) {
    const Real pi = std::numbers::pi_v<Real>;
    if (name == "sinusoid")
        return 2 - std::cos(std::acos(std::clamp(point[0], Real(-1), Real(1))) / Real(1.2));
    if (name == "harmonic") {
        // sin(2*latitude)^16 cos(16*longitude), evaluated algebraically.
        Real radial = std::hypot(point[0], point[1]);
        if (radial < 1e-20)
            return 2;
        Real real = point[0] / radial, imaginary = point[1] / radial;
        for (int power = 0; power < 4; ++power) {
            Real next = real * real - imaginary * imaginary;
            imaginary = 2 * real * imaginary;
            real = next;
        }
        Real envelope = 2 * point[2] * radial;
        for (int power = 0; power < 4; ++power)
            envelope *= envelope;
        return 2 + envelope * real;
    }
    if (name == "vortex") {
        const Real central_lon = 5.5, central_lat = 0.2;
        const Real along = point[0] * std::cos(central_lon) + point[1] * std::sin(central_lon);
        const Real rotated_x = std::sin(central_lat) * along - std::cos(central_lat) * point[2];
        const Real rotated_y = point[1] * std::cos(central_lon) - point[0] * std::sin(central_lon);
        const Real radial = std::hypot(rotated_x, rotated_y), rho = 3 * radial;
        const Real omega = rho > 1e-20 ? (3 * std::sqrt(Real(3)) / 2) * std::tanh(rho) /
                                             (rho * std::cosh(rho) * std::cosh(rho))
                                       : 3 * std::sqrt(Real(3)) / 2;
        const Real rotated_sine =
            radial > 1e-20
                ? (rotated_y * std::cos(6 * omega) - rotated_x * std::sin(6 * omega)) / radial
                : 0;
        return 2 * (1 + std::tanh(rho / 5 * rotated_sine));
    }
    const Real lon = std::atan2(point[1], point[0]);
    const Real lat = std::asin(std::clamp(point[2], Real(-1), Real(1)));
    const Real sinusoid =
        2 - std::cos(std::acos(std::clamp(point[0], Real(-1), Real(1))) / Real(1.2));
    if (name == "gulfstream") {
        const Real radians = pi / 180;
        const Real dx = lon + 80 * radians, dy = lat - 25 * radians;
        const Real radius = std::hypot(dx, dy), angle = std::atan2(dy, dx);
        const Real end = std::hypot(Real(78.2), Real(25)) * radians;
        const Real damping = std::hypot(Real(54.5), Real(30.5)) * radians;
        Real envelope = radius > end ? 0 : 1.3;
        if (radius <= end && radius > damping)
            envelope *= std::cos(pi / 2 * (radius - damping) / (end - damping));
        const Real phase = Real(.4) * (Real(.5) * radius + angle) +
                           Real(.007) * std::cos(50 * angle) + Real(.37) * pi;
        const Real wave = 1000 * std::sin(phase);
        return sinusoid + std::max(Real(0), wave - 999) * envelope;
    }
    throw Error(2, "Unknown Valcke analytical field: " + name);
}
}  // namespace paper
