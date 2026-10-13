// SPDX-License-Identifier: Apache-2.0
#include <axis/solver/vertical_conservative.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using Op = axis::solver::VerticalConservativeRegridder<Kokkos::HostSpace>;
using Order = axis::solver::VerticalConservativeOrder;
using Norm = axis::solver::VerticalConservativeNorm;
static int example(const std::string &name, std::vector<double> se, std::vector<double> de,
                   std::vector<double> qv, bool mask = false) {
    Op::Field s("s", 1, se.size()), d("d", 1, de.size()), q("q", 1, qv.size()),
        v("v", 1, de.size() - 1);
    Kokkos::View<int **, Kokkos::HostSpace> sm("sm", 1, qv.size());
    Kokkos::deep_copy(sm, 1);
    for (size_t i = 0; i < se.size(); ++i)
        s(0, i) = se[i];
    for (size_t i = 0; i < de.size(); ++i)
        d(0, i) = de[i];
    for (size_t i = 0; i < qv.size(); ++i)
        q(0, i) = qv[i];
    if (mask)
        sm(0, 1) = 0;
    Op op(s, d, 1, sm);
    int failures = 0;
    for (auto order : {Order::First, Order::LimitedLinear})
        for (auto norm : {Norm::DestinationMeasure, Norm::CoveredMeasure}) {
            op.apply(q, v, order, norm);
            auto stock = op.covered_source_integral(q, order);
            double total = 0, minimum = INFINITY, maximum = -INFINITY;
            int mapped = 0;
            for (size_t j = 0; j < de.size() - 1; ++j) {
                double h = norm == Norm::CoveredMeasure ? op.destination_covered_measure()(0, j)
                                                        : std::abs(de[j + 1] - de[j]);
                total += v(0, j) * h;
                if (op.destination_coverage()(0, j) > 0) {
                    ++mapped;
                    minimum = std::fmin(minimum, v(0, j));
                    maximum = std::fmax(maximum, v(0, j));
                }
            }
            double residual = std::abs(total - stock(0));
            bool ok = std::isfinite(total) &&
                      residual <= 1e-13 * std::fmax(std::abs(stock(0)), 1e-12) && minimum >= -1e-15;
            if (!ok)
                ++failures;
            std::cout << name << ',' << (order == Order::First ? "first" : "limited_linear") << ','
                      << (norm == Norm::DestinationMeasure ? "dstarea" : "fracarea") << ','
                      << stock(0) << ',' << total << ',' << residual << ',' << minimum << ','
                      << maximum << ',' << mapped << ',' << (ok ? "pass" : "fail") << '\n';
        }
    return failures;
}
int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    int failures = 0;
    try {
        std::cout << std::setprecision(17)
                  << "case,method,norm,covered_source_content,destination_content,absolute_"
                     "residual,min,max,mapped,status\n";
        failures += example("volume_concentration", {0, 1, 2, 3, 4, 5}, {0, .5, 1.5, 2.5, 4.5, 5},
                            {1, 2, 4, 4, 2});
        failures += example("carrier_mass_mixing_ratio", {0, 100, 200, 500, 750, 1000},
                            {0, 50, 150, 400, 900, 1000}, {1e-6, 2e-6, 4e-6, 4e-6, 2e-6});
        failures += example("partial_source_layer", {0, 1, 2, 3}, {1, 1.5}, {1, 2, 3});
        failures +=
            example("masked_land_layer", {0, 1, 2, 3}, {0, .5, 1.5, 2.5, 3}, {1, NAN, 3}, true);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        ++failures;
    }
    Kokkos::finalize();
    return failures ? 1 : 0;
}
