// SPDX-License-Identifier: Apache-2.0
// Independent analytic references; no AXIS or ESMF geometry calls.
#include "case_io.hpp"
#include "reference_geometry.hpp"
#include "result_writer.hpp"

namespace paper {
Reference compute_reference(const Case &case_data) {
    auto reference = ReferenceGeometry<long double>::compute_reference(case_data);
    const auto standard = ReferenceGeometry<double>::compute_reference(case_data);
    const double field_bound = paper::field_bound(case_data.field);
    reference.precision_difference.resize(reference.dst.size());
    reference.area_uncertainty.resize(reference.dst.size());
    for (size_t cell = 0; cell < reference.dst.size(); ++cell) {
        const double difference = std::abs(reference.dst[cell] - standard.dst[cell]);
        reference.precision_difference[cell] = difference;
        reference.uncertainty[cell] += 4 * difference;
        reference.area_uncertainty[cell] =
            4 * std::abs(reference.dst_covered[cell] - standard.dst_covered[cell]) +
            64 * std::numeric_limits<double>::epsilon() * reference.dst_area[cell] +
            reference.roundoff_discarded_area[cell] + standard.roundoff_discarded_area[cell];
        if (reference.uncertainty[cell] > case_data.tolerance * field_bound ||
            reference.area_uncertainty[cell] > case_data.area_tolerance * reference.dst_area[cell])
            throw Error(6, "Reference double/extended-precision agreement exceeds tolerance");
    }
    for (size_t cell = 0; cell < reference.src.size(); ++cell) {
        reference.source_uncertainty[cell] +=
            4 * std::abs(reference.src[cell] - standard.src[cell]);
        if (reference.source_uncertainty[cell] > case_data.tolerance * field_bound)
            throw Error(6, "Source reference precision agreement exceeds tolerance");
    }
    return reference;
}
}  // namespace paper

int main(int argc, char **argv) {
    return paper::guarded([&] {
        paper::Args arguments(argc, argv);
        auto case_data = paper::read_case(arguments.require("case"));
        auto reference = paper::compute_reference(case_data);
        paper::write_reference(arguments.require("output"), reference);
    });
}
