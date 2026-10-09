// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <map>
#include <sstream>

#include "analytical_fields.hpp"
#include "case_io.hpp"
#include "result_writer.hpp"
using namespace paper;
int main(int argc, char **argv) {
    return guarded([&] {
        Args a(argc, argv);
        auto c = read_case(a.require("case"));
        auto r = read_reference(a.require("reference"));
        std::filesystem::path dir = a.require("run");
        auto o = read_output(dir / "fields.nc");
        if (o.src.size() != c.src.size() || o.dst.size() != c.dst.size() ||
            o.src_area.size() != o.src.size() || o.src_frac.size() != o.src.size() ||
            o.dst_area.size() != o.dst.size() || o.dst_frac.size() != o.dst.size() ||
            o.row_sum.size() != o.dst.size())
            throw Error(6, "Invalid output extents");
        double mae = 0, rms = 0, bias = 0, maxerr = 0, weight = 0, si = 0, di = 0, canon_s = 0,
               canon_d = 0, constant = 0, rowerr = 0, over = 0, under = 0, area_delta = 0;
        int unmapped = 0, masked = 0, nonfinite = 0, eligible = 0, missing_expected = 0;
        int negligible_coverage = 0, unexpected_mapped = 0;
        double relative_sum = 0, relative_squared = 0, relative_max = 0, reference_integral = 0;
        double reference_min = INFINITY, reference_max = -INFINITY, output_min = INFINITY,
               output_max = -INFINITY;
        int relative_count = 0, relative_excluded = 0;
        double low = field_lower_bound(c.field);
        double high = field_bound(c.field);
        for (size_t i = 0; i < o.src.size(); ++i)
            if (c.src.mask[i]) {
                si += o.src[i] * o.src_area[i] * o.src_frac[i];
                canon_s += o.src[i] * r.src_covered[i];
                area_delta =
                    std::max(area_delta, std::abs(o.src_area[i] - r.src_area[i]) / r.src_area[i]);
            }
        std::map<int, std::pair<double, double>> parents;
        for (size_t j = 0; j < o.dst.size(); ++j) {
            if (!c.dst.mask[j]) {
                ++masked;
                continue;
            }
            if (!std::isfinite(o.dst[j])) {
                ++nonfinite;
                continue;
            }
            if (o.dst_frac[j] <= 0) {
                ++unmapped;
                if (r.dst_covered[j] > r.dst_area[j] * 1e-10)
                    ++missing_expected;
                continue;
            }
            if (r.dst_covered[j] <= 0) {
                ++negligible_coverage;
                if (o.dst_frac[j] > c.area_tolerance)
                    ++unexpected_mapped;
                continue;
            }
            ++eligible;
            double w = r.dst_covered[j];
            double e = o.dst[j] - r.dst[j];
            weight += w;
            mae += w * std::abs(e);
            rms += w * e * e;
            bias += w * e;
            maxerr = std::max(maxerr, std::abs(e));
            if (std::abs(r.dst[j]) > c.near_zero_tolerance * paper::field_bound(c.field)) {
                const double relative_error = std::abs(e / r.dst[j]);
                relative_sum += relative_error;
                relative_squared += relative_error * relative_error;
                relative_max = std::max(relative_max, relative_error);
                ++relative_count;
            } else
                ++relative_excluded;
            reference_min = std::min(reference_min, r.dst[j]);
            reference_max = std::max(reference_max, r.dst[j]);
            output_min = std::min(output_min, o.dst[j]);
            output_max = std::max(output_max, o.dst[j]);
            reference_integral +=
                r.dst[j] * (c.norm == "fracarea" ? r.dst_covered[j] : r.dst_area[j]);
            double fraction = c.norm == "fracarea" ? o.dst_frac[j] : 1;
            di += o.dst[j] * o.dst_area[j] * fraction;
            canon_d += o.dst[j] * (c.norm == "fracarea" ? r.dst_covered[j] : r.dst_area[j]);
            double expected = c.norm == "fracarea" ? 1 : o.dst_frac[j];
            if (c.field == "constant")
                constant = std::max(constant, std::abs(o.dst[j] - expected));
            rowerr = std::max(rowerr, std::abs(o.row_sum[j] - expected));
            over = std::max(over, o.dst[j] - high * expected);
            under = std::max(under, low * expected - o.dst[j]);
            area_delta =
                std::max(area_delta, std::abs(o.dst_area[j] - r.dst_area[j]) / r.dst_area[j]);
            const double parent_weight = c.norm == "fracarea" ? w : r.dst_area[j];
            parents[c.dst.parent[j]].first += o.dst[j] * parent_weight;
            parents[c.dst.parent[j]].second += parent_weight;
        }
        const double field_bound = paper::field_bound(c.field);
        double residual = di - si, scale = field_bound * weight;
        double relative = std::abs(si) > c.near_zero_tolerance * scale ? residual / si : NAN;
        double reproduction_error = 0;
        const bool comparing = !a.get("compare").empty();
        if (comparing) {
            const auto previous = read_output(a.get("compare"));
            if (previous.dst.size() != o.dst.size())
                throw Error(6, "Reproduction comparison has incompatible extents");
            for (size_t cell = 0; cell < o.dst.size(); ++cell) {
                if (!std::isfinite(previous.dst[cell]) || !std::isfinite(o.dst[cell]))
                    reproduction_error = INFINITY;
                else
                    reproduction_error =
                        std::max(reproduction_error, std::abs(previous.dst[cell] - o.dst[cell]));
            }
        }
        bool pass = eligible > 0 && nonfinite == 0 && missing_expected == 0 &&
                    unexpected_mapped == 0 && rowerr <= c.constant_tolerance &&
                    std::abs(residual) <= c.conservation_tolerance * scale &&
                    (c.field != "constant" || (constant <= c.constant_tolerance * field_bound &&
                                               maxerr <= c.constant_tolerance * field_bound)) &&
                    (c.field != "smooth" ||
                     std::sqrt(rms / weight) <= c.smooth_rms_tolerance * field_bound) &&
                    (!comparing || reproduction_error <= c.reproduction_tolerance * field_bound);
        std::ostringstream out;
        out << "{\"schema_version\":1,\"eligible\":" << eligible
            << ",\"negligible_coverage\":" << negligible_coverage
            << ",\"unexpected_mapped\":" << unexpected_mapped
            << ",\"missing_expected_mapped\":" << missing_expected << ",\"masked\":" << masked
            << ",\"unmapped\":" << unmapped << ",\"nonfinite\":" << nonfinite
            << ",\"mean_absolute_error\":" << number(weight > 0 ? mae / weight : NAN)
            << ",\"rms_error\":" << number(weight > 0 ? std::sqrt(rms / weight) : NAN)
            << ",\"bias\":" << number(weight > 0 ? bias / weight : NAN)
            << ",\"max_error\":" << number(maxerr) << ",\"source_integral\":" << number(si)
            << ",\"destination_integral\":" << number(di)
            << ",\"conservation_absolute\":" << number(std::abs(residual))
            << ",\"conservation_relative\":" << number(relative) << ",\"relative_residual_reason\":"
            << quote(std::isfinite(relative) ? "defined" : "near-zero source integral")
            << ",\"canonical_source_integral\":" << number(canon_s)
            << ",\"canonical_destination_integral\":" << number(canon_d)
            << ",\"constant_error\":" << (c.field == "constant" ? number(constant) : "null")
            << ",\"row_sum_error\":" << number(rowerr) << ",\"overshoot\":" << number(over)
            << ",\"undershoot\":" << number(under)
            << ",\"max_relative_area_difference\":" << number(area_delta)
            << ",\"numerical_validation\":" << quote(pass ? "pass" : "fail")
            << ",\"reproduction_max_error\":" << (comparing ? number(reproduction_error) : "null")
            << ",\"reproduction_reason\":"
            << quote(comparing ? "compared with quality pass"
                               : "quality pass; no comparison requested")
            << ",\"parent_averages\":{";
        bool first = true;
        for (auto [id, v] : parents) {
            if (!first)
                out << ',';
            first = false;
            out << quote(std::to_string(id)) << ':' << number(v.first / v.second);
        }
        out << "},\"relative_mean_percent\":"
            << number(relative_count ? 100 * relative_sum / relative_count : NAN)
            << ",\"relative_rms_percent\":"
            << number(relative_count ? 100 * std::sqrt(relative_squared / relative_count) : NAN)
            << ",\"relative_max_percent\":" << number(relative_count ? 100 * relative_max : NAN)
            << ",\"relative_eligible\":" << relative_count
            << ",\"relative_excluded_near_zero\":" << relative_excluded
            << ",\"source_conservation_percent\":"
            << number(std::isfinite(relative) ? 100 * std::abs(relative) : NAN)
            << ",\"target_conservation_percent\":"
            << number(std::abs(reference_integral) > c.near_zero_tolerance * scale
                          ? 100 * std::abs(canon_d - reference_integral) /
                                std::abs(reference_integral)
                          : NAN)
            << ",\"lmin\":"
            << number((reference_min - output_min) /
                      std::max(std::abs(reference_min), std::abs(reference_max)))
            << ",\"lmax\":"
            << number((output_max - reference_max) /
                      std::max(std::abs(reference_min), std::abs(reference_max)))
            << "}";
        atomic_text(dir / "metrics.json", out.str());
    });
}
