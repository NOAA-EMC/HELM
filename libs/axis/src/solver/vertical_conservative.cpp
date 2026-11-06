// SPDX-License-Identifier: Apache-2.0
/// @file src/solver/vertical_conservative.cpp
/// @brief Conservative remapping of layer means in an explicit physical measure.
///
/// Implements interface-overlap integration with first-order or limited-linear
/// (minmod) reconstruction, aliasing guards, and coverage diagnostics.

#include <axis/solver/vertical_conservative.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace axis::solver {
namespace {
KOKKOS_INLINE_FUNCTION double minimum(double a, double b) {
    return a < b ? a : b;
}
KOKKOS_INLINE_FUNCTION double maximum(double a, double b) {
    return a > b ? a : b;
}
KOKKOS_INLINE_FUNCTION double minmod(double a, double b, double c) {
    if (a > 0 && b > 0 && c > 0) return minimum(a, minimum(b, c));
    if (a < 0 && b < 0 && c < 0) return maximum(a, maximum(b, c));
    return 0;
}
template <class A, class B>
bool aliases(A a, B b) {
    if (!a.size() || !b.size()) return false;
    auto x = reinterpret_cast<std::uintptr_t>(a.data()), y = reinterpret_cast<std::uintptr_t>(b.data());
    return x < y + b.span() * sizeof(double) && y < x + a.span() * sizeof(double);
}
}  // namespace

template <class M>
VerticalConservativeRegridder<M>::VerticalConservativeRegridder(ConstField src, ConstField dst, std::size_t columns, Mask src_mask, Mask dst_mask)
    : columns_(columns),
      ns_(src.extent(1) ? src.extent(1) - 1 : 0),
      nd_(dst.extent(1) ? dst.extent(1) - 1 : 0),
      source_rows_(src.extent(0)),
      destination_rows_(dst.extent(0)),
      geometry_rows_(source_rows_ == 1 && destination_rows_ == 1 ? 1 : columns) {
    if (!columns || !ns_ || !nd_ || (source_rows_ != 1 && source_rows_ != columns) || (destination_rows_ != 1 && destination_rows_ != columns))
        throw std::invalid_argument("Vertical conservative: interfaces need shared/varying rows and at least one layer");
    auto check_mask = [&](Mask mask, std::size_t n) {
        if (mask.size() && (mask.extent(0) != columns || mask.extent(1) != n))
            throw std::invalid_argument("Vertical conservative: incompatible mask shape");
    };
    check_mask(src_mask, ns_);
    check_mask(dst_mask, nd_);
    using Exec = typename M::execution_space;
    auto validate_interfaces = [&](ConstField edges) {
        int bad = 0;
        Kokkos::parallel_reduce(
            "VerticalMeasureValidation", Kokkos::RangePolicy<Exec>(0, edges.extent(0)),
            KOKKOS_LAMBDA(std::size_t c, int &errors) {
                bool increasing = edges(c, edges.extent(1) - 1) >= edges(c, 0), invalid = false;
                for (std::size_t i = 0; i < edges.extent(1); ++i) {
                    invalid = invalid || !Kokkos::isfinite(edges(c, i));
                    if (i) {
                        double h = edges(c, i) - edges(c, i - 1);
                        invalid = invalid || !Kokkos::isfinite(h) || (increasing ? h < 0 : h > 0);
                    }
                }
                if (invalid) ++errors;
            },
            bad);
        if (bad) throw std::invalid_argument("Vertical conservative: interfaces must be finite and monotone");
    };
    validate_interfaces(src);
    validate_interfaces(dst);
    source_width_ = Field("vertical_source_width", source_rows_, ns_);
    destination_width_ = Field("vertical_destination_width", destination_rows_, nd_);
    source_direction_ = decltype(source_direction_)("vertical_source_direction", source_rows_);
    lower_source_ = Field("vertical_lower_source", source_rows_, ns_);
    lower_destination_ = Field("vertical_lower_destination", destination_rows_, nd_);
    auto sw = source_width_, dw = destination_width_, sl = lower_source_, dl = lower_destination_;
    auto direction = source_direction_;
    const auto ns = ns_, nd = nd_, nc = columns_, ng = geometry_rows_, sr = source_rows_, dr = destination_rows_;
    Kokkos::parallel_for(
        "VerticalSourceMeasures", Kokkos::RangePolicy<Exec>(0, sr), KOKKOS_LAMBDA(std::size_t c) {
            direction(c) = src(c, ns) < src(c, 0) ? -1 : 1;
            for (std::size_t i = 0; i < ns; ++i) {
                sl(c, i) = minimum(src(c, i), src(c, i + 1));
                sw(c, i) = std::abs(src(c, i + 1) - src(c, i));
            }
        });
    Kokkos::parallel_for(
        "VerticalDestinationMeasures", Kokkos::RangePolicy<Exec>(0, dr), KOKKOS_LAMBDA(std::size_t c) {
            for (std::size_t j = 0; j < nd; ++j) {
                dl(c, j) = minimum(dst(c, j), dst(c, j + 1));
                dw(c, j) = std::abs(dst(c, j + 1) - dst(c, j));
            }
        });
    source_mask_ = decltype(source_mask_)("vertical_source_mask", nc, ns);
    destination_mask_ = decltype(destination_mask_)("vertical_destination_mask", nc, nd);
    if (src_mask.size())
        Kokkos::deep_copy(source_mask_, src_mask);
    else
        Kokkos::deep_copy(source_mask_, 1);
    if (dst_mask.size())
        Kokkos::deep_copy(destination_mask_, dst_mask);
    else
        Kokkos::deep_copy(destination_mask_, 1);
    source_index_ = decltype(source_index_)("vertical_overlap_source", ng, ns + nd);
    destination_index_ = decltype(destination_index_)("vertical_overlap_destination", ng, ns + nd);
    overlap_ = Field("vertical_overlap_measure", ng, ns + nd);
    moment_ = Field("vertical_overlap_moment", ng, ns + nd);
    counts_ = decltype(counts_)("vertical_overlap_counts", ng);
    auto si = source_index_, di = destination_index_;
    auto ov = overlap_, mo = moment_;
    auto counts = counts_;
    Kokkos::parallel_for(
        "VerticalBuildOverlaps", Kokkos::RangePolicy<Exec>(0, ng), KOKKOS_LAMBDA(std::size_t c) {
            auto srow = sr == 1 ? 0 : c, drow = dr == 1 ? 0 : c;
            bool reverse_s = src(srow, ns) < src(srow, 0), reverse_d = dst(drow, nd) < dst(drow, 0);
            std::size_t a = 0, b = 0, k = 0;
            while (a < ns && b < nd) {
                auto i = reverse_s ? ns - 1 - a : a, j = reverse_d ? nd - 1 - b : b;
                if (sw(srow, i) == 0) {
                    ++a;
                    continue;
                }
                if (dw(drow, j) == 0) {
                    ++b;
                    continue;
                }
                double sh = sl(srow, i) + sw(srow, i), dh = dl(drow, j) + dw(drow, j);
                double low = maximum(sl(srow, i), dl(drow, j)), high = minimum(sh, dh);
                if (high > low) {
                    si(c, k) = i;
                    di(c, k) = j;
                    ov(c, k) = high - low;
                    mo(c, k) = (high - low) * ((low - sl(srow, i)) - .5 * sw(srow, i) + .5 * (high - low));
                    ++k;
                }
                if (sh <= dh) ++a;
                if (dh <= sh) ++b;
            }
            counts(c) = k;
        });
    source_covered_ = Field("vertical_source_covered", nc, ns);
    destination_covered_ = Field("vertical_destination_covered", nc, nd);
    source_moment_ = Field("vertical_source_covered_moment", nc, ns);
    source_fraction_ = Field("vertical_source_fraction", nc, ns);
    destination_fraction_ = Field("vertical_destination_fraction", nc, nd);
    auto sm = source_mask_, dm = destination_mask_;
    auto cs = source_covered_, cd = destination_covered_, ms = source_moment_;
    auto sf = source_fraction_, df = destination_fraction_;
    Kokkos::parallel_for(
        "VerticalCoverage", Kokkos::RangePolicy<Exec>(0, nc), KOKKOS_LAMBDA(std::size_t c) {
            auto g = ng == 1 ? 0 : c, s = sr == 1 ? 0 : c, d = dr == 1 ? 0 : c;
            for (std::size_t k = 0; k < counts(g); ++k) {
                auto i = si(g, k), j = di(g, k);
                if (!sm(c, i) || !dm(c, j)) continue;
                cs(c, i) += ov(g, k);
                cd(c, j) += ov(g, k);
                ms(c, i) += mo(g, k);
            }
            for (std::size_t i = 0; i < ns; ++i) sf(c, i) = sw(s, i) > 0 ? cs(c, i) / sw(s, i) : 0;
            for (std::size_t j = 0; j < nd; ++j) df(c, j) = dw(d, j) > 0 ? cd(c, j) / dw(d, j) : 0;
        });
    Exec().fence();
}

template <class M>
void VerticalConservativeRegridder<M>::validate_field(ConstField src) const {
    if (src.extent(0) != columns_ || src.extent(1) != ns_) throw std::invalid_argument("Vertical conservative: source shape differs from operator");
    using Exec = typename M::execution_space;
    auto sm = source_mask_;
    auto sw = source_width_;
    auto sr = source_rows_;
    int bad = 0;
    Kokkos::parallel_reduce(
        "VerticalTracerValidation", Kokkos::RangePolicy<Exec>(0, columns_),
        KOKKOS_LAMBDA(std::size_t c, int &errors) {
            auto s = sr == 1 ? 0 : c;
            for (std::size_t i = 0; i < src.extent(1); ++i)
                if (sm(c, i) && sw(s, i) > 0 && !Kokkos::isfinite(src(c, i))) ++errors;
        },
        bad);
    if (bad) throw std::invalid_argument("Vertical conservative: wet source layer means must be finite");
}

template <class M>
typename VerticalConservativeRegridder<M>::Field VerticalConservativeRegridder<M>::slopes(ConstField src, VerticalConservativeOrder order,
                                                                                          Field work) const {
    if (order == VerticalConservativeOrder::First) return {};
    if (order != VerticalConservativeOrder::LimitedLinear) throw std::invalid_argument("Vertical conservative: invalid reconstruction order");
    if (!work.size()) work = Field("vertical_slope_workspace", columns_, ns_);
    if (work.extent(0) != columns_ || work.extent(1) != ns_ || aliases(src, work))
        throw std::invalid_argument("Vertical conservative: invalid or aliased slope workspace");
    using Exec = typename M::execution_space;
    auto sm = source_mask_;
    auto sw = source_width_;
    auto direction = source_direction_;
    auto sr = source_rows_, ns = ns_;
    Kokkos::parallel_for(
        "VerticalLimitedSlopes", Kokkos::RangePolicy<Exec>(0, columns_), KOKKOS_LAMBDA(std::size_t c) {
            auto s = sr == 1 ? 0 : c;
            for (std::size_t i = 0; i < ns; ++i) {
                double slope = 0;
                if (i > 0 && i + 1 < ns && sm(c, i - 1) && sm(c, i) && sm(c, i + 1) && sw(s, i - 1) > 0 && sw(s, i) > 0 && sw(s, i + 1) > 0) {
                    double q = src(c, i), prev = src(c, i - 1), next = src(c, i + 1);
                    double left = direction(s) * (.5 * sw(s, i - 1) + .5 * sw(s, i)), right = direction(s) * (.5 * sw(s, i) + .5 * sw(s, i + 1));
                    slope = minmod(2 * (q - prev) / left, (next - prev) / (left + right), 2 * (next - q) / right);
                    double lo = minimum(q, minimum(prev, next)), hi = maximum(q, maximum(prev, next));
                    double bound = 2 * minimum(q - lo, hi - q) / sw(s, i);
                    slope = maximum(-bound, minimum(bound, slope));
                }
                work(c, i) = slope;
            }
        });
    return work;
}

template <class M>
void VerticalConservativeRegridder<M>::apply(ConstField src, Field dst, VerticalConservativeOrder order, VerticalConservativeNorm norm,
                                             Field workspace) const {
    if (dst.extent(0) != columns_ || dst.extent(1) != nd_ || aliases(src, dst) || aliases(dst, workspace))
        throw std::invalid_argument("Vertical conservative: invalid or aliased destination");
    if (norm != VerticalConservativeNorm::DestinationMeasure && norm != VerticalConservativeNorm::CoveredMeasure)
        throw std::invalid_argument("Vertical conservative: invalid normalization");
    validate_field(src);
    auto slope = slopes(src, order, workspace);
    bool linear = order == VerticalConservativeOrder::LimitedLinear;
    auto si = source_index_, di = destination_index_;
    auto ov = overlap_, mo = moment_;
    auto counts = counts_;
    auto sm = source_mask_, dm = destination_mask_;
    auto dw = destination_width_, cd = destination_covered_;
    auto ng = geometry_rows_, dr = destination_rows_, nd = nd_;
    using Exec = typename M::execution_space;
    Kokkos::parallel_for(
        "VerticalConservativeApply", Kokkos::RangePolicy<Exec>(0, columns_), KOKKOS_LAMBDA(std::size_t c) {
            auto g = ng == 1 ? 0 : c, d = dr == 1 ? 0 : c;
            for (std::size_t j = 0; j < nd; ++j) dst(c, j) = 0;
            for (std::size_t k = 0; k < counts(g); ++k) {
                auto i = si(g, k), j = di(g, k);
                if (!sm(c, i) || !dm(c, j)) continue;
                dst(c, j) += ov(g, k) * src(c, i) + (linear ? mo(g, k) * slope(c, i) : 0.);
            }
            for (std::size_t j = 0; j < nd; ++j) {
                double measure = norm == VerticalConservativeNorm::CoveredMeasure ? cd(c, j) : dw(d, j);
                if (measure > 0)
                    dst(c, j) /= measure;
                else
                    dst(c, j) = 0;
            }
        });
    Exec().fence();
}

template <class M>
typename VerticalConservativeRegridder<M>::Integral VerticalConservativeRegridder<M>::covered_source_integral(ConstField src,
                                                                                                              VerticalConservativeOrder order,
                                                                                                              Field workspace) const {
    validate_field(src);
    auto slope = slopes(src, order, workspace);
    bool linear = order == VerticalConservativeOrder::LimitedLinear;
    Integral total("vertical_covered_source_integral", columns_);
    auto cs = source_covered_, ms = source_moment_;
    auto sm = source_mask_;
    auto ns = ns_;
    using Exec = typename M::execution_space;
    Kokkos::parallel_for(
        "VerticalCoveredSourceIntegral", Kokkos::RangePolicy<Exec>(0, columns_), KOKKOS_LAMBDA(std::size_t c) {
            double sum = 0;
            for (std::size_t i = 0; i < ns; ++i)
                if (sm(c, i) && cs(c, i) > 0) sum += src(c, i) * cs(c, i) + (linear ? slope(c, i) * ms(c, i) : 0.);
            total(c) = sum;
        });
    Exec().fence();
    return total;
}

template class VerticalConservativeRegridder<Kokkos::HostSpace>;
#ifdef KOKKOS_ENABLE_CUDA
template class VerticalConservativeRegridder<Kokkos::CudaSpace>;
#endif
#ifdef KOKKOS_ENABLE_HIP
template class VerticalConservativeRegridder<Kokkos::HIPSpace>;
#endif
}  // namespace axis::solver
