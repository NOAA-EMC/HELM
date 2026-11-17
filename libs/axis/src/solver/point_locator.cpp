// SPDX-License-Identifier: Apache-2.0
#include <ArborX.hpp>
#include <algorithm>
#include <axis/solver/point_locator.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
namespace axis::solver {
namespace {
using V = std::array<double, 3>;
using namespace topology;
constexpr double pi = std::numbers::pi;
V sub(V a, V b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
// Compensate cancellation in edge cross products before normalizing them.
double difference_of_products(double a, double b, double c, double d) {
    double product = c * d;
    return std::fma(a, b, -product) + std::fma(-c, d, product);
}
V cross(V a, V b) {
    return {difference_of_products(a[1], b[2], a[2], b[1]), difference_of_products(a[2], b[0], a[0], b[2]),
            difference_of_products(a[0], b[1], a[1], b[0])};
}
constexpr double relative_roundoff = 64 * std::numeric_limits<double>::epsilon();
double dot(V a, V b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double norm(V a) {
    return std::hypot(a[0], a[1], a[2]);
}
V scale(V a, double x) {
    for (auto &v : a) v *= x;
    return a;
}
V unit(V a) {
    double n = norm(a);
    if (!(n > 0) || !std::isfinite(n)) throw std::invalid_argument("point locator: degenerate edge/normal");
    for (auto &x : a) x /= n;
    return a;
}
void require(bool x, const char *message) {
    if (!x) throw std::invalid_argument(message);
}
bool spherical(CoordinateSystem c) {
    return c == CoordinateSystem::SphericalDeg || c == CoordinateSystem::SphericalRad;
}
V sphere(double lon, double lat) {
    if (std::abs(lat) == pi / 2) return {0, 0, std::copysign(1., lat)};
    return {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
}
struct Cell {
    index_t id, local;
    std::vector<V> vertices, normals;
    V plane{};
    double lon0 = 0, lon1 = 0, lat0 = 0, lat1 = 0;
    double roundoff = 0;
};
}  // namespace
using Box = ArborX::Box<3, double>;
using Tree = decltype(ArborX::BoundingVolumeHierarchy(Kokkos::DefaultHostExecutionSpace{},
                                                      ArborX::Experimental::attach_indices(Kokkos::View<Box *, Kokkos::HostSpace>{})));
struct PointLocator::Impl {
    std::unique_ptr<Tree> tree;
    std::vector<Cell> cells;
    BoundaryModel model;
    bool is_sphere;
    double tol;
    std::array<double, 6> box{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
    PointLocationStatus contains(const Cell &c, V p) const {
        bool boundary = false;
        const double effective_tolerance = tol + c.roundoff;
        if (model == BoundaryModel::ConstantLatitude) {
            // atan2 retains polar latitude when sin(latitude) rounds to 1.
            double lat = std::atan2(p[2], std::hypot(p[0], p[1]));
            // Unit meridian normals measure angular distance, not longitude
            // coordinate separation (which is singular at the poles).
            double west = -std::sin(c.lon0) * p[0] + std::cos(c.lon0) * p[1];
            double east = std::sin(c.lon1) * p[0] - std::cos(c.lon1) * p[1];
            double threshold = std::sin(tol) + c.roundoff;
            if (lat < c.lat0 - effective_tolerance || lat > c.lat1 + effective_tolerance || west < -threshold || east < -threshold)
                return PointLocationStatus::Unmapped;
            boundary = std::abs(lat - c.lat0) <= effective_tolerance || std::abs(lat - c.lat1) <= effective_tolerance ||
                       std::abs(west) <= threshold || std::abs(east) <= threshold;
        } else {
            if (!is_sphere && std::abs(dot(c.plane, sub(p, c.vertices[0]))) > effective_tolerance) return PointLocationStatus::Unmapped;
            double threshold = is_sphere ? std::sin(tol) + c.roundoff : effective_tolerance;
            for (size_t i = 0; i < c.normals.size(); ++i) {
                double d = dot(c.normals[i], is_sphere ? p : sub(p, c.vertices[i]));
                if (d < -threshold) return PointLocationStatus::Unmapped;
                boundary = boundary || std::abs(d) <= threshold;
            }
        }
        return boundary ? PointLocationStatus::Boundary : PointLocationStatus::Interior;
    }
};
PointLocator::PointLocator(const UnstructuredMesh<Kokkos::HostSpace> &mesh, std::span<const index_t> ids, PointLocationOptions options) {
    auto p = std::make_shared<Impl>();
    p->tol = options.tolerance;
    p->is_sphere = spherical(mesh.coord_system());
    p->model = mesh.geometry_metadata().boundary_model;
    require(std::isfinite(p->tol) && p->tol >= 0 && p->tol < .01, "point locator: invalid tolerance");
    require(p->is_sphere || mesh.coord_system() == CoordinateSystem::Cartesian3D, "point locator: invalid coordinates");
    require(ids.size() == mesh.n_cells(), "point locator: cell ID extent mismatch");
    std::set<index_t> seen;
    for (auto id : ids) require(id >= 0 && seen.insert(id).second, "point locator: invalid/duplicate cell ID");
    if (!ids.empty())
        require(p->is_sphere ? (p->model == BoundaryModel::GreatCircle || p->model == BoundaryModel::ConstantLatitude)
                             : p->model == BoundaryModel::CartesianStraight,
                "point locator: explicit supported boundary model required");
    auto off = mesh.conn_offsets();
    auto ind = mesh.conn_indices();
    auto xyz = mesh.node_coords();
    require(off.extent(0) == 0 || (off[0] == 0 && off[off.extent(0) - 1] == static_cast<index_t>(ind.extent(0))), "point locator: invalid offsets");
    double factor = mesh.coord_system() == CoordinateSystem::SphericalDeg ? pi / 180 : 1;
    Kokkos::View<Box *, Kokkos::HostSpace> boxes("point_location_cells", ids.size());
    for (size_t k = 0; k < ids.size(); ++k) {
        require(off[k] >= 0 && off[k + 1] >= off[k] + 3 && off[k + 1] <= static_cast<index_t>(ind.extent(0)), "point locator: invalid connectivity");
        Cell c{};
        c.id = ids[k];
        c.local = k;
        std::vector<V> raw;
        for (index_t j = off[k]; j < off[k + 1]; ++j) {
            auto n = ind[j];
            require(n >= 0 && n < static_cast<index_t>(mesh.n_nodes()), "point locator: invalid vertex index");
            V v{xyz(n, 0), xyz(n, 1), xyz.extent(1) > 2 ? xyz(n, 2) : 0};
            for (double x : v) require(std::isfinite(x), "point locator: nonfinite geometry");
            if (p->is_sphere) {
                v[0] = std::remainder(v[0] * factor, 2 * pi);
                v[1] *= factor;
                require(std::abs(v[1]) <= pi / 2, "point locator: invalid latitude");
            }
            raw.push_back(v);
            c.vertices.push_back(p->is_sphere ? sphere(v[0], v[1]) : v);
        }
        if (p->model == BoundaryModel::ConstantLatitude) {
            require(raw.size() == 4, "point locator: constant-latitude cells must be rectangles");
            double base = raw[0][0];
            c.lon0 = c.lon1 = base;
            c.lat0 = c.lat1 = raw[0][1];
            for (auto v : raw) {
                double lon = base + std::remainder(v[0] - base, 2 * pi);
                c.lon0 = std::min(c.lon0, lon);
                c.lon1 = std::max(c.lon1, lon);
                c.lat0 = std::min(c.lat0, v[1]);
                c.lat1 = std::max(c.lat1, v[1]);
            }
            require(c.lon1 - c.lon0 > 1e-14 && c.lon1 - c.lon0 < pi && c.lat1 - c.lat0 > 1e-14, "point locator: degenerate/ambiguous rectangle");
            std::set<int> corners;
            for (size_t j = 0; j < 4; ++j) {
                double lon = base + std::remainder(raw[j][0] - base, 2 * pi), lat = raw[j][1];
                require((std::abs(lon - c.lon0) < 1e-13 || std::abs(lon - c.lon1) < 1e-13) &&
                            (std::abs(lat - c.lat0) < 1e-13 || std::abs(lat - c.lat1) < 1e-13),
                        "point locator: nonrectangular latitude cell");
                corners.insert((std::abs(lon - c.lon1) < 1e-13 ? 1 : 0) + (std::abs(lat - c.lat1) < 1e-13 ? 2 : 0));
                auto next = raw[(j + 1) % 4];
                require(std::abs(std::remainder(next[0] - raw[j][0], 2 * pi)) < 1e-13 || std::abs(next[1] - lat) < 1e-13,
                        "point locator: crossed rectangle");
            }
            require(corners.size() == 4, "point locator: repeated rectangle vertex");
        } else {
            V center{};
            for (auto v : c.vertices)
                for (int d = 0; d < 3; ++d) center[d] += v[d] / c.vertices.size();
            if (p->is_sphere)
                center = unit(center);
            else {
                // Choose a well-conditioned basis, allowing collinear initial
                // vertices. Normalize edges first to avoid area under/overflow.
                V normal{};
                double strongest = 0;
                for (size_t j = 1; j + 1 < c.vertices.size(); ++j) {
                    auto a = unit(sub(c.vertices[j], c.vertices[0]));
                    auto b = unit(sub(c.vertices[j + 1], c.vertices[0]));
                    auto candidate = cross(a, b);
                    double strength = norm(candidate);
                    if (strength > strongest) {
                        normal = candidate;
                        strongest = strength;
                    }
                }
                require(strongest > 1e-14, "point locator: degenerate planar polygon");
                c.plane = unit(normal);
            }
            for (size_t j = 0; j < c.vertices.size(); ++j) {
                auto a = c.vertices[j], b = c.vertices[(j + 1) % c.vertices.size()];
                V normal = p->is_sphere ? unit(cross(a, b)) : unit(cross(c.plane, unit(sub(b, a))));
                double inward = dot(normal, p->is_sphere ? center : sub(center, a));
                require(std::abs(inward) > 0, "point locator: degenerate polygon");
                if (inward < 0) normal = scale(normal, -1);
                for (auto v : c.vertices) {
                    require(dot(normal, p->is_sphere ? v : sub(v, a)) >= -relative_roundoff * (p->is_sphere ? 1. : norm(sub(v, a))),
                            "point locator: nonconvex polygon");
                    if (!p->is_sphere)
                        require(std::abs(dot(c.plane, sub(v, c.vertices[0]))) <= relative_roundoff * norm(sub(v, c.vertices[0])),
                                "point locator: nonplanar polygon");
                    else
                        require(dot(center, v) > 1e-12, "point locator: polygon must fit an open hemisphere");
                }
                c.normals.push_back(normal);
            }
        }
        // A homothetic expansion about an interior point bounds the polygon
        // expanded by tolerance, including acute corners. Add plane distance.
        V center{};
        for (auto v : c.vertices)
            for (int d = 0; d < 3; ++d) center[d] += v[d] / c.vertices.size();
        double radius = 0, margin = INFINITY;
        if (!p->is_sphere) {
            for (auto v : c.vertices) radius = std::max(radius, norm(sub(v, center)));
            for (size_t j = 0; j < c.normals.size(); ++j) margin = std::min(margin, dot(c.normals[j], sub(center, c.vertices[j])));
        }
        // This arithmetic floor is distinct from the user-selected geometric
        // tolerance. Use it for both predicates and conservative search bounds.
        double coordinate_scale = p->is_sphere ? 1. : 0.;
        for (const auto &v : c.vertices)
            for (double x : v) coordinate_scale = std::max(coordinate_scale, std::abs(x));
        c.roundoff = relative_roundoff * coordinate_scale;
        double effective_tolerance = p->tol + c.roundoff;
        double padding = p->is_sphere ? effective_tolerance : 2 * effective_tolerance * (1 + radius / margin);
        std::array<double, 6> cb{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int d = 0; d < 3; ++d) {
            if (p->is_sphere) {
                cb[d] = -1 - padding;
                cb[d + 3] = 1 + padding;
            } else
                for (auto v : c.vertices) {
                    cb[d] = std::min(cb[d], v[d] - padding);
                    cb[d + 3] = std::max(cb[d + 3], v[d] + padding);
                }
            p->box[d] = std::min(p->box[d], cb[d]);
            p->box[d + 3] = std::max(p->box[d + 3], cb[d + 3]);
        }
        boxes(k) = Box{{std::nextafter(cb[0], -INFINITY), std::nextafter(cb[1], -INFINITY), std::nextafter(cb[2], -INFINITY)},
                       {std::nextafter(cb[3], INFINITY), std::nextafter(cb[4], INFINITY), std::nextafter(cb[5], INFINITY)}};
        p->cells.push_back(std::move(c));
    }
    p->tree = std::make_unique<Tree>(Kokkos::DefaultHostExecutionSpace{}, ArborX::Experimental::attach_indices(boxes));
    impl_ = std::move(p);
}
std::vector<V> PointLocator::canonical_points(PointSetView points) const {
    auto xyz = points.coordinates;
    bool sph = spherical(points.coordinate_system);
    require(sph == impl_->is_sphere && (sph || points.coordinate_system == CoordinateSystem::Cartesian3D),
            "point locator: incompatible coordinate systems");
    require(xyz.extent(0) == points.ids.size() && (xyz.extent(1) == 2 || xyz.extent(1) == 3), "point locator: point coordinate extent mismatch");
    std::set<index_t> seen;
    std::vector<V> result;
    result.reserve(points.ids.size());
    double factor = points.coordinate_system == CoordinateSystem::SphericalDeg ? pi / 180 : 1;
    for (size_t i = 0; i < points.ids.size(); ++i) {
        require(points.ids[i] >= 0 && seen.insert(points.ids[i]).second, "point locator: invalid/duplicate point ID");
        V v{xyz(i, 0), xyz(i, 1), xyz.extent(1) == 3 ? xyz(i, 2) : 0};
        for (double x : v) require(std::isfinite(x), "point locator: nonfinite point");
        if (sph) {
            v[0] = std::remainder(v[0] * factor, 2 * pi);
            v[1] *= factor;
            require(std::abs(v[1]) <= pi / 2, "point locator: invalid point latitude");
            v = sphere(v[0], v[1]);
        }
        result.push_back(v);
    }
    return result;
}
PointLocation PointLocator::locate_canonical(const V &p) const {
    PointLocation best;
    if (impl_->cells.empty()) return best;
    Kokkos::View<decltype(ArborX::intersects(ArborX::Point<3, double>{})) *, Kokkos::HostSpace> queries("point_query", 1);
    queries(0) = ArborX::intersects(ArborX::Point<3, double>{p[0], p[1], p[2]});
    Kokkos::View<Tree::value_type *, Kokkos::HostSpace> candidates;
    Kokkos::View<int *, Kokkos::HostSpace> offsets;
    impl_->tree->query(Kokkos::DefaultHostExecutionSpace{}, queries, candidates, offsets);
    for (size_t k = 0; k < candidates.extent(0); ++k) {
        const auto &c = impl_->cells[candidates(k).index];
        auto status = impl_->contains(c, p);
        if (status != PointLocationStatus::Unmapped && (best.cell_id < 0 || c.id < best.cell_id)) {
            best.cell_id = c.id;
            best.owner_rank = 0;
            best.owner_local_index = c.local;
            best.status = status;
        }
    }
    return best;
}
std::vector<PointLocation> PointLocator::locate(PointSetView points) const {
    auto coords = canonical_points(points);
    std::vector<PointLocation> result;
    result.reserve(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        auto r = locate_canonical(coords[i]);
        r.point_id = points.ids[i];
        result.push_back(r);
    }
    return result;
}
std::array<double, 6> PointLocator::bounds() const {
    return impl_->box;
}
}  // namespace axis::solver
