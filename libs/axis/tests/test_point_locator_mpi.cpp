// SPDX-License-Identifier: Apache-2.0
#include <mpi.h>
#include <sys/resource.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <axis/distributed/point_locator.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace axis;
using namespace axis::topology;
using namespace axis::solver;
using namespace axis::distributed;
using Mesh = UnstructuredMesh<Kokkos::HostSpace>;
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
Mesh mesh(std::vector<std::array<double, 3>> vertices, std::vector<index_t> offsets, BoundaryModel model, CoordinateSystem system) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> xyz("vertices", vertices.size(), 3);
    Kokkos::View<index_t *, Kokkos::HostSpace> off("offsets", offsets.size()), ind("indices", vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        ind(i) = i;
        for (int d = 0; d < 3; ++d) xyz(i, d) = vertices[i][d];
    }
    for (size_t i = 0; i < offsets.size(); ++i) off(i) = offsets[i];
    GeometryMetadata meta;
    meta.boundary_model = model;
    return Mesh(xyz, off, ind, system, {}, {}, meta);
}
PointSetView points(const std::vector<index_t> &ids, const std::vector<double> &xyz, CoordinateSystem system) {
    return {ids, field_view<const double, 2>(xyz.data(), ids.size(), 3), system};
}
void local_tests() {
    auto m = mesh({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {1, 1, 0}}, {0, 4, 8},
                  BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    std::vector<index_t> cells{90, 10}, ids{7, 8, 9, 10, 11};
    // Column-major x,y,z. Two distinct records share exactly the same coordinates.
    std::vector<double> xyz{.5, .5, 1, 3, 1.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0};
    PointLocator locator(m, cells);
    auto r = locator.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
    check(r.size() == 5 && r[0].point_id == 7 && r[1].point_id == 8, "point identity");
    check(r[0].cell_id == 90 && r[1].cell_id == 90, "coincident points preserved");
    check(r[2].cell_id == 10 && r[2].status == PointLocationStatus::Boundary, "stable boundary tie");
    check(r[3].status == PointLocationStatus::Unmapped && r[3].cell_id == -1, "outside");
    check(r[4].cell_id == 10 && r[4].status == PointLocationStatus::Interior, "interior");
    auto sphere =
        mesh({{170, -10, 0}, {-170, -10, 0}, {-170, 10, 0}, {170, 10, 0}}, {0, 4}, BoundaryModel::ConstantLatitude, CoordinateSystem::SphericalDeg);
    PointLocator spherical(sphere, std::vector<index_t>{42});
    ids = {1, 2, 3};
    xyz = {180, 0, 170, 0, 0, 10, 0, 0, 0};
    r = spherical.locate(points(ids, xyz, CoordinateSystem::SphericalDeg));
    check(r[0].cell_id == 42 && r[1].cell_id == -1 && r[2].status == PointLocationStatus::Boundary, "dateline rectangle");
    for (double &x : xyz) x *= std::acos(-1.) / 180.;
    check(spherical.locate(points(ids, xyz, CoordinateSystem::SphericalRad))[0].cell_id == 42, "radian queries");
    auto polar = mesh({{0, 80, 0}, {120, 80, 0}, {-120, 80, 0}}, {0, 3}, BoundaryModel::GreatCircle, CoordinateSystem::SphericalDeg);
    PointLocator pole(polar, std::vector<index_t>{6});
    ids = {3, 4};
    xyz = {0, 0, 90, 0, 0, 0};
    r = pole.locate(points(ids, xyz, CoordinateSystem::SphericalDeg));
    check(r[0].cell_id == 6 && r[1].cell_id == -1, "polar great circle");
}
template <class F>
void rejects(F f, const char *message) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}
void geometry_failures() {
    auto triangle = mesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, {0, 3}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    PointLocator loc(triangle, std::vector<index_t>{2});
    std::vector<index_t> ids{1, 2, 3};
    std::vector<double> xyz{.25, 1, 1.5, .25, 1, 1.5, 0, 0, 0};
    auto r = loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
    check(r[0].status == PointLocationStatus::Interior && r[1].status == PointLocationStatus::Boundary && r[2].cell_id == -1, "triangle containment");
    xyz[0] = NAN;
    rejects([&] { (void)loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D)); }, "nonfinite points rejected");
    rejects([&] { PointLocator bad(triangle, std::vector<index_t>{}); }, "ID extents rejected");
    auto concave = mesh({{0, 0, 0}, {2, 0, 0}, {.5, .5, 0}, {0, 2, 0}}, {0, 4}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    rejects([&] { PointLocator bad(concave, std::vector<index_t>{1}); }, "concave polygon rejected");
    auto degenerate = mesh({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, {0, 3}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    rejects([&] { PointLocator bad(degenerate, std::vector<index_t>{1}); }, "degenerate polygon rejected");
    auto unspecified = triangle;
    auto meta = unspecified.geometry_metadata();
    meta.boundary_model = BoundaryModel::Unspecified;
    unspecified.set_geometry_metadata(meta);
    rejects([&] { PointLocator bad(unspecified, std::vector<index_t>{1}); }, "unspecified geometry rejected");
    auto rad = mesh({{-.2, -.2, 0}, {.2, -.2, 0}, {.2, .2, 0}, {-.2, .2, 0}}, {0, 4}, BoundaryModel::GreatCircle, CoordinateSystem::SphericalRad);
    PointLocator radial(rad, std::vector<index_t>{3});
    ids = {4};
    xyz = {0, 0, 0};
    check(radial.locate(points(ids, xyz, CoordinateSystem::SphericalDeg))[0].cell_id == 3, "radian mesh");
}
void review_regressions() {
    int failures = 0;
    auto run = [&](auto test, const char *name) {
        try {
            test();
        } catch (const std::exception &e) {
            ++failures;
            std::cerr << name << ": " << e.what() << '\n';
        }
    };
    run(
        [] {
            auto m =
                mesh({{0, 80, 0}, {10, 80, 0}, {10, 90, 0}, {0, 90, 0}}, {0, 4}, BoundaryModel::ConstantLatitude, CoordinateSystem::SphericalDeg);
            PointLocator loc(m, std::vector<index_t>{1});
            std::vector<index_t> ids{1, 2};
            std::vector<double> xyz{180, 5, 89.9999999, 89.9999999, 0, 0};
            auto r = loc.locate(points(ids, xyz, CoordinateSystem::SphericalDeg));
            check(r[0].status == PointLocationStatus::Unmapped && r[1].status == PointLocationStatus::Interior, "near-pole classification");
            PointLocationOptions exact;
            exact.tolerance = 0;
            PointLocator zero_tolerance(m, std::vector<index_t>{1}, exact);
            ids = {3};
            xyz = {180, 90, 0};
            check(zero_tolerance.locate(points(ids, xyz, CoordinateSystem::SphericalDeg))[0].status == PointLocationStatus::Boundary,
                  "exact pole at zero tolerance");
        },
        "review polar latitude");
    run(
        [] {
            double pi = std::acos(-1.);
            auto m =
                mesh({{0, 80, 0}, {10, 80, 0}, {10, 90, 0}, {0, 90, 0}}, {0, 4}, BoundaryModel::ConstantLatitude, CoordinateSystem::SphericalDeg);
            PointLocator loc(m, std::vector<index_t>{1});
            std::vector<index_t> ids{1};
            std::vector<double> xyz{-5e-9, 89 * pi / 180, 0};
            check(loc.locate(points(ids, xyz, CoordinateSystem::SphericalRad))[0].status == PointLocationStatus::Boundary,
                  "angular meridian tolerance");
        },
        "review meridian distance");
    run(
        [] {
            for (double side : {1e-8, 1e8, 1e150}) {
                auto m = mesh({{0, 0, 0}, {side, 0, 0}, {side, side, 0}, {0, side, 0}}, {0, 4}, BoundaryModel::CartesianStraight,
                              CoordinateSystem::Cartesian3D);
                PointLocationOptions opts;
                opts.tolerance = 0;
                PointLocator loc(m, std::vector<index_t>{1}, opts);
                std::vector<index_t> ids{1};
                std::vector<double> xyz{side / 2, side / 2, 0};
                check(loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D))[0].status == PointLocationStatus::Interior,
                      "scale-safe containment");
            }
        },
        "review Cartesian scale");
    run(
        [] {
            auto m = mesh({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {0, 1, 0}}, {0, 5}, BoundaryModel::CartesianStraight,
                          CoordinateSystem::Cartesian3D);
            PointLocator loc(m, std::vector<index_t>{1});
            std::vector<index_t> ids{1};
            std::vector<double> xyz{1, .5, 0};
            check(loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D))[0].status == PointLocationStatus::Interior,
                  "collinear initial vertices");
        },
        "review polygon basis");
    check(failures == 0, "review regressions failed");
}
void numerical_precision_regressions() {
    int failures = 0;
    auto run = [&](auto test, const char *name) {
        try {
            test();
        } catch (const std::exception &e) {
            ++failures;
            std::cerr << name << ": " << e.what() << '\n';
        }
    };
    run(
        [] {
            for (double scale : {1e-13, 1., 1e6, 1e7, 1e8, 1e150}) {
                auto m = mesh({{0, 0, 0}, {scale, 2 * scale, 3 * scale}, {3 * scale, 3 * scale, 5 * scale}, {2 * scale, scale, 2 * scale}}, {0, 4},
                              BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
                PointLocationOptions opts;
                opts.tolerance = 0;
                PointLocator loc(m, std::vector<index_t>{1}, opts);
                std::vector<index_t> ids{7, 8};
                std::vector<double> xyz{1.5 * scale, 1.5 * scale, 1.5 * scale, 1.5 * scale, 2.5 * scale, (2.5 + 1e-6) * scale};
                auto r = loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
                check(r[0].status == PointLocationStatus::Interior, "oblique interior");
                check(r[1].status == PointLocationStatus::Unmapped, "oblique outside plane");
            }
        },
        "large oblique cells");
    run(
        [] {
            for (double scale : {1e-13, 1e-8, 1., 1e6}) {
                auto m = mesh({{0, 0, 0}, {2 * scale, 0, 0}, {.5 * scale, .5 * scale, 0}, {0, 2 * scale, 0}}, {0, 4},
                              BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
                rejects([&] { PointLocator bad(m, std::vector<index_t>{1}); }, "scaled concave polygon must reject");
                auto warped = mesh({{0, 0, 0}, {scale, 0, 0}, {scale, scale, scale * 1e-5}, {0, scale, 0}}, {0, 4}, BoundaryModel::CartesianStraight,
                                   CoordinateSystem::Cartesian3D);
                rejects([&] { PointLocator bad(warped, std::vector<index_t>{1}); }, "scaled warped polygon must reject");
            }
        },
        "small invalid geometry");
    run(
        [] {
            auto m = mesh({{10, 40, 0}, {12, 40, 0}, {12, 42, 0}, {10, 42, 0}}, {0, 4}, BoundaryModel::GreatCircle, CoordinateSystem::SphericalDeg);
            PointLocationOptions opts;
            opts.tolerance = 0;
            PointLocator loc(m, std::vector<index_t>{42}, opts);
            std::vector<index_t> ids{1, 2, 3, 4, 5};
            std::vector<double> xyz{10, 12, 12, 10, 9.999999, 40, 40, 42, 42, 41, 0, 0, 0, 0, 0};
            auto r = loc.locate(points(ids, xyz, CoordinateSystem::SphericalDeg));
            for (size_t k = 0; k < 4; ++k)
                check(r[k].cell_id == 42 && r[k].status == PointLocationStatus::Boundary, "exact spherical vertex at zero tolerance");
            check(r[4].status == PointLocationStatus::Unmapped, "zero tolerance keeps outside point unmapped");
        },
        "zero tolerance vertices");
    check(failures == 0, "numerical precision regressions");
}
void distributed_precision_regressions(int rank, int size) {
    std::vector<std::array<double, 3>> vertices;
    std::vector<index_t> offsets{0}, cells;
    if (rank < 2) {
        vertices = {{10, 40, 0}, {12, 40, 0}, {12, 42, 0}, {10, 42, 0}};
        offsets.push_back(4);
        cells.push_back(rank == 0 ? 90 : 10);
    }
    auto m = mesh(vertices, offsets, BoundaryModel::GreatCircle, CoordinateSystem::SphericalDeg);
    PointLocationOptions opts;
    opts.tolerance = 0;
    DistributedPointLocator loc(m, cells, MPI_COMM_WORLD, opts);
    std::vector<index_t> ids{rank * 10, rank * 10 + 1, rank * 10 + 2, rank * 10 + 3};
    std::vector<double> xyz{10, 12, 12, 10, 40, 40, 42, 42, 0, 0, 0, 0};
    auto r = loc.locate(points(ids, xyz, CoordinateSystem::SphericalDeg));
    for (const auto &result : r)
        check(result.cell_id == (size > 1 ? 10 : 90) && result.owner_rank == (size > 1 ? 1 : 0) && result.status == PointLocationStatus::Boundary,
              "distributed zero-tolerance stable tie");
    for (double scale : {1e-13, 1e6, 1e150}) {
        vertices.clear();
        offsets = {0};
        cells.clear();
        if (rank == 0) {
            vertices = {{0, 0, 0}, {scale, 2 * scale, 3 * scale}, {3 * scale, 3 * scale, 5 * scale}, {2 * scale, scale, 2 * scale}};
            offsets.push_back(4);
            cells.push_back(7);
        }
        auto planar = mesh(vertices, offsets, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
        DistributedPointLocator oblique(planar, cells, MPI_COMM_WORLD, opts);
        ids = {rank};
        xyz = {1.5 * scale, 1.5 * scale, 2.5 * scale};
        auto result = oblique.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
        check(result[0].cell_id == 7 && result[0].status == PointLocationStatus::Interior, "distributed scaled oblique cell");
    }
}
void collective_failures(int rank, int size) {
    auto m = mesh(rank == 0 ? std::vector<std::array<double, 3>>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}} : std::vector<std::array<double, 3>>{},
                  rank == 0 ? std::vector<index_t>{0, 4} : std::vector<index_t>{0}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    std::vector<index_t> cells = rank == 0 ? std::vector<index_t>{42} : std::vector<index_t>{};
    DistributedPointLocator loc(m, cells, MPI_COMM_WORLD);
    std::vector<index_t> ids = rank == 0 ? std::vector<index_t>{1} : std::vector<index_t>{};
    std::vector<double> xyz = rank == 0 ? std::vector<double>{NAN, .5, 0} : std::vector<double>{};
    rejects([&] { (void)loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D)); }, "nonfinite query collective failure");
    if (size > 1) {
        ids = {123};
        xyz = {.5, .5, 0};
        rejects([&] { (void)loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D)); }, "duplicate remote point ID");
    }
    PointLocationOptions opts;
    opts.max_import_bytes = 64;
    ids = rank == 0 ? std::vector<index_t>{1, 2, 3, 4, 5, 6, 7, 8, 9} : std::vector<index_t>{};
    xyz.assign(ids.size() * 3, .5);
    DistributedPointLocator limited(m, cells, MPI_COMM_WORLD, opts);
    rejects([&] { (void)limited.locate(points(ids, xyz, CoordinateSystem::Cartesian3D)); }, "directory budget collective failure");
    opts.max_message_bytes = 1;
    rejects([&] { DistributedPointLocator bad(m, cells, MPI_COMM_WORLD, opts); }, "invalid message budget");
    if (size > 1) {
        opts = {};
        if (rank == 0) opts.tolerance = 1e-9;
        rejects([&] { DistributedPointLocator bad(m, cells, MPI_COMM_WORLD, opts); }, "inconsistent options");
        auto duplicate = mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 3}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
        rejects([&] { DistributedPointLocator bad(duplicate, std::vector<index_t>{42}, MPI_COMM_WORLD); }, "duplicate remote cell ID");
    }
    auto empty = mesh({}, {0}, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
    DistributedPointLocator no_mesh(empty, {}, MPI_COMM_WORLD);
    ids = {100 + rank};
    xyz = {.5, .5, 0};
    check(no_mesh.locate(points(ids, xyz, CoordinateSystem::Cartesian3D))[0].cell_id == -1, "globally empty mesh");
}
void distributed_tests(int rank, int size) {
    for (int layout = 0; layout < 2; ++layout) {
        std::vector<std::array<double, 3>> vertices;
        std::vector<index_t> off{0}, cells;
        for (int c = 0; c < 3; ++c)
            if ((c + layout) % size == rank) {
                vertices.insert(vertices.end(), {{double(c), 0, 0}, {double(c + 1), 0, 0}, {double(c + 1), 1, 0}, {double(c), 1, 0}});
                off.push_back(vertices.size());
                cells.push_back(100 - 10 * c);
            }
        auto m = mesh(vertices, off, BoundaryModel::CartesianStraight, CoordinateSystem::Cartesian3D);
        PointLocationOptions opts;
        opts.max_message_bytes = 48;
        opts.query_batch_size = 2;
        DistributedPointLocator loc(m, cells, MPI_COMM_WORLD, opts);
        std::vector<index_t> ids;
        std::vector<double> xs;
        for (int p = 9; p >= 0; --p)
            if ((p + 1 + layout) % size == rank) {
                ids.push_back(1000 + p);
                xs.push_back(p < 8 ? .5 : 1.);
            }
        std::vector<double> xyz = xs;
        xyz.insert(xyz.end(), ids.size(), .5);
        xyz.insert(xyz.end(), ids.size(), 0.);
        auto result = loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
        check(result.size() == ids.size(), "distributed result count");
        for (size_t k = 0; k < ids.size(); ++k) {
            int c = ids[k] >= 1008 ? 1 : 0;
            check(result[k].point_id == ids[k] && result[k].cell_id == 100 - 10 * c, "layout invariant IDs");
            check(result[k].owner_rank == (c + layout) % size, "owner routing");
        }
        std::vector<index_t> empty;
        std::vector<double> no_coords;
        check(loc.locate(points(empty, no_coords, CoordinateSystem::Cartesian3D)).empty(), "empty query");
        if (rank == 0) {
            ids = {999, 999};
            xyz = {.5, .5, .5, .5, 0, 0};
        } else {
            ids.clear();
            xyz.clear();
        }
        bool threw = false;
        try {
            (void)loc.locate(points(ids, xyz, CoordinateSystem::Cartesian3D));
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw, "duplicate IDs fail collectively");
    }
}
void larger_partitions(int rank, int size) {
    constexpr int nx = 128, ny = 8, count = nx * ny;
    for (bool spherical : {false, true})
        for (bool fragmented : {false, true}) {
            std::vector<std::array<double, 3>> v;
            std::vector<index_t> off{0}, cells;
            for (int c = count - 1; c >= 0; --c) {
                int owner = fragmented ? c % size : static_cast<int>(static_cast<long long>(c) * size / count);
                if (owner != rank) continue;
                double x = c % nx, y = c / nx;
                v.insert(v.end(), {{x, y, 0}, {x + 1, y, 0}, {x + 1, y + 1, 0}, {x, y + 1, 0}});
                off.push_back(v.size());
                cells.push_back(100000 + c * 3);
            }
            auto system = spherical ? CoordinateSystem::SphericalDeg : CoordinateSystem::Cartesian3D;
            auto model = spherical ? BoundaryModel::ConstantLatitude : BoundaryModel::CartesianStraight;
            auto m = mesh(v, off, model, system);
            std::vector<index_t> ids;
            std::vector<double> x, y;
            for (int i = 0; i < count * 2; ++i)
                if ((i + 3) % size == rank) {
                    ids.push_back(200000 + i * 7);
                    x.push_back((i / 2) % nx + .25);
                    y.push_back((i / 2) / nx + .5);
                }
            std::vector<double> xyz = x;
            xyz.insert(xyz.end(), y.begin(), y.end());
            xyz.insert(xyz.end(), x.size(), 0.);
            auto start = std::chrono::steady_clock::now();
            DistributedPointLocator locator(m, cells, MPI_COMM_WORLD);
            auto setup = std::chrono::steady_clock::now();
            auto result = locator.locate(points(ids, xyz, system));
            auto end = std::chrono::steady_clock::now();
            for (size_t k = 0; k < ids.size(); ++k) {
                auto c = (ids[k] - 200000) / 7 / 2;
                check(result[k].point_id == ids[k] && result[k].cell_id == 100000 + 3 * c, "large partition oracle");
                check(result[k].status == PointLocationStatus::Interior, "large interior status");
                auto owner = fragmented ? c % size : c * size / count;
                check(result[k].owner_rank == owner && cells.size() <= count, "large owner");
            }
            double timings[2] = {std::chrono::duration<double>(setup - start).count(), std::chrono::duration<double>(end - setup).count()}, maxima[2];
            MPI_Allreduce(timings, maxima, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0)
                std::cout << "point_locator ranks=" << size << " spherical=" << spherical << " fragmented=" << fragmented << " cells=" << count
                          << " points=" << count * 2 << " max_setup_s=" << maxima[0] << " max_query_s=" << maxima[1] << '\n';
        }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    long rss = usage.ru_maxrss, max_rss = 0;
    MPI_Allreduce(&rss, &max_rss, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "point_locator peak_rank_rss_kib=" << max_rss << '\n';
}
int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Kokkos::initialize(argc, argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int failed = 0;
    try {
        numerical_precision_regressions();
        review_regressions();
        local_tests();
        geometry_failures();
        distributed_tests(rank, size);
        distributed_precision_regressions(rank, size);
        collective_failures(rank, size);
        larger_partitions(rank, size);
    } catch (const std::exception &e) {
        std::cerr << rank << ": " << e.what() << '\n';
        failed = 1;
    }
    int global;
    MPI_Allreduce(&failed, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    Kokkos::finalize();
    MPI_Finalize();
    return global;
}
