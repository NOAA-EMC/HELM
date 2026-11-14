// SPDX-License-Identifier: Apache-2.0
// CECE-oriented geometry example: application attributes never enter AXIS.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <axis/distributed/point_locator.hpp>
#include <chrono>
#include <iostream>
#include <vector>

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
        using namespace axis;
        using namespace axis::topology;
        using namespace axis::solver;
        std::vector<index_t> cell_ids;
        for (int cell = 0; cell < 3; ++cell)
            if (cell % size == rank) cell_ids.push_back(100 + cell);
        const auto n = cell_ids.size();
        Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> coords("corners", 4 * n, 2);
        Kokkos::View<index_t *, Kokkos::HostSpace> offsets("offsets", n + 1), indices("indices", 4 * n);
        for (size_t k = 0; k < n; ++k) {
            double west = cell_ids[k] - 100;
            coords(4 * k, 0) = west;
            coords(4 * k + 1, 0) = west + 1;
            coords(4 * k + 2, 0) = west + 1;
            coords(4 * k + 3, 0) = west;
            coords(4 * k, 1) = 40;
            coords(4 * k + 1, 1) = 40;
            coords(4 * k + 2, 1) = 41;
            coords(4 * k + 3, 1) = 41;
            offsets(k) = 4 * k;
            for (int j = 0; j < 4; ++j) indices(4 * k + j) = 4 * k + j;
        }
        offsets(n) = 4 * n;
        GeometryMetadata geometry;
        geometry.boundary_model = BoundaryModel::ConstantLatitude;
        UnstructuredMesh<Kokkos::HostSpace> mesh(coords, offsets, indices, CoordinateSystem::SphericalDeg, {}, {}, geometry);
        // These arrays belong to the application. IDs do not depend on rank.
        std::vector<index_t> source_ids;
        std::vector<double> source_height;
        for (int source = 0; source < 8; ++source)
            if ((source + 1) % size == rank) {
                source_ids.push_back(1000 + source);
                source_height.push_back(50 + 10 * source);
            }
        // All eight distinct sources share the same coordinates and host cell.
        std::vector<double> xy(source_ids.size() * 2);
        for (size_t k = 0; k < source_ids.size(); ++k) {
            xy[k] = .5;
            xy[k + source_ids.size()] = 40.5;
        }
        auto start = std::chrono::steady_clock::now();
        axis::distributed::DistributedPointLocator locator(mesh, cell_ids, MPI_COMM_WORLD);
        auto setup = std::chrono::steady_clock::now();
        auto mapping = locator.locate({source_ids, field_view<const double, 2>(xy.data(), source_ids.size(), 2), CoordinateSystem::SphericalDeg});
        auto end = std::chrono::steady_clock::now();
        for (size_t k = 0; k < mapping.size(); ++k) {
            const auto &result = mapping[k];
            if (result.point_id != source_ids[k] || result.cell_id != 100) throw std::runtime_error("incorrect source mapping");
            std::cout << "rank=" << rank << " source=" << result.point_id << " height=" << source_height[k] << " host_cell=" << result.cell_id
                      << " host_rank=" << result.owner_rank << " host_local_index=" << result.owner_local_index << '\n';
        }
        if (rank == 0)
            std::cout << "setup_seconds=" << std::chrono::duration<double>(setup - start).count()
                      << " query_seconds=" << std::chrono::duration<double>(end - setup).count() << '\n';
        // CECE can copy mapping columns to device arrays, translate the mesh
        // local index using its own cell ordering, and retain each source's
        // independent state. Remote host indices are never local array indices.
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        failed = 1;
    }
    int global;
    MPI_Allreduce(&failed, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    Kokkos::finalize();
    MPI_Finalize();
    return global;
}
