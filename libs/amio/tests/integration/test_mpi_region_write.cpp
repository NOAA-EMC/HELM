/** @file
 * @brief Public MPI region writes across multiple outputs, including zero-worker mode and independent Zarr reads.
 */
#include <netcdf.h>

#include <algorithm>
#include <filesystem>
#include <halo/collectives.hpp>
#include <logs/logger.hpp>
#include <string>
#include <vector>

#include "amio/amio.h"
#ifdef AMIO_TEST_TENSORSTORE
#include <tensorstore/open.h>
#include <tensorstore/tensorstore.h>
#endif

/** @brief Exercise shared outputs with uneven slabs and verify independent readback. */
int main(int argc, char **argv) {
    const bool single = argc > 1 && std::string(argv[1]) == "single";
    const bool synchronous = argc > 1 && std::string(argv[1]) == "sync";
    const bool fallback = argc > 1 && std::string(argv[1]) == "nczarr";
    const bool zarr = fallback || (argc > 1 && std::string(argv[1]) == "zarr3");
    int provided = 0;
    MPI_Init_thread(&argc, &argv, (single || synchronous) ? MPI_THREAD_SINGLE : MPI_THREAD_MULTIPLE, &provided);
    halo::Environment::initialize();
    {
        halo::Communicator comm(MPI_COMM_WORLD);
        logs::Logger logger;
        logger.configure_communicator(comm.handle());
        auto check = [&](bool ok, const std::string &message) {
            if (!ok) {
                logger.log(logs::Severity_Level::ERROR, message);
                MPI_Abort(comm.handle(), 1);
            }
        };
        const int rank = comm.rank(), ranks = comm.size();
        check(single || synchronous || provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE required");
        const int rows = ranks * (ranks + 1) / 2;  // rank r owns r+1 rows
        const int offset = rank * (rank + 1) / 2;
        const std::string path = (std::filesystem::current_path() / (zarr ? "mpi-regions.zarr" : "mpi-regions.nc")).string();
        if (fallback) {
            if (rank == 0) std::filesystem::remove_all(path);
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
        }
        std::string manifest =
            "backend: " + std::string(zarr ? "zarr3" : "netcdf4") +
            "\nstaging_pool: {buffer_count: 8, buffer_capacity_bytes: 4096}\nworker_pool: {threads: " + std::string(synchronous ? "0" : "4") + "}\n";
        if (zarr) {
            // One shard shared by all ranks exercises concurrent shard updates.
            manifest += "uri: " + path + "/\narray_shape: [" + std::to_string(rows) + ", 3]\nchunk_shape: [1, 3]\nshard_shape: [" +
                        std::to_string(rows) + ", 3]\ncodec: zstandard\ndtype: float32\n";
        } else {
            manifest += "path: " + path + "\ndata_model: enhanced\ncodec: {active_codec: zstandard, lossless_allow_list: [zstandard]}\n";
        }
        amio_core_handle core = nullptr;
        check(amio_init_from_string(manifest.c_str(), "yaml", &core) == AMIO_OK, "init");
        amio_dataset_handle dataset = nullptr;
        const auto opened = amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_WRITE, &dataset);
        if (fallback) {
            check(opened == AMIO_ERR_INVALID_INPUT && dataset == nullptr, "reject shared NCZarr before modifying storage");
            check(!std::filesystem::exists(path), "rejected NCZarr open creates no store");
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
            if (rank == 0) {
                // Build a tiny input without collective AMIO output setup.
                const std::string uri = "file://" + path + "#mode=nczarr,file";
                int ncid = -1, dimid = -1, varid = -1;
                check(nc_create(uri.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid) == NC_NOERR, "create NCZarr input fixture");
                check(nc_def_dim(ncid, "x", 1, &dimid) == NC_NOERR, "define fixture dimension");
                check(nc_def_var(ncid, "field", NC_FLOAT, 1, &dimid, &varid) == NC_NOERR, "define fixture field");
                check(nc_close(ncid) == NC_NOERR, "close fixture");
                check(amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_READ, &dataset) == AMIO_OK,
                      "rank-zero-only NCZarr read open");
                check(amio_close_dataset(dataset) == AMIO_OK, "rank-zero-only NCZarr read close");
            }
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
        } else if (single) {
            check(opened == AMIO_ERR_THREADING_UNSUPPORTED && dataset == nullptr, "reject insufficient thread support before file creation");
        } else {
            check(opened == AMIO_OK, "collectively open shared output");
            std::vector<amio_dataset_handle> datasets{dataset};
            const std::string second_path = path + ".second.nc";
            if (!zarr) {
                std::string second_manifest = manifest;
                second_manifest.replace(second_manifest.find(path), path.size(), second_path);
                amio_dataset_handle second = nullptr;
                check(amio_open_dataset_from_string(core, second_manifest.c_str(), "yaml", AMIO_MODE_WRITE, &second) == AMIO_OK,
                      "open second active NetCDF output");
                datasets.push_back(second);
            }
            amio_shape_t shape{};
            shape.rank = 2;
            shape.extents[0] = rank + 1;
            shape.extents[1] = 3;
            amio_write_region_t region{};
            region.rank = 2;
            region.global_extents[0] = rows;
            region.global_extents[1] = 3;
            region.offsets[0] = offset;
            std::vector<float> data((rank + 1) * 3);
            std::vector<amio_io_handle> pending;
            for (int step = 0; step < 12; ++step) {
                for (std::size_t file = 0; file < datasets.size(); ++file) {
                    for (int i = 0; i < rank + 1; ++i)
                        for (int j = 0; j < 3; ++j) data[i * 3 + j] = 100000 * file + 1000 * step + (offset + i) * 3 + j;
                    amio_io_handle io = nullptr;
                    const std::string name = zarr ? "field" : "field_" + std::to_string(step % 3);
                    check(amio_write_region(datasets[file], name.c_str(), data.data(), AMIO_DTYPE_F32, &shape, &region, &io) == AMIO_OK,
                          "enqueue interleaved output region");
                    std::fill(data.begin(), data.end(), -999);  // Host can reuse immediately.
                    pending.push_back(io);
                }
            }
            for (auto io : pending) check(amio_wait(io, 30000) == AMIO_OK, "write completes");
            for (auto output : datasets) {
                check(amio_flush(output, 30000) == AMIO_OK, "flush shared output");
                check(amio_close_dataset(output) == AMIO_OK, "close shared output");
            }
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
            if (rank == 0) {
                std::vector<float> actual(rows * 3);
                if (!zarr) {
                    for (std::size_t file = 0; file < datasets.size(); ++file) {
                        int ncid = -1;
                        const auto &output_path = file == 0 ? path : second_path;
                        check(nc_open(output_path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR, "reopen output independently");
                        for (int field = 0; field < 3; ++field) {
                            int varid = -1;
                            check(nc_inq_varid(ncid, ("field_" + std::to_string(field)).c_str(), &varid) == NC_NOERR, "find variable");
                            check(nc_get_var_float(ncid, varid, actual.data()) == NC_NOERR, "read global field");
                            for (int i = 0; i < rows * 3; ++i)
                                check(actual[i] == 100000 * file + 1000 * (9 + field) + i, "regions preserved in both ordered outputs");
                        }
                        check(nc_close(ncid) == NC_NOERR, "close independent reader");
                    }
                }
#ifdef AMIO_TEST_TENSORSTORE
                else {
                    amio_dataset_handle input = nullptr;
                    check(amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_READ, &input) == AMIO_OK,
                          "rank-zero-only TensorStore read open");
                    check(amio_close_dataset(input) == AMIO_OK, "rank-zero-only TensorStore read close");
                    auto opened_store = tensorstore::Open({{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", path + "/"}}}},
                                                          tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read)
                                            .result();
                    check(opened_store.ok(), "reopen Zarr independently");
                    auto array = tensorstore::UnownedToShared(tensorstore::Array(actual.data(), {rows, 3}, tensorstore::c_order));
                    check(tensorstore::Read(opened_store.value(), array).result().ok(), "read Zarr global field");
                    for (int i = 0; i < rows * 3; ++i) check(actual[i] == 11000 + i, "concurrent updates to shared Zarr shard preserved");
                }
#endif
            }
        }
        check(amio_finalize(core) == AMIO_OK, "finalize");
        (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
        if (rank == 0 && !single) {
            std::filesystem::remove_all(path);
            if (!zarr) std::filesystem::remove_all(path + ".second.nc");
        }
    }
    MPI_Finalize();
}
