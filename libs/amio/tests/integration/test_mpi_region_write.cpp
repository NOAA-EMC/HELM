// Real shared-file/store round trips with uneven MPI slabs and four workers.
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

int main(int argc, char **argv) {
    const bool single = argc > 1 && std::string(argv[1]) == "single";
    const bool fallback = argc > 1 && std::string(argv[1]) == "nczarr";
    const bool zarr = fallback || (argc > 1 && std::string(argv[1]) == "zarr3");
    int provided = 0;
    MPI_Init_thread(&argc, &argv, single ? MPI_THREAD_SINGLE : MPI_THREAD_MULTIPLE, &provided);
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
        check(single || provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE required");
        const int rows = ranks * (ranks + 1) / 2;  // rank r owns r+1 rows
        const int offset = rank * (rank + 1) / 2;
        const std::string path = (std::filesystem::current_path() / (zarr ? "mpi-regions.zarr" : "mpi-regions.nc")).string();
        if (fallback) {
            if (rank == 0) std::filesystem::remove_all(path);
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
        }
        std::string manifest = "backend: " + std::string(zarr ? "zarr3" : "netcdf4") +
                               "\nstaging_pool: {buffer_count: 8, buffer_capacity_bytes: 4096}\nworker_pool: {threads: 4}\n";
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
        } else if (single) {
            check(opened == AMIO_ERR_THREADING_UNSUPPORTED && dataset == nullptr, "reject insufficient thread support before file creation");
        } else {
            check(opened == AMIO_OK, "collectively open shared output");
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
                for (int i = 0; i < rank + 1; ++i)
                    for (int j = 0; j < 3; ++j) data[i * 3 + j] = 1000 * step + (offset + i) * 3 + j;
                amio_io_handle io = nullptr;
                const std::string name = zarr ? "field" : "field_" + std::to_string(step % 3);
                check(amio_write_region(dataset, name.c_str(), data.data(), AMIO_DTYPE_F32, &shape, &region, &io) == AMIO_OK, "enqueue region");
                std::fill(data.begin(), data.end(), -999);  // Host can reuse immediately.
                pending.push_back(io);
            }
            for (auto io : pending) check(amio_wait(io, 30000) == AMIO_OK, "background write completes");
            check(amio_flush(dataset, 30000) == AMIO_OK, "flush shared output");
            check(amio_close_dataset(dataset) == AMIO_OK, "close shared output");
            (void)halo::allreduce(comm, std::vector<int>{1}, MPI_SUM);
            if (rank == 0) {
                std::vector<float> actual(rows * 3);
                if (!zarr) {
                    int ncid = -1;
                    check(nc_open(path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR, "reopen output independently");
                    for (int field = 0; field < 3; ++field) {
                        int varid = -1;
                        check(nc_inq_varid(ncid, ("field_" + std::to_string(field)).c_str(), &varid) == NC_NOERR, "find variable");
                        check(nc_get_var_float(ncid, varid, actual.data()) == NC_NOERR, "read global field");
                        for (int i = 0; i < rows * 3; ++i) check(actual[i] == 1000 * (9 + field) + i, "all regions preserved in final ordered write");
                    }
                    check(nc_close(ncid) == NC_NOERR, "close independent reader");
                }
#ifdef AMIO_TEST_TENSORSTORE
                else {
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
        if (rank == 0 && !single) std::filesystem::remove_all(path);
    }
    MPI_Finalize();
}
