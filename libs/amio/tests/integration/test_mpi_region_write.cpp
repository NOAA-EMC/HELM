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
#include "mpi_comm_tracker.hpp"
#ifdef AMIO_TEST_TENSORSTORE
#include <tensorstore/open.h>
#include <tensorstore/tensorstore.h>
#endif

/** @brief Exercise shared outputs with uneven slabs and verify independent readback. */
int main(int argc, char **argv) {
    const bool single = argc > 1 && std::string(argv[1]) == "single";
    const bool synchronous = argc > 1 && std::string(argv[1]) == "sync";
    const bool fallback = argc > 1 && std::string(argv[1]) == "nczarr";
    const bool prepare_schema = argc > 1 && std::string(argv[1]) == "prepared-schema";
    const bool mismatch_schema = argc > 1 && std::string(argv[1]) == "schema-mismatch";
    const bool late_schema = argc > 1 && std::string(argv[1]) == "late-schema";
    const bool batch_invalid = argc > 1 && std::string(argv[1]) == "batch-invalid";
    const bool batch_mismatch = argc > 1 && std::string(argv[1]) == "batch-mismatch";
    const bool batch = argc > 1 && (std::string(argv[1]) == "batch" || batch_invalid || batch_mismatch);
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
        const auto baseline_comms = amio_test::live_duplicates();
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
        } else if (single && ranks > 1) {
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
            std::vector<std::string> schema_names{"field_0", "field_1", "field_2"};
            std::vector<amio_write_variable_t> schema(schema_names.size());
            for (std::size_t i = 0; i < schema.size(); ++i) {
                schema[i].struct_size = static_cast<std::uint32_t>(sizeof(amio_write_variable_t));
                schema[i].version = AMIO_WRITE_VARIABLE_VERSION;
                schema[i].name = schema_names[i].c_str();
                schema[i].dtype = AMIO_DTYPE_F32;
                schema[i].rank = 2;
                schema[i].global_extents[0] = rows;
                schema[i].global_extents[1] = 3;
            }
            if (prepare_schema || mismatch_schema || batch) {
                if (mismatch_schema) {
                    const auto *local_schema = rank == 1 ? nullptr : schema.data();
                    check(amio_prepare_write_schema(dataset, local_schema, schema.size()) == AMIO_ERR_INVALID_INPUT,
                          "agree on invalid schema input across ranks");
                    if (rank == 1) schema[1].name = "different_field";
                }
                const auto prepared = amio_prepare_write_schema(dataset, schema.data(), schema.size());
                check(prepared == (mismatch_schema ? AMIO_ERR_SCHEMA_MISMATCH : AMIO_OK),
                      mismatch_schema ? "reject rank schema mismatch collectively" : "prepare shared schema");
                if (prepare_schema) {
                    auto invalid_region = amio_write_region_t{};
                    invalid_region.rank = 2;
                    invalid_region.global_extents[0] = rows + 1;
                    invalid_region.global_extents[1] = 3;
                    amio_shape_t invalid_shape{};
                    invalid_shape.rank = 2;
                    invalid_shape.extents[0] = rank + 1;
                    invalid_shape.extents[1] = 3;
                    std::vector<float> invalid_data((rank + 1) * 3);
                    amio_io_handle invalid_io = nullptr;
                    check(amio_write_region(dataset, "field_0", invalid_data.data(), AMIO_DTYPE_F32, &invalid_shape, &invalid_region, &invalid_io) ==
                              AMIO_ERR_SCHEMA_MISMATCH,
                          "reject write that differs from prepared global shape");
                }
            }
            if (batch && datasets.size() > 1)
                for (std::size_t file = 1; file < datasets.size(); ++file)
                    check(amio_prepare_write_schema(datasets[file], schema.data(), schema.size()) == AMIO_OK,
                          "prepare batch schema for second output");
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
            std::vector<std::vector<float>> batch_data(3, std::vector<float>(data.size()));
            std::vector<amio_region_write_t> batch_writes(3);
            const auto fill_batch = [&](int step, std::size_t file) {
                for (int field = 0; field < 3; ++field) {
                    for (int i = 0; i < rank + 1; ++i)
                        for (int j = 0; j < 3; ++j)
                            batch_data[field][static_cast<std::size_t>(i * 3 + j)] =
                                100000 * static_cast<float>(file) + 1000 * step + 100 * field + (offset + i) * 3 + j;
                    batch_writes[field] = {schema_names[field].c_str(), batch_data[field].data(), AMIO_DTYPE_F32, &shape, &region};
                }
            };
            if (batch_invalid || batch_mismatch) {
                fill_batch(0, 0);
                auto rejected_batch = batch_writes;
                if (batch_invalid && rank == 1) rejected_batch[0].host_data = nullptr;
                if (batch_mismatch && rank == 1) std::swap(rejected_batch[0], rejected_batch[1]);
                amio_io_handle rejected_io = nullptr;
                const auto expected = batch_invalid ? AMIO_ERR_INVALID_INPUT : AMIO_ERR_SCHEMA_MISMATCH;
                check(amio_write_region_batch(dataset, rejected_batch.data(), rejected_batch.size(), &rejected_io) == expected,
                      batch_invalid ? "reject invalid batch input collectively" : "reject ordered batch schema mismatch collectively");
                check(rejected_io == nullptr, "rejected batch returns no I/O handle");
            }
            std::vector<amio_io_handle> pending;
            for (int step = 0; step < 12; ++step) {
                for (std::size_t file = 0; file < datasets.size(); ++file) {
                    amio_io_handle io = nullptr;
                    if (batch) {
                        fill_batch(step, file);
                        check(amio_write_region_batch(datasets[file], batch_writes.data(), batch_writes.size(), &io) == AMIO_OK,
                              "enqueue region batch");
                        for (auto &values : batch_data) std::fill(values.begin(), values.end(), -999);
                    } else {
                        for (int i = 0; i < rank + 1; ++i)
                            for (int j = 0; j < 3; ++j) data[i * 3 + j] = 100000 * file + 1000 * step + (offset + i) * 3 + j;
                        const std::string name = zarr ? "field" : "field_" + std::to_string(step % 3);
                        check(amio_write_region(datasets[file], name.c_str(), data.data(), AMIO_DTYPE_F32, &shape, &region, &io) == AMIO_OK,
                              "enqueue interleaved output region");
                        std::fill(data.begin(), data.end(), -999);  // Host can reuse immediately.
                    }
                    if (late_schema && step == 0 && file == 0) {
                        check(amio_prepare_write_schema(dataset, schema.data(), schema.size()) == AMIO_ERR_INVALID_POLICY,
                              "reject schema preparation after writes begin");
                    }
                    pending.push_back(io);
                }
            }
            for (auto io : pending) check(amio_wait(io, 30000) == AMIO_OK, "write completes");
            for (auto output : datasets) {
                check(amio_flush(output, 30000) == AMIO_OK, "flush shared output");
                check(amio_close_dataset(output) == AMIO_OK, "close shared output");
            }
            check(amio_test::live_duplicates() == baseline_comms, "closed outputs release communicators before core finalization");
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
                            const int expected_step = batch ? 11 : 9 + field;
                            const int field_offset = batch ? 100 * field : 0;
                            for (int i = 0; i < rows * 3; ++i)
                                check(actual[i] == 100000 * file + 1000 * expected_step + field_offset + i,
                                      "regions preserved in both ordered outputs");
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
        if (rank == 0 && !(single && ranks > 1)) {
            std::filesystem::remove_all(path);
            if (!zarr) std::filesystem::remove_all(path + ".second.nc");
        }
    }
    MPI_Finalize();
}
