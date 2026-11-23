#include <amio/amio.h>
#include <mpi.h>
#include <netcdf.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
}  // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check(provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE required");
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const std::string field = argc > 1 && std::string(argv[1]) == "dotted" ? "PM2.5" : "tracer";
    const bool invalid = argc > 1 && std::string(argv[1]) == "invalid";
    const int rows = ranks * (ranks + 1) / 2, offset = rank * (rank + 1) / 2;
    const std::string path = (std::filesystem::current_path() / "named-dimensions.nc").string();
    const std::string manifest = "backend: netcdf4\npath: " + path +
                                 "\ndata_model: enhanced\nworker_pool: {threads: 2}\n"
                                 "staging_pool: {buffer_count: 2, buffer_capacity_bytes: 4096}\n"
                                 "unlimited_dimensions: [time]\nvariable_dimensions:\n"
                                 "  time: [time]\n  lev: [lev]\n  lat: [lat]\n  lon: [lon]\n  \"" +
                                 field + "\": " + (invalid ? "[time, lon]\n" : "[time, lev, lat, lon]\n");
    amio_core_handle core = nullptr;
    amio_dataset_handle dataset = nullptr;
    check(amio_init_from_string(manifest.c_str(), "yaml", &core) == AMIO_OK, "init");
    check(amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_WRITE, &dataset) == AMIO_OK, "open");
    std::vector<amio_write_variable_t> schema(5);
    const char *names[] = {"time", "lev", "lat", "lon", field.c_str()};
    const int lengths[] = {1, 1, rows, 3};
    for (int v = 0; v < 5; ++v) {
        schema[v].struct_size = sizeof(amio_write_variable_t);
        schema[v].version = AMIO_WRITE_VARIABLE_VERSION;
        schema[v].name = names[v];
        schema[v].dtype = AMIO_DTYPE_F64;
        schema[v].rank = v < 4 ? 1 : 4;
        if (v < 4)
            schema[v].global_extents[0] = lengths[v];
        else
            for (int d = 0; d < 4; ++d) schema[v].global_extents[d] = lengths[d];
    }
    const auto prepared = amio_prepare_write_schema(dataset, schema.data(), schema.size());
    if (invalid) {
        check(prepared != AMIO_OK, "reject explicit dimension count different from variable rank");
    } else {
        check(prepared == AMIO_OK, "prepare named schema");
        amio_shape_t local{};
        local.rank = 4;
        local.extents[0] = local.extents[1] = 1;
        local.extents[2] = rank + 1;
        local.extents[3] = 3;
        amio_write_region_t region{};
        region.rank = 4;
        for (int d = 0; d < 4; ++d) region.global_extents[d] = lengths[d];
        region.offsets[2] = offset;
        std::vector<double> values((rank + 1) * 3);
        for (size_t i = 0; i < values.size(); ++i) values[i] = offset * 3 + i;
        amio_region_write_t write{field.c_str(), values.data(), AMIO_DTYPE_F64, &local, &region};
        amio_io_handle io = nullptr;
        check(amio_write_region_batch(dataset, &write, 1, &io) == AMIO_OK, "submit named region batch");
        check(amio_wait(io, 0) == AMIO_OK, "complete named region batch");
    }
    check(amio_close(dataset) == AMIO_OK, "close");
    check(amio_finalize(core) == AMIO_OK, "finalize");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !invalid) {
        int ncid = -1, tracer = -1, time = -1, unlimited = -1;
        check(nc_open(path.c_str(), NC_WRITE, &ncid) == NC_NOERR, "reopen");
        check(nc_inq_unlimdim(ncid, &unlimited) == NC_NOERR && unlimited >= 0, "time must remain unlimited after prepared writes");
        check(nc_inq_varid(ncid, field.c_str(), &tracer) == NC_NOERR, "find tracer");
        int dims[4]{};
        check(nc_inq_vardimid(ncid, tracer, dims) == NC_NOERR, "read dimensions");
        check(dims[0] == unlimited, "tracer uses unlimited time dimension");
        for (int d = 0; d < 4; ++d) {
            char name[NC_MAX_NAME + 1]{};
            size_t len = 0;
            check(nc_inq_dim(ncid, dims[d], name, &len) == NC_NOERR, "read dimension");
            check(std::string(name) == names[d] && len == static_cast<size_t>(lengths[d]), "explicit axis name and length");
        }
        std::vector<double> actual(rows * 3);
        check(nc_get_var_double(ncid, tracer, actual.data()) == NC_NOERR, "read tracer");
        for (size_t i = 0; i < actual.size(); ++i) check(actual[i] == i, "rank-local values preserved");
        check(nc_inq_varid(ncid, "time", &time) == NC_NOERR, "find time");
        const size_t start = 1, count = 1;
        const double model_time = 3600;
        check(nc_put_vara_double(ncid, time, &start, &count, &model_time) == NC_NOERR, "append time record");
        size_t len = 0;
        check(nc_inq_dimlen(ncid, unlimited, &len) == NC_NOERR && len == 2, "time remains appendable");
        check(nc_close(ncid) == NC_NOERR, "close reader");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::filesystem::remove(path);
    MPI_Finalize();
}
