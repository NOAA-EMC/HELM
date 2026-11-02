#include "drivers/netcdf/netcdf_driver.hpp"
#include "schema/schema.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

#include <mpi.h>
#include <netcdf.h>

using namespace amio::detail;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size == 4);

    const auto path = std::filesystem::temp_directory_path() / "amio-mpi-netcdf-step-payload.nc";
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Schema schema;
    Variable required;
    required.name = "required";
    required.source_dtype = AMIO_DTYPE_F32;
    required.stored_dtype = AMIO_DTYPE_F32;
    required.dimensions.push_back({"x", 6, std::nullopt});
    assert(schema.add_variable(std::move(required)).ok());

    Variable optional;
    optional.name = "optional";
    optional.source_dtype = AMIO_DTYPE_F32;
    optional.stored_dtype = AMIO_DTYPE_F32;
    optional.required = false;
    optional.dimensions.push_back({"x", 6, std::nullopt});
    assert(schema.add_variable(std::move(optional)).ok());
    assert(schema.seal().ok());

    std::map<std::string, NetCDFStepFieldView, std::less<>> fields;
    std::array<float, 2> required_values{};
    if (rank < 3) {
        required_values = {static_cast<float>(rank * 2 + 1), static_cast<float>(rank * 2 + 2)};
        RegionSelection region;
        region.offsets = {static_cast<std::uint64_t>(rank * 2)};
        region.extents = {2};
        fields.emplace("required", NetCDFStepFieldView{region, required_values.data()});
    }

    std::array<float, 2> optional_values{10.0f, 11.0f};
    if (rank == 0) {
        RegionSelection region;
        region.offsets = {0};
        region.extents = {2};
        fields.emplace("optional", NetCDFStepFieldView{region, optional_values.data()});
    }

    const auto communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    write_netcdf_step_payload(path.string(), schema, 42.5, fields, communicator);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        int ncid = -1;
        assert(nc_open(path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR);
        int varid = -1;
        assert(nc_inq_varid(ncid, "required", &varid) == NC_NOERR);
        std::array<float, 6> required_read{};
        assert(nc_get_var_float(ncid, varid, required_read.data()) == NC_NOERR);
        assert((required_read == std::array<float, 6>{1, 2, 3, 4, 5, 6}));

        assert(nc_inq_varid(ncid, "optional", &varid) == NC_NOERR);
        std::array<float, 6> optional_read{};
        assert(nc_get_var_float(ncid, varid, optional_read.data()) == NC_NOERR);
        assert(optional_read[0] == 10.0f && optional_read[1] == 11.0f);
        for (std::size_t i = 2; i < optional_read.size(); ++i) assert(optional_read[i] == NC_FILL_FLOAT);
        assert(nc_close(ncid) == NC_NOERR);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto chunk_path = std::filesystem::temp_directory_path() / "amio-mpi-netcdf-independent-chunks.nc";
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove(chunk_path, ec);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
        NetCDF_Driver driver;
        driver.set_communicator(MPI_COMM_WORLD);
        driver.open_step_write(chunk_path.string(), schema, 43.5);
        driver.prepare_step_independent_writes("required");
        if (rank < 3) {
            RegionSelection region;
            region.offsets = {static_cast<std::uint64_t>(rank * 2)};
            region.extents = {rank == 1 ? 2U : 1U};
            driver.write_step_region_independent("required", region, required_values.data());
            if (rank != 1) {
                ++region.offsets[0];
                driver.write_step_region_independent("required", region, required_values.data() + 1);
            }
        }
        driver.close_step_write();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        int ncid = -1;
        int varid = -1;
        std::array<float, 6> readback{};
        assert(nc_open(chunk_path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR);
        assert(nc_inq_varid(ncid, "required", &varid) == NC_NOERR);
        assert(nc_get_var_float(ncid, varid, readback.data()) == NC_NOERR);
        assert((readback == std::array<float, 6>{1, 2, 3, 4, 5, 6}));
        assert(nc_close(ncid) == NC_NOERR);
        std::error_code ec;
        std::filesystem::remove(chunk_path, ec);
    }

    MPI_Finalize();
    return 0;
}
