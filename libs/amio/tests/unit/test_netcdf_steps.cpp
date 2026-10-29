#include "drivers/netcdf/netcdf_steps.hpp"
#include "schema/schema.hpp"

#include <cassert>
#include <filesystem>
#include <string>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif
#ifdef AMIO_HAS_PARALLEL_NETCDF
#include <netcdf_par.h>
#endif

using namespace amio::detail;

int main(int argc, char **argv) {
#ifdef AMIO_HAS_MPI
    MPI_Init(&argc, &argv);
#endif
    const auto path = std::filesystem::temp_directory_path() / "amio-netcdf-step-test.nc";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    Schema schema;
    Variable temperature;
    temperature.name = "temperature";
    temperature.source_dtype = AMIO_DTYPE_F32;
    temperature.stored_dtype = AMIO_DTYPE_F32;
    temperature.dimensions.push_back({"x", 4, std::nullopt});
    temperature.attributes.emplace("units", Attribute{std::string("K")});
    assert(schema.add_variable(std::move(temperature)).ok());
    assert(schema.seal().ok());

    int ncid = -1;
#ifdef AMIO_HAS_PARALLEL_NETCDF
    assert(nc_create_par(path.c_str(), NC_NETCDF4 | NC_CLOBBER, MPI_COMM_SELF, MPI_INFO_NULL, &ncid) == NC_NOERR);
#else
    assert(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid) == NC_NOERR);
#endif
    std::string error;
    assert(define_netcdf_step_schema(ncid, schema, 12.5, error));
    RegionSelection region;
    region.offsets = {1};
    region.extents = {2};
    const float values[] = {11.0f, 12.0f};
    assert(write_netcdf_step_region(ncid, "temperature", region, values, error));

    RegionSelection empty;
    empty.offsets = {4};
    empty.extents = {0};
    empty.empty = true;
    assert(write_netcdf_step_region(ncid, "temperature", empty, nullptr, error));
    assert(nc_close(ncid) == NC_NOERR);

    float readback[2] = {};
    assert(read_netcdf_step_region(path.string(), "temperature", region, readback, error));
    assert(readback[0] == values[0] && readback[1] == values[1]);

    // Independent native NetCDF-C reader check, separate from the AMIO step
    // read adapter above.
    int reader = -1;
    assert(nc_open(path.c_str(), NC_NOWRITE, &reader) == NC_NOERR);
    int variable_id = -1;
    assert(nc_inq_varid(reader, "temperature", &variable_id) == NC_NOERR);
    const std::size_t start[] = {1};
    const std::size_t count[] = {2};
    float native_values[2] = {};
    assert(nc_get_vara_float(reader, variable_id, start, count, native_values) == NC_NOERR);
    assert(native_values[0] == values[0] && native_values[1] == values[1]);
    char units[2] = {};
    assert(nc_get_att_text(reader, variable_id, "units", units) == NC_NOERR);
    assert(units[0] == 'K');
    assert(nc_close(reader) == NC_NOERR);
    std::filesystem::remove(path, ec);
#ifdef AMIO_HAS_MPI
    MPI_Finalize();
#endif
}
