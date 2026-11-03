#include "native_readers.hpp"

#include <netcdf.h>

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "reference_fields.hpp"

namespace amio::qualification {

std::string field_name(std::uint64_t field) {
    std::ostringstream text;
    text << "field_" << std::setw(2) << std::setfill('0') << field;
    return text.str();
}

std::string step_payload_path(const std::filesystem::path &root, std::uint64_t step) {
    std::ostringstream text;
    text << "payloads/steps/" << std::setw(20) << std::setfill('0') << step << ".nc";
    return (root / text.str()).string();
}

void verify_native_netcdf_field(const std::filesystem::path &root, std::uint64_t step, std::uint64_t cells, std::uint64_t field,
                                double expected_model_time, const std::string &expected_schema_digest) {
    const auto path = step_payload_path(root, step);
    int ncid = -1;
    if (nc_open(path.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) throw std::runtime_error("native NetCDF reader cannot open step payload");
    double model_time = 0.0;
    if (nc_get_att_double(ncid, NC_GLOBAL, "amio_model_time", &model_time) != NC_NOERR || model_time != expected_model_time) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF model-time metadata mismatch");
    }
    std::size_t digest_bytes = 0;
    if (nc_inq_attlen(ncid, NC_GLOBAL, "amio_schema_digest", &digest_bytes) != NC_NOERR || digest_bytes != expected_schema_digest.size()) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF schema-digest metadata size mismatch");
    }
    std::string schema_digest(digest_bytes, '\0');
    if (nc_get_att_text(ncid, NC_GLOBAL, "amio_schema_digest", schema_digest.data()) != NC_NOERR || schema_digest != expected_schema_digest) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF schema-digest metadata mismatch");
    }
    const auto name = field_name(field);
    int varid = -1;
    if (nc_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF reader cannot find field");
    }
    std::vector<float> values(static_cast<std::size_t>(cells));
    if (nc_get_var_float(ncid, varid, values.data()) != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF full-field read failed");
    }
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        if (values[static_cast<std::size_t>(cell)] != reference_value(step, field, cell)) {
            nc_close(ncid);
            throw std::runtime_error("native NetCDF data mismatch at step/field/cell");
        }
    }
    // The native reader's partition starts at a different global offset than
    // the full-field writer layout.
    constexpr std::size_t offset = 7;
    const std::size_t count = static_cast<std::size_t>(cells) - offset;
    std::vector<float> selection(count);
    const std::size_t start[] = {offset};
    const std::size_t extent[] = {count};
    if (nc_get_vara_float(ncid, varid, start, extent, selection.data()) != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("native NetCDF different-partition read failed");
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (selection[i] != reference_value(step, field, offset + i)) {
            nc_close(ncid);
            throw std::runtime_error("native NetCDF selected data mismatch");
        }
    }
    if (nc_close(ncid) != NC_NOERR) throw std::runtime_error("native NetCDF close failed");
}

}  // namespace amio::qualification
