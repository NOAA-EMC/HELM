#ifndef AMIO_SRC_DRIVERS_NETCDF_NETCDF_STEP_WRITER_HPP
#define AMIO_SRC_DRIVERS_NETCDF_NETCDF_STEP_WRITER_HPP

#include <cstdint>
#include <string>

#include "schema/schema.hpp"
#include "steps/contribution_plan.hpp"

namespace amio::detail {

// The concrete driver has backend-dependent fields. Keep its allocation,
// destruction and member access in the driver translation unit.
struct NetCDFStepWriter;
NetCDFStepWriter *create_netcdf_step_writer();
void destroy_netcdf_step_writer(NetCDFStepWriter *writer) noexcept;
void open_netcdf_step_writer(NetCDFStepWriter &writer, const std::string &path, const Schema &schema, double model_time,
                             std::int64_t fortran_communicator, bool parallel);
void write_netcdf_step_writer(NetCDFStepWriter &writer, const std::string &field, const RegionSelection &region, const void *data);
void close_netcdf_step_writer(NetCDFStepWriter &writer);

}  // namespace amio::detail
#endif
