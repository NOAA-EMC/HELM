#ifndef AMIO_SRC_DRIVERS_NETCDF_NETCDF_STEPS_HPP
#define AMIO_SRC_DRIVERS_NETCDF_NETCDF_STEPS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "schema/schema.hpp"
#include "steps/contribution_plan.hpp"

#ifdef AMIO_HAS_NETCDF
#include <netcdf.h>
#endif

namespace amio::detail {

/** Immutable file location for one logical step. */
struct NetCDFStepLayout {
    std::string payload_path;
};

/** Construct collection/payloads/<canonical step id>.nc. */
[[nodiscard]] bool make_netcdf_step_layout(const std::string &collection_root, std::uint64_t step_id, NetCDFStepLayout &layout, std::string &error);

/** Validate native NetCDF naming/dimension constraints and checked extents. */
[[nodiscard]] bool validate_netcdf_step_schema(const Schema &schema, std::string &error);

/** Validate a region selection against a variable's global dimensions. Empty selections are valid. */
[[nodiscard]] bool validate_netcdf_region(const RegionSelection &region, const std::vector<std::uint64_t> &global_extents, std::string &error);

#ifdef AMIO_HAS_NETCDF
/** Define a complete native schema in an already-created NetCDF file. */
[[nodiscard]] bool define_netcdf_step_schema(int ncid, const Schema &schema, double model_time, std::string &error);
/** Write a hyperslab collectively, or independently after explicit preparation. */
[[nodiscard]] bool write_netcdf_step_region(int ncid, const std::string &variable, const RegionSelection &region, const void *data,
                                            std::string &error, bool independent = false);
/** Configure a parallel variable for independent writes; every writer follows the same field order. */
[[nodiscard]] bool prepare_netcdf_independent_region_writes(int ncid, const std::string &variable, std::string &error, bool parallel);
/** Open a closed step file independently, read one variable, and close it. */
[[nodiscard]] bool read_netcdf_step_region(const std::string &payload_path, const std::string &variable, const RegionSelection &region, void *data,
                                           std::string &error);
#endif

}  // namespace amio::detail

#endif
