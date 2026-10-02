// netcdf_driver.hpp -- AMIO NetCDF_Driver (Parallel HDF5 + MPI-IO).
//
// This header is PRIVATE to the AMIO_Core build (`src/drivers/netcdf/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// NetCDF_Driver is the concrete Backend_Driver implementation for
// NetCDF-4 output via netCDF-cxx4 over Parallel HDF5 with MPI-IO
// collective operations.  It registers with the BackendFactory under
// the key "netcdf4" at static initialization time.
//
// On construction, the driver verifies that the linked netCDF-cxx4
// library was built with Parallel HDF5 + MPI-IO support.  If that
// capability is absent, construction raises std::runtime_error (R7.1).
//
// Data model support:
//   - NetCDF-4 Classic (default)  -- NC_CLASSIC_MODEL | NC_NETCDF4
//   - NetCDF-4 Enhanced           -- NC_NETCDF4 (no classic flag)
//   - Any other data model raises std::runtime_error (R7.2, R7.3).
//
// Compression:
//   - Only lossless filters/compression from the manifest allow-list
//     are applied (R7.5).  Lossy codecs are rejected.
//
// Parallel access:
//   - Files are opened in parallel mode with MPI-IO collective
//     operations for multi-rank writes (R7.4).
//
// Error handling:
//   - All NetCDF/HDF5 errors are wrapped in std::runtime_error with
//     the underlying error codes (R7.6).
//
// Thread safety
// -------------
// A single NetCDF_Driver instance is NOT thread-safe.  The
// Worker_Pool's per-(dataset, variable) ordering mutex serializes
// calls to the same driver instance (R6.3).
//
// Validates: R7.1, R7.2, R7.3, R7.4, R7.5, R7.6

#ifndef AMIO_SRC_DRIVERS_NETCDF_NETCDF_DRIVER_HPP
#define AMIO_SRC_DRIVERS_NETCDF_NETCDF_DRIVER_HPP

#include "factory/backend_driver.hpp"

#ifdef AMIO_HAS_NETCDF
#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>
#endif

#include <string>
#include <vector>

#include "drivers/common/var_attributes.hpp"

namespace amio::detail {

// DataModel -- supported NetCDF-4 data models.
enum class NetCDF4DataModel {
    Classic,  // NC_CLASSIC_MODEL | NC_NETCDF4
    Enhanced  // NC_NETCDF4 (full model)
};

// NetCDF_Driver -- concrete Backend_Driver for NetCDF-4 via
// Parallel HDF5 + MPI-IO.
//
// Factory key: "netcdf4"
// Registration: BackendRegistrar<NetCDF_Driver>("netcdf4")
class NetCDF_Driver : public Backend_Driver {
   public:
    NetCDF_Driver();
    ~NetCDF_Driver() override;

    // Backend_Driver interface.
    void open_write(const conf::Config &config) override;
    void open_read(const conf::Config &config) override;
    void write(const StagingBuffer &src, const VarMeta &meta) override;
    void read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) override;
    void flush() override;
    void close() override;

#if defined(AMIO_HAS_MPI) && defined(AMIO_HAS_NETCDF)
    // Override set_communicator to receive the I/O communicator
    // handle from the Worker_Pool's IOCommunicator (R3.3, R10.4).
    // Must be called before open_write / open_read.
    void set_communicator(MPI_Comm comm_handle) override;
#endif

    // Report a variable's element type, shape, and timestep count by
    // introspecting the open NetCDF file (nc_inq_varid / nc_inq_var /
    // nc_inq_dimlen + the unlimited/record dimension).  Returns
    // VariableInfo{found = false} when the variable is absent, the
    // driver is not open for reading, or the netCDF element type has no
    // AMIO dtype mapping.  (Req 4.1, 4.2, 4.5, 9.1)
    VariableInfo describe_variable(const std::string &name) override;

    // Read a CF text attribute ("units", "calendar", ...) for a variable, or
    // a global attribute when var_name is empty.  Handles both NC_CHAR and
    // NC_STRING storage; for a multi-element NC_STRING attribute only the first
    // string is returned.  Returns nullopt if absent, numeric, or the driver is
    // not open for reading.  (CF time-axis decoding support.)
    std::optional<std::string> get_text_attribute(const std::string &var_name, const std::string &attr_name) override;

    // Read a numeric attribute ("scale_factor", "add_offset", ...) for a
    // variable, or a global attribute when var_name is empty.  Returns
    // nullopt if absent, non-numeric, or the driver is not open for reading.
    std::optional<double> get_numeric_attribute(const std::string &var_name, const std::string &attr_name) override;

   public:
    // ----- Static utility methods (public for testability) -----

    // Validate the data model string from configuration.
    static NetCDF4DataModel parse_data_model(const std::string &model_str);

   private:
    // Verify that the linked netCDF library has Parallel HDF5 + MPI-IO.
    static void verify_parallel_support();

    // Validate that a codec is on the lossless allow-list.
    static void validate_codec(const std::string &codec, const std::vector<std::string> &allow_list);

    // Map amio_dtype_t to netCDF type constant.
    static int dtype_to_nc_type(amio_dtype_t dtype);

    // Get the byte size of an amio_dtype_t element.
    static std::size_t dtype_byte_size(amio_dtype_t dtype);

#ifdef AMIO_HAS_NETCDF
    // Internal state.
    int ncid_ = -1;                  // NetCDF file ID
    MPI_Comm comm_ = MPI_COMM_NULL;  // MPI communicator for parallel I/O
    MPI_Info info_ = MPI_INFO_NULL;  // MPI info object
#endif

    NetCDF4DataModel data_model_ = NetCDF4DataModel::Classic;
    std::string file_path_;
    std::string active_codec_;
    std::vector<std::string> codec_allow_list_;
    bool is_open_ = false;
    bool is_write_mode_ = false;
    bool use_parallel_ = false;

    // CF/UGRID convention metadata + per-variable attributes parsed
    // from the dataset manifest (R7.x convention compliance).
    DatasetAttributes attributes_;
    bool global_attrs_written_ = false;
};

}  // namespace amio::detail

#endif  // AMIO_SRC_DRIVERS_NETCDF_NETCDF_DRIVER_HPP
