// backend_driver.hpp -- AMIO Backend_Driver abstract base class.
//
// This header is PRIVATE to the AMIO_Core build (`src/factory/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Backend_Driver is the polymorphic interface that all concrete
// backend drivers (NetCDF_Driver, Zarr_Driver, GRIB2_Driver)
// implement.  It declares pure virtual methods for opening,
// writing, reading, flushing, and closing datasets.  The
// Backend_Factory (AMIO's own string-keyed singleton registry)
// dispatches to concrete implementations based on a string key
// registered at static initialization time.
//
// The class operates on:
//   * conf::Config for dataset/variable configuration
//   * StagingBuffer references for data payloads
//   * VarMeta descriptors for variable metadata (dtype, shape, name)
//   * BoundingBox descriptors for selective reads
//
// No concrete driver class, header, or symbol is part of the public
// API.  The factory key is the only externally visible coupling
// (R4.8).
//
// Thread safety
// -------------
// Concrete drivers are invoked exclusively from Worker_Pool threads.
// A single driver instance is NOT required to be thread-safe; the
// Worker_Pool's per-(dataset, variable) ordering mutex serializes
// calls to the same driver instance (R6.3).
//
// Validates: R4.2, R4.8

#ifndef AMIO_SRC_FACTORY_BACKEND_DRIVER_HPP
#define AMIO_SRC_FACTORY_BACKEND_DRIVER_HPP

#include <conf/config.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

#include "amio/amio_types.h"

namespace amio::detail {

// Forward-declare StagingBuffer (defined in staging/staging_pool.hpp).
struct StagingBuffer;

// ---------------------------------------------------------------
// Supporting types private to AMIO_Core
// ---------------------------------------------------------------

// DatasetId -- opaque identifier for an open dataset within
// AMIO_Core.  Assigned by the C-Boundary handle table on
// dataset open.
using DatasetId = std::uint64_t;

// VariableId -- opaque identifier for a variable within a dataset.
// Used for per-(dataset, variable) ordering in the Worker_Pool.
using VariableId = std::uint64_t;

// VarMeta -- variable metadata descriptor passed to Backend_Driver
// write and read methods.  Carries the information a driver needs
// to locate and encode/decode a variable's payload within a dataset.
//
// Fields:
//   dataset_id   - identifies the owning dataset
//   variable_id  - identifies the variable within the dataset
//   name         - human-readable variable name (used by GRIB2 for
//                  WMO code table lookup, by NetCDF/Zarr for the
//                  variable path within the file/store)
//   dtype        - element type of the payload
//   shape        - N-dimensional shape descriptor (rank, extents,
//                  strides)
//   timestep     - the temporal index for this write/read operation
//                  (-1 if not applicable)
struct VarMeta {
    DatasetId dataset_id = 0;
    VariableId variable_id = 0;
    std::string name;
    amio_dtype_t dtype = AMIO_DTYPE_F32;
    amio_shape_t shape = {};
    std::int64_t timestep = -1;
};

// BoundingBox -- multidimensional sub-region descriptor for
// selective reads (R5.7).  When passed to Backend_Driver::read,
// the driver requests only the byte ranges that intersect the
// specified region, avoiding transfer of data outside the box.
//
// Fields:
//   rank     - number of valid dimensions (must match the
//              variable's rank)
//   offsets  - per-dimension start offset (0-based, inclusive)
//   extents  - per-dimension extent (number of elements)
//   strides  - per-dimension stride (1 = contiguous along that dim)
//
// Invariants:
//   * rank in [1, AMIO_MAX_RANK]
//   * For each dim d in [0, rank):
//       offsets[d] >= 0
//       extents[d] >= 1
//       strides[d] >= 1
//       offsets[d] + (extents[d] - 1) * strides[d] < variable_extent[d]
struct BoundingBox {
    std::int32_t rank = 0;
    std::int64_t offsets[AMIO_MAX_RANK] = {};
    std::int64_t extents[AMIO_MAX_RANK] = {};
    std::int64_t strides[AMIO_MAX_RANK] = {};
};

// element_size -- byte width of a single element of the given dtype.
//
// Shared helper for the read path and the concrete drivers so the
// dtype-size mapping lives in one place (Req 4.3).  The total payload
// byte count for a variable is:
//     element_size(dtype) * product(shape.extents[0..rank-1])
//
// Returns 0 for an unrecognized dtype tag; callers treat a 0-width
// element as an error sentinel (an unknown/unsupported element type)
// rather than a valid zero-byte payload.
inline std::size_t element_size(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return 4;
        case AMIO_DTYPE_F64:
            return 8;
        case AMIO_DTYPE_I8:
            return 1;
        case AMIO_DTYPE_I16:
            return 2;
        case AMIO_DTYPE_I32:
            return 4;
        case AMIO_DTYPE_I64:
            return 8;
        case AMIO_DTYPE_U8:
            return 1;
        case AMIO_DTYPE_U16:
            return 2;
        case AMIO_DTYPE_U32:
            return 4;
        case AMIO_DTYPE_U64:
            return 8;
        default:
            return 0;
    }
}

// VariableInfo -- driver-reported metadata describing a variable's
// element type, shape, and timestep count (Dataset_Metadata).
//
// Returned by Backend_Driver::describe_variable.  The read path caches
// this per variable and uses it to size staging buffers, populate
// VarMeta dtype/shape for every fetch, set the Prefetch_Queue's total
// timestep count, and validate bounding boxes (Req 4.1, 4.2, 4.5).
//
// Fields:
//   found            - true if the driver located and introspected the
//                      variable; false means the variable is absent or
//                      the driver cannot describe it (read fails with
//                      AMIO_ERR_BACKEND_FAILURE)
//   dtype            - element type of the variable's payload
//   shape            - N-dimensional shape (rank + extents); strides
//                      are contiguous/row-major (0)
//   total_timesteps  - number of temporal indices available for the
//                      variable (1 if the variable is not time-varying)
struct VariableInfo {
    bool found = false;
    amio_dtype_t dtype = AMIO_DTYPE_F32;
    amio_shape_t shape = {};
    std::int64_t total_timesteps = 1;
};

// ---------------------------------------------------------------
// Backend_Driver -- abstract base class for all backend drivers
// ---------------------------------------------------------------

// Backend_Driver is the pure virtual interface that concrete drivers
// implement.  It is registered with AMIO's Backend_Factory (a
// string-keyed singleton registry) keyed by a string (e.g.,
// "netcdf4", "zarr3", "grib2").  New drivers register at static
// initialization via BackendRegistrar<ConcreteDriver>("key")
// without any modification to the public API (R4.2).
//
// Lifecycle:
//   1. Factory instantiates the driver (default constructor).
//   2. open_write() or open_read() is called with the dataset
//      configuration.
//   3. write() / read() are called zero or more times.
//   4. flush() is called to ensure all pending operations are
//      durable.
//   5. close() is called to release resources.
//
// Exception contract:
//   All methods may throw std::exception subclasses (including
//   conf::Conf_Error for configuration problems) on failure.
//   The Worker_Pool exception cordon catches these and translates
//   them to AMIO_ERR_* codes recorded against the originating
//   handle (R12.1, R12.2).
class Backend_Driver {
   public:
    virtual ~Backend_Driver() = default;

    // open_write -- prepare the driver for write operations.
    //
    // The configuration contains dataset-level parameters such as
    // file path / URI, data model, compression codec, chunk/shard
    // shapes, and any driver-specific options.
    //
    // Throws std::exception (or conf::Conf_Error) on failure
    // (e.g., missing Parallel HDF5 support, invalid data model,
    // missing required fields).
    virtual void open_write(const conf::Config &config) = 0;

    // open_read -- prepare the driver for read operations.
    //
    // The configuration contains dataset-level parameters such as
    // file path / URI and any driver-specific options.
    //
    // Throws std::exception (or conf::Conf_Error) on failure.
    virtual void open_read(const conf::Config &config) = 0;

    // write -- serialize a staging buffer payload to storage.
    //
    // Parameters:
    //   src  - the StagingBuffer containing the payload bytes
    //          (src.data[0..src.used_bytes) is the payload)
    //   meta - variable metadata (name, dtype, shape, timestep)
    //
    // The driver encodes the payload according to its format and
    // writes it to the storage layer.  On success, the data is
    // durable (or at least committed to the driver's internal
    // write pipeline).  On failure, throws std::exception.
    //
    // The caller (Worker_Pool) holds the per-(dataset, variable)
    // ordering mutex during this call, ensuring writes to the same
    // (dataset, variable) pair execute in submission order (R6.3).
    virtual void write(const StagingBuffer &src, const VarMeta &meta) = 0;

    // read -- deserialize a payload from storage into a staging buffer.
    //
    // Parameters:
    //   dst  - the StagingBuffer to populate (dst.data has capacity
    //          dst.capacity_bytes; the driver sets dst.used_bytes)
    //   meta - variable metadata identifying what to read
    //   timestep - the temporal index to read
    //   bbox - optional bounding box for selective reads; when
    //          present, the driver requests only intersecting byte
    //          ranges from storage (R5.7)
    //
    // On success, dst.data[0..dst.used_bytes) contains the decoded
    // payload.  On failure, throws std::exception.
    virtual void read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) = 0;

    // get_text_attribute -- read a CF text attribute (e.g. "units",
    // "calendar") for a variable, or a global attribute when var_name is
    // empty.  Returns nullopt when the attribute is absent, the driver is
    // not open for reading, or the backend cannot provide attributes.
    // Default: unsupported (nullopt) so non-netCDF drivers need not
    // implement it.
    virtual std::optional<std::string> get_text_attribute([[maybe_unused]] const std::string &var_name,
                                                          [[maybe_unused]] const std::string &attr_name) {
        return std::nullopt;
    }

    // get_numeric_attribute -- read a numeric attribute (e.g. CF packing
    // attributes "scale_factor"/"add_offset") for a variable, or a global
    // attribute when var_name is empty.  The value is converted to double
    // whatever its on-disk type; only the first element of a multi-valued
    // attribute is returned.  Returns nullopt when the attribute is absent,
    // non-numeric, the driver is not open for reading, or the backend cannot
    // provide attributes.  Default: unsupported (nullopt).
    virtual std::optional<double> get_numeric_attribute([[maybe_unused]] const std::string &var_name, [[maybe_unused]] const std::string &attr_name) {
        return std::nullopt;
    }

    // flush -- ensure all previously written data is durable.
    //
    // Blocks until all pending internal write operations (if any)
    // have completed.  Throws std::exception on failure.
    virtual void flush() = 0;

    // close -- release all resources held by the driver.
    //
    // After close(), no further write/read/flush calls are valid.
    // Throws std::exception if outstanding operations cannot be
    // completed.
    virtual void close() = 0;

    // set_communicator -- provide the MPI communicator handle for
    // parallel I/O operations.
    //
    // Called by the C_Boundary after factory construction but before
    // open_write / open_read.  Drivers that need a raw MPI_Comm for
    // parallel file creation (e.g., nc_create_par, nc_open_par) override
    // this to store the handle.  The default implementation is a no-op
    // so that drivers without MPI needs (Zarr, GRIB2) remain unchanged.
    //
    // The caller (AMIO_Core) guarantees that the owning
    // halo::Communicator in IOCommunicator outlives all driver usages
    // of the handle (the IOCommunicator lives on the Worker_Pool, which
    // is destroyed after all datasets are closed).  (R3.3, R10.4)
    //
    // Parameters:
    //   comm_handle - raw MPI_Comm from IOCommunicator::handle().
    //                 When no communicator split was performed, this is
    //                 MPI_COMM_WORLD.  When MPI is unavailable, this
    //                 method is never called.
#ifdef AMIO_HAS_MPI
    virtual void set_communicator(MPI_Comm comm_handle) {
        (void)comm_handle;
    }
#else
    // Keep vtable layout identical in standalone / non-MPI builds to prevent vtable misalignment and segfaults.
    virtual void set_communicator(int comm_handle) {
        (void)comm_handle;
    }
#endif

    // describe_variable -- report a variable's element type, shape, and
    // timestep count (Dataset_Metadata).
    //
    // The read path calls this once per variable (lazily, on first
    // read) to size staging buffers, populate VarMeta dtype/shape for
    // every fetch, set the Prefetch_Queue's total timestep count, and
    // validate bounding boxes (Req 4.1, 4.2, 4.5).
    //
    // This is intentionally NOT pure virtual: the default implementation
    // returns VariableInfo{found = false} so existing drivers (and test
    // mocks) compile unchanged and opt in only when they support read
    // introspection.  A returned `found == false` causes the read path
    // to fail the read with AMIO_ERR_BACKEND_FAILURE.
    //
    // Unlike the I/O methods, this is invoked from the calling thread
    // during variable resolution rather than from a Worker_Pool thread.
    virtual VariableInfo describe_variable(const std::string &name) {
        (void)name;
        return VariableInfo{};
    }
};

}  // namespace amio::detail

#endif  // AMIO_SRC_FACTORY_BACKEND_DRIVER_HPP
