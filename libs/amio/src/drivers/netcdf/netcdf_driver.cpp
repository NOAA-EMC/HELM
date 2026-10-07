// netcdf_driver.cpp -- AMIO NetCDF_Driver implementation.
//
// Concrete Backend_Driver for NetCDF-4 output via netCDF-cxx4 over
// Parallel HDF5 with MPI-IO collective operations.
//
// Compilation:
//   - When AMIO_HAS_NETCDF is defined: full implementation using
//     netCDF-cxx4 API with Parallel HDF5 + MPI-IO.
//   - When AMIO_HAS_NETCDF is NOT defined: the driver still registers
//     with the factory but raises std::runtime_error on any operation
//     (graceful degradation for builds without netCDF).
//
// Validates: R7.1, R7.2, R7.3, R7.4, R7.5, R7.6

#include "drivers/netcdf/netcdf_driver.hpp"

#include <conf/config.hpp>
#include <conf/error.hpp>

#include "drivers/common/var_attributes.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

#ifdef AMIO_HAS_NETCDF
#include <mpi.h>
#include <netcdf.h>
#include <netcdf_meta.h>
#include <netcdf_par.h>

extern MPI_Comm g_amio_parent_comm;
#endif

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace amio::detail {

static std::mutex g_nc_driver_mutex;

// ===================================================================
// Helper: wrap NetCDF error codes in exceptions.
// ===================================================================

#ifdef AMIO_HAS_NETCDF

// Check a netCDF return code and throw std::runtime_error on failure.
static void nc_check(int status, const std::string &context) {
    if (status != NC_NOERR) {
        std::cerr << "[AMIO ERROR] in " << context << ": " << nc_strerror(status) << " (nc_errno=" << status << ")" << std::endl;
        std::string msg = "NetCDF error in " + context + ": " + nc_strerror(status) + " (nc_errno=" + std::to_string(status) + ")";
        throw std::runtime_error(msg);
    }
}

// Write one attribute set onto `varid` (use NC_GLOBAL for file-level
// attributes).  Must be called in define mode.  Numeric attributes are
// emitted with a type matching their literal form (integer literals ->
// NC_INT64 / NC_DOUBLE for reals); everything else is written as text.
// `_FillValue` is special-cased to the variable's own type so it
// round-trips correctly per CF.
static void nc_write_attributes(int ncid, int varid, const VarAttributes &attrs, int var_nc_type) {
    for (const auto &kv : attrs.items) {
        const std::string &name = kv.first;
        const AttrValue &val = kv.second;

        if (val.is_numeric) {
            if (name == "_FillValue" && varid != NC_GLOBAL && var_nc_type != NC_NAT) {
                // Emit _FillValue in the variable's own type (CF rule).
                switch (var_nc_type) {
                    case NC_FLOAT: {
                        float f = static_cast<float>(val.number);
                        nc_check(nc_put_att_float(ncid, varid, name.c_str(), NC_FLOAT, 1, &f), "nc_put_att_float('" + name + "')");
                        continue;
                    }
                    case NC_DOUBLE: {
                        double d = val.number;
                        nc_check(nc_put_att_double(ncid, varid, name.c_str(), NC_DOUBLE, 1, &d), "nc_put_att_double('" + name + "')");
                        continue;
                    }
                    case NC_INT: {
                        int i = static_cast<int>(val.number);
                        nc_check(nc_put_att_int(ncid, varid, name.c_str(), NC_INT, 1, &i), "nc_put_att_int('" + name + "')");
                        continue;
                    }
                    case NC_INT64: {
                        long long ll = static_cast<long long>(val.number);
                        nc_check(nc_put_att_longlong(ncid, varid, name.c_str(), NC_INT64, 1, &ll), "nc_put_att_longlong('" + name + "')");
                        continue;
                    }
                    default:
                        break;  // fall through to generic numeric handling
                }
            }

            if (val.is_integer) {
                long long ll = static_cast<long long>(val.number);
                nc_check(nc_put_att_longlong(ncid, varid, name.c_str(), NC_INT64, 1, &ll), "nc_put_att_longlong('" + name + "')");
            } else {
                double d = val.number;
                nc_check(nc_put_att_double(ncid, varid, name.c_str(), NC_DOUBLE, 1, &d), "nc_put_att_double('" + name + "')");
            }
        } else {
            nc_check(nc_put_att_text(ncid, varid, name.c_str(), val.text.size(), val.text.c_str()), "nc_put_att_text('" + name + "')");
        }
    }
}

#endif  // AMIO_HAS_NETCDF

// ===================================================================
// NetCDF_Driver -- construction and destruction.
// ===================================================================

NetCDF_Driver::NetCDF_Driver() {
    // On construction, verify that the linked netCDF library has
    // Parallel HDF5 + MPI-IO support (R7.1).
    verify_parallel_support();
}

NetCDF_Driver::~NetCDF_Driver() {
    // Best-effort close if still open.
    if (is_open_) {
        try {
            close();
        } catch (...) {
            // Suppress exceptions in destructor.
        }
    }
}

// ===================================================================
// verify_parallel_support -- check for Parallel HDF5 + MPI-IO (R7.1).
// ===================================================================

void NetCDF_Driver::verify_parallel_support() {
#ifdef AMIO_HAS_NETCDF
    // Query the netcdf-c library for parallel I/O support.
    // NC_HAS_PARALLEL is defined in netcdf_meta.h when the library
    // was built with --enable-parallel4 (Parallel HDF5 + MPI-IO).
#if defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
    // Parallel HDF5 + MPI-IO is available.
    return;
#else
    throw std::runtime_error(
        "NetCDF_Driver: linked netCDF-cxx4 library does not have "
        "Parallel HDF5 + MPI-IO support. Rebuild netCDF with "
        "--enable-parallel4 (R7.1)");
#endif
#else
    throw std::runtime_error(
        "NetCDF_Driver: AMIO was built without netCDF support "
        "(AMIO_HAS_NETCDF not defined). Cannot use netcdf4 backend.");
#endif
}

// ===================================================================
// parse_data_model -- validate data model string (R7.2, R7.3).
// ===================================================================

NetCDF4DataModel NetCDF_Driver::parse_data_model(const std::string &model_str) {
    if (model_str.empty() || model_str == "classic" || model_str == "netcdf4_classic" || model_str == "NC4_CLASSIC") {
        return NetCDF4DataModel::Classic;
    }
    if (model_str == "enhanced" || model_str == "netcdf4_enhanced" || model_str == "NC4") {
        return NetCDF4DataModel::Enhanced;
    }
    throw std::runtime_error("NetCDF_Driver: invalid data model '" + model_str +
                             "'. "
                             "Supported values: 'classic' (default), 'enhanced' (R7.2, R7.3)");
}

// ===================================================================
// validate_codec -- ensure codec is lossless and on allow-list (R7.5).
// ===================================================================

void NetCDF_Driver::validate_codec(const std::string &codec, const std::vector<std::string> &allow_list) {
    if (codec.empty()) {
        // No compression requested -- valid.
        return;
    }

    // Check against the allow-list.
    auto it = std::find(allow_list.begin(), allow_list.end(), codec);
    if (it == allow_list.end()) {
        throw std::runtime_error("NetCDF_Driver: codec '" + codec +
                                 "' is not on the "
                                 "lossless compression allow-list. Only lossless filters "
                                 "are permitted for NetCDF-4 output (R7.5)");
    }

    // Additional check: reject known lossy codecs even if somehow
    // they appear on the allow-list (defense in depth).
    static const std::vector<std::string> lossy_codecs = {"lossy", "sz", "zfp_lossy", "fpzip_lossy"};
    for (const auto &lossy : lossy_codecs) {
        if (codec == lossy) {
            throw std::runtime_error("NetCDF_Driver: codec '" + codec +
                                     "' is a lossy codec "
                                     "and cannot be used with NetCDF-4 output (R7.5)");
        }
    }
}

// ===================================================================
// dtype_to_nc_type -- map AMIO dtype to netCDF type constant.
// ===================================================================

int NetCDF_Driver::dtype_to_nc_type(amio_dtype_t dtype) {
#ifdef AMIO_HAS_NETCDF
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return NC_FLOAT;
        case AMIO_DTYPE_F64:
            return NC_DOUBLE;
        case AMIO_DTYPE_I8:
            return NC_BYTE;
        case AMIO_DTYPE_I16:
            return NC_SHORT;
        case AMIO_DTYPE_I32:
            return NC_INT;
        case AMIO_DTYPE_I64:
            return NC_INT64;
        case AMIO_DTYPE_U8:
            return NC_UBYTE;
        case AMIO_DTYPE_U16:
            return NC_USHORT;
        case AMIO_DTYPE_U32:
            return NC_UINT;
        case AMIO_DTYPE_U64:
            return NC_UINT64;
        default:
            throw std::runtime_error("NetCDF_Driver: unsupported dtype " + std::to_string(static_cast<int>(dtype)));
    }
#else
    (void)dtype;
    throw std::runtime_error("NetCDF_Driver: built without netCDF support");
#endif
}

// ===================================================================
// dtype_byte_size -- get element size for an AMIO dtype.
// ===================================================================

std::size_t NetCDF_Driver::dtype_byte_size(amio_dtype_t dtype) {
    // Delegate to the shared element_size helper (backend_driver.hpp) so the
    // dtype->byte-width mapping lives in one place.  Preserve the driver's
    // existing contract of throwing on an unrecognized dtype (element_size
    // returns 0 as its unknown-dtype sentinel).
    const std::size_t size = element_size(dtype);
    if (size == 0) {
        throw std::runtime_error("NetCDF_Driver: unsupported dtype " + std::to_string(static_cast<int>(dtype)));
    }
    return size;
}

// ===================================================================
// resolve_dataset_path -- read the dataset file path from the config.
//
// The manifest schema and the example/test dataset configs are not fully
// consistent: some use `path`, others use `output_path` (the GRIB2 driver
// already accepts both).  Accept either here so the NetCDF driver opens
// against the same key the rest of the toolchain emits.  Returns an empty
// string when neither key is present.
// ===================================================================
static std::string resolve_dataset_path(const conf::Config &config) {
    if (config.has("path")) {
        return config.get_string("path");
    }
    if (config.has("output_path")) {
        return config.get_string("output_path");
    }
    return std::string{};
}

#if defined(AMIO_HAS_MPI) && defined(AMIO_HAS_NETCDF)
void NetCDF_Driver::set_communicator(MPI_Comm comm_handle) {
    comm_ = comm_handle;
}
#endif

// ===================================================================
// open_write -- prepare for parallel write operations (R7.1, R7.4).
// ===================================================================

void NetCDF_Driver::open_write(const conf::Config &config) {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (is_open_) {
        throw std::runtime_error("NetCDF_Driver::open_write: driver is already open");
    }

#ifdef AMIO_HAS_NETCDF
    // Extract configuration parameters.
    std::string path = resolve_dataset_path(config);
    if (path.empty()) {
        throw std::runtime_error("NetCDF_Driver::open_write: 'path' (or 'output_path') field is required");
    }
    file_path_ = path;

    // Parse data model (R7.2, R7.3).
    std::string model_str = config.get_or<std::string>("data_model", "classic");
    data_model_ = parse_data_model(model_str);

    // Parse and validate compression codec (R7.5).
    //
    // The manifest nests the codec settings under a `codec:` map
    // (`codec.active_codec` + `codec.lossless_allow_list`), matching the
    // ConfigLoader schema and the example manifests.  Read those nested
    // keys via CONF's dotted-path access.  Fall back to the legacy flat
    // keys (`codec` / `codec_allow_list`) when the nested form is absent
    // so older configs keep working.
    if (config.has("codec.active_codec") || config.has("codec.lossless_allow_list")) {
        active_codec_ = config.get_or<std::string>("codec.active_codec", "");
        codec_allow_list_ =
            config.has("codec.lossless_allow_list") ? config.get_string_list("codec.lossless_allow_list") : std::vector<std::string>{};
    } else {
        codec_allow_list_ = config.has("codec_allow_list") ? config.get_string_list("codec_allow_list") : std::vector<std::string>{};
        active_codec_ = config.get_or<std::string>("codec", "");
    }
    validate_codec(active_codec_, codec_allow_list_);

    // Parse CF/UGRID convention metadata + per-variable attributes
    // from the manifest.  The global Conventions attribute is written
    // below; per-variable attributes are applied at variable-define
    // time in write().
    attributes_ = parse_dataset_attributes(config);

    // Determine the MPI communicator for parallel I/O.
    // Default to g_amio_parent_comm if not specified.
    if (comm_ == MPI_COMM_NULL) {
        comm_ = g_amio_parent_comm;
    }
    info_ = MPI_INFO_NULL;

    // Determine NetCDF creation mode flags.
    int cmode = NC_NETCDF4 | NC_CLOBBER;
    if (data_model_ == NetCDF4DataModel::Classic) {
        cmode |= NC_CLASSIC_MODEL;
    }
    // Enhanced mode: just NC_NETCDF4 (no classic flag).

    // Determine if parallel I/O is required (only if MPI is initialized and rank size > 1)
    use_parallel_ = false;
#ifdef AMIO_HAS_MPI
    if (comm_ != MPI_COMM_NULL) {
        int comm_size = 1;
        MPI_Comm_size(comm_, &comm_size);
        if (comm_size > 1) {
            use_parallel_ = true;
        }
    }
#endif

    // Create the file (R7.4 parallel mode or fallback sequential).
    int status;
    if (use_parallel_) {
        status = nc_create_par(path.c_str(), cmode, comm_, info_, &ncid_);
        nc_check(status, "nc_create_par('" + path + "')");
    } else {
        status = nc_create(path.c_str(), cmode, &ncid_);
        nc_check(status, "nc_create('" + path + "')");
    }

    // Write the global CF/UGRID `Conventions` attribute plus any extra
    // global attributes declared in the manifest.  The file is in
    // define mode immediately after nc_create_par.
    {
        const std::string &conv = attributes_.conventions;
        nc_check(nc_put_att_text(ncid_, NC_GLOBAL, "Conventions", conv.size(), conv.c_str()), "nc_put_att_text(Conventions)");
        nc_write_attributes(ncid_, NC_GLOBAL, attributes_.global, NC_NAT);
        global_attrs_written_ = true;
    }

    is_open_ = true;
    is_write_mode_ = true;

#else
    (void)config;
    throw std::runtime_error("NetCDF_Driver::open_write: AMIO built without netCDF support");
#endif
}

// ===================================================================
// open_read -- prepare for parallel read operations.
// ===================================================================

void NetCDF_Driver::open_read(const conf::Config &config) {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (is_open_) {
        throw std::runtime_error("NetCDF_Driver::open_read: driver is already open");
    }

#ifdef AMIO_HAS_NETCDF
    // Extract configuration parameters.
    std::string path = resolve_dataset_path(config);
    if (path.empty()) {
        throw std::runtime_error("NetCDF_Driver::open_read: 'path' (or 'output_path') field is required");
    }
    file_path_ = path;

    // Parse data model for validation (R7.2, R7.3).
    std::string model_str = config.get_or<std::string>("data_model", "classic");
    data_model_ = parse_data_model(model_str);

    // MPI communicator for parallel reads.
    comm_ = g_amio_parent_comm;
    info_ = MPI_INFO_NULL;

    int comm_size = 1;
    use_parallel_ = false;
#ifdef AMIO_HAS_MPI
    if (comm_ != MPI_COMM_NULL) {
        MPI_Comm_size(comm_, &comm_size);
        if (comm_size > 1) {
            use_parallel_ = true;
        }
    }
#endif

    // Open the file in parallel read mode or fallback sequential.
    int status;
    if (use_parallel_) {
        status = nc_open_par(path.c_str(), NC_NOWRITE, comm_, info_, &ncid_);
        if (status != NC_NOERR) {
            std::cerr << "[AMIO WARN] nc_open_par failed for '" << path << "' with " << nc_strerror(status) << " (nc_errno=" << status
                      << "); retrying serial nc_open." << std::endl;
            use_parallel_ = false;
            status = nc_open(path.c_str(), NC_NOWRITE, &ncid_);
            nc_check(status, "nc_open('" + path + "')");
        }
    } else {
        status = nc_open(path.c_str(), NC_NOWRITE, &ncid_);
        nc_check(status, "nc_open('" + path + "')");
    }

    is_open_ = true;
    is_write_mode_ = false;

#else
    (void)config;
    throw std::runtime_error("NetCDF_Driver::open_read: AMIO built without netCDF support");
#endif
}

// ===================================================================
// write -- serialize StagingBuffer to NetCDF variable (R7.4, R7.5).
// ===================================================================

void NetCDF_Driver::write(const StagingBuffer &src, const VarMeta &meta) {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (!is_open_ || !is_write_mode_) {
        throw std::runtime_error("NetCDF_Driver::write: driver not open for writing");
    }

#ifdef AMIO_HAS_NETCDF
    // Look up or create the variable in the NetCDF file.
    int varid = -1;
    int status = nc_inq_varid(ncid_, meta.name.c_str(), &varid);

    if (status == NC_ENOTVAR) {
        // Variable does not exist yet -- define it.
        // First, define dimensions based on the shape.
        std::vector<int> dimids(meta.shape.rank);

        // Switch to define mode if needed.
        int redef_status = nc_redef(ncid_);
        // NC_EINDEFINE means we're already in define mode -- that's fine.
        if (redef_status != NC_NOERR && redef_status != NC_EINDEFINE) {
            nc_check(redef_status, "nc_redef");
        }

        auto find_matching_dim = [&](const std::string &candidate_name, std::size_t target_len, int &out_dimid) -> bool {
            int dimid = -1;
            if (nc_inq_dimid(ncid_, candidate_name.c_str(), &dimid) == NC_NOERR) {
                size_t dim_len = 0;
                if (nc_inq_dimlen(ncid_, dimid, &dim_len) == NC_NOERR) {
                    if (dim_len == target_len) {
                        out_dimid = dimid;
                        return true;
                    }
                }
            }
            return false;
        };

        for (int32_t d = 0; d < meta.shape.rank; ++d) {
            std::size_t target_len = static_cast<std::size_t>(meta.shape.extents[d]);
            std::string dim_name = meta.name + "_dim" + std::to_string(d);
            if ((meta.name == "lon" || meta.name == "lat" || meta.name == "lev" || meta.name == "time") && meta.shape.rank == 1) {
                dim_name = meta.name;
            }

            int existing_dimid = -1;
            bool found_shared = false;

            if (meta.name != "lon" && meta.name != "lat" && meta.name != "lev" && meta.name != "time") {
                // Find all existing dimensions with matching length.
                int num_dims = 0;
                if (nc_inq_ndims(ncid_, &num_dims) == NC_NOERR) {
                    std::vector<int> matching_dimids;
                    std::vector<std::string> matching_names;
                    for (int i = 0; i < num_dims; ++i) {
                        size_t dim_len = 0;
                        char name_buf[NC_MAX_NAME + 1];
                        if (nc_inq_dim(ncid_, i, name_buf, &dim_len) == NC_NOERR) {
                            if (dim_len == target_len) {
                                matching_dimids.push_back(i);
                                matching_names.push_back(std::string(name_buf));
                            }
                        }
                    }

                    if (matching_dimids.size() == 1) {
                        // Exactly one matching dimension exists -- reuse it!
                        existing_dimid = matching_dimids[0];
                        found_shared = true;
                    } else if (matching_dimids.size() > 1) {
                        // Multiple exist -- choose the best match based on axis type.
                        // Determine standard axis role for dimension d:
                        // d = Rank - 1: X axis (longitude/x)
                        // d = Rank - 2: Y axis (latitude/y)
                        // d = Rank - 3: Z axis (vertical/level)
                        // d = Rank - 4: T axis (time/t)
                        int axis_role = -1;  // 0=X, 1=Y, 2=Z, 3=T
                        if (d == meta.shape.rank - 1)
                            axis_role = 0;
                        else if (d == meta.shape.rank - 2)
                            axis_role = 1;
                        else if (d == meta.shape.rank - 3)
                            axis_role = 2;
                        else if (d == meta.shape.rank - 4)
                            axis_role = 3;

                        int best_score = -1;
                        int best_dimid = -1;
                        for (size_t i = 0; i < matching_dimids.size(); ++i) {
                            const std::string &name = matching_names[i];
                            int score = 0;
                            if (axis_role == 0) {  // X axis
                                if (name == "lon" || name == "x")
                                    score = 3;
                                else if (name.find("lon") != std::string::npos || name.find("x") != std::string::npos)
                                    score = 2;
                                else if (name.find("dim") != std::string::npos)
                                    score = 1;
                            } else if (axis_role == 1) {  // Y axis
                                if (name == "lat" || name == "y")
                                    score = 3;
                                else if (name.find("lat") != std::string::npos || name.find("y") != std::string::npos)
                                    score = 2;
                                else if (name.find("dim") != std::string::npos)
                                    score = 1;
                            } else if (axis_role == 2) {  // Z axis
                                if (name == "lev" || name == "z" || name == "level")
                                    score = 3;
                                else if (name.find("lev") != std::string::npos || name.find("z") != std::string::npos)
                                    score = 2;
                                else if (name.find("dim") != std::string::npos)
                                    score = 1;
                            } else if (axis_role == 3) {  // Time axis
                                if (name == "time" || name == "t")
                                    score = 3;
                                else if (name.find("time") != std::string::npos || name.find("t") != std::string::npos)
                                    score = 2;
                                else if (name.find("dim") != std::string::npos)
                                    score = 1;
                            }

                            if (score > best_score) {
                                best_score = score;
                                best_dimid = matching_dimids[i];
                            }
                        }

                        if (best_dimid != -1) {
                            existing_dimid = best_dimid;
                            found_shared = true;
                        }
                    }
                }
            }

            if (found_shared) {
                dimids[d] = existing_dimid;
            } else {
                // Check if default dimension already exists.
                int dim_status = nc_inq_dimid(ncid_, dim_name.c_str(), &existing_dimid);
                if (dim_status == NC_NOERR) {
                    dimids[d] = existing_dimid;
                } else {
                    // A dimension named "time" is the CF record dimension: define
                    // it as NC_UNLIMITED so downstream tools (ncview, ncrcat, ...)
                    // can concatenate per-timestep files into a single series.
                    const std::size_t def_len = (dim_name == "time") ? static_cast<std::size_t>(NC_UNLIMITED) : target_len;
                    status = nc_def_dim(ncid_, dim_name.c_str(), def_len, &dimids[d]);
                    nc_check(status, "nc_def_dim('" + dim_name + "')");
                }
            }
        }

        // Define the variable.
        int nc_type = dtype_to_nc_type(meta.dtype);
        status = nc_def_var(ncid_, meta.name.c_str(), nc_type, meta.shape.rank, dimids.data(), &varid);
        nc_check(status, "nc_def_var('" + meta.name + "')");

        // Apply CF/UGRID per-variable attributes declared in the
        // manifest (units, standard_name, _FillValue, cf_role, ...).
        // We are still inside define mode here.
        if (const VarAttributes *var_attrs = attributes_.find(meta.name)) {
            nc_write_attributes(ncid_, varid, *var_attrs, nc_type);
        }

        // Apply lossless compression if configured (R7.5).
        if (!active_codec_.empty()) {
            if (active_codec_ == "zstandard" || active_codec_ == "zstd") {
                // Use shuffle + deflate as a portable lossless option.
                // NetCDF-4 supports deflate natively.
                status = nc_def_var_deflate(ncid_, varid,
                                            /*shuffle=*/1,
                                            /*deflate=*/1,
                                            /*deflate_level=*/4);
                nc_check(status, "nc_def_var_deflate('" + meta.name + "')");
            } else if (active_codec_ == "blosc") {
                // Use shuffle + deflate as portable fallback.
                // Blosc filter requires HDF5 plugin; use deflate if
                // the plugin is not available.
                status = nc_def_var_deflate(ncid_, varid,
                                            /*shuffle=*/1,
                                            /*deflate=*/1,
                                            /*deflate_level=*/4);
                nc_check(status, "nc_def_var_deflate('" + meta.name + "')");
            } else {
                // Generic lossless: apply shuffle + deflate.
                status = nc_def_var_deflate(ncid_, varid,
                                            /*shuffle=*/1,
                                            /*deflate=*/1,
                                            /*deflate_level=*/4);
                nc_check(status, "nc_def_var_deflate('" + meta.name + "')");
            }
        }

        // End define mode.
        status = nc_enddef(ncid_);
        nc_check(status, "nc_enddef");

    } else if (status != NC_NOERR) {
        nc_check(status, "nc_inq_varid('" + meta.name + "')");
    }

    // Set collective access mode for parallel MPI-IO writes (R7.4).
    if (use_parallel_) {
        status = nc_var_par_access(ncid_, varid, NC_COLLECTIVE);
        nc_check(status, "nc_var_par_access('" + meta.name + "', NC_COLLECTIVE)");
    }

    // Compute start/count arrays for the write.
    std::vector<std::size_t> start(meta.shape.rank, 0);
    std::vector<std::size_t> count(meta.shape.rank);
    for (int32_t d = 0; d < meta.shape.rank; ++d) {
        count[d] = static_cast<std::size_t>(meta.shape.extents[d]);
    }

    // Write the data from the staging buffer using collective I/O.
    status = nc_put_vara(ncid_, varid, start.data(), count.data(), src.data);
    nc_check(status, "nc_put_vara('" + meta.name + "')");

#else
    (void)src;
    (void)meta;
    throw std::runtime_error("NetCDF_Driver::write: AMIO built without netCDF support");
#endif
}

// ===================================================================
// read -- read variable data into StagingBuffer.
// ===================================================================

void NetCDF_Driver::read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (!is_open_ || is_write_mode_) {
        throw std::runtime_error("NetCDF_Driver::read: driver not open for reading");
    }

#ifdef AMIO_HAS_NETCDF
    // Look up the variable.
    int varid = -1;
    int status = nc_inq_varid(ncid_, meta.name.c_str(), &varid);
    nc_check(status, "nc_inq_varid('" + meta.name + "') for read");

    // Set collective access mode for parallel MPI-IO reads (R7.4).
    if (use_parallel_) {
        status = nc_var_par_access(ncid_, varid, NC_COLLECTIVE);
        nc_check(status, "nc_var_par_access('" + meta.name + "', NC_COLLECTIVE) for read");
    }

    // Query variable rank from file to check if it's time-varying
    int ndims = 0;
    status = nc_inq_varndims(ncid_, varid, &ndims);
    nc_check(status, "nc_inq_varndims('" + meta.name + "')");

    bool is_time_varying = (ndims == meta.shape.rank + 1);

    // Compute start/count arrays.
    std::vector<std::size_t> start(ndims);
    std::vector<std::size_t> count(ndims);

    if (is_time_varying) {
        start[0] = static_cast<std::size_t>(timestep);
        count[0] = 1;
        if (bbox.has_value()) {
            // Selective read using bounding box (R5.7).
            const auto &box = bbox.value();
            for (int32_t d = 0; d < box.rank; ++d) {
                start[d + 1] = static_cast<std::size_t>(box.offsets[d]);
                count[d + 1] = static_cast<std::size_t>(box.extents[d]);
            }
        } else {
            // Full variable read.
            for (int32_t d = 0; d < meta.shape.rank; ++d) {
                start[d + 1] = 0;
                count[d + 1] = static_cast<std::size_t>(meta.shape.extents[d]);
            }
        }
    } else {
        if (bbox.has_value()) {
            // Selective read using bounding box (R5.7).
            const auto &box = bbox.value();
            for (int32_t d = 0; d < box.rank; ++d) {
                start[d] = static_cast<std::size_t>(box.offsets[d]);
                count[d] = static_cast<std::size_t>(box.extents[d]);
            }
        } else {
            // Full variable read.
            for (int32_t d = 0; d < meta.shape.rank; ++d) {
                start[d] = 0;
                count[d] = static_cast<std::size_t>(meta.shape.extents[d]);
            }
        }
    }

    // Calculate total bytes to read.
    std::size_t elem_size = dtype_byte_size(meta.dtype);
    std::size_t total_elems = 1;
    if (is_time_varying) {
        for (int32_t d = 0; d < meta.shape.rank; ++d) {
            total_elems *= static_cast<std::size_t>(count[d + 1]);
        }
    } else {
        for (int32_t d = 0; d < meta.shape.rank; ++d) {
            total_elems *= static_cast<std::size_t>(count[d]);
        }
    }
    std::size_t total_bytes = total_elems * elem_size;

    if (total_bytes > dst.capacity_bytes) {
        throw std::runtime_error("NetCDF_Driver::read: required " + std::to_string(total_bytes) + " bytes but buffer capacity is " +
                                 std::to_string(dst.capacity_bytes));
    }

    // Handle strided reads if bounding box has strides.
    bool has_strides = false;
    std::vector<ptrdiff_t> strides(ndims, 1);
    if (bbox.has_value()) {
        const auto &box = bbox.value();
        int offset = is_time_varying ? 1 : 0;
        for (int32_t d = 0; d < box.rank; ++d) {
            if (box.strides[d] > 1) {
                has_strides = true;
                strides[d + offset] = static_cast<ptrdiff_t>(box.strides[d]);
            }
        }
    }

    if (has_strides) {
        status = nc_get_vars(ncid_, varid, start.data(), count.data(), strides.data(), dst.data);
        nc_check(status, "nc_get_vars('" + meta.name + "')");
    } else {
        status = nc_get_vara(ncid_, varid, start.data(), count.data(), dst.data);
        nc_check(status, "nc_get_vara('" + meta.name + "')");
    }

    dst.used_bytes = total_bytes;

    (void)timestep;  // Timestep handling is managed by the variable's
                     // unlimited dimension if present.

#else
    (void)dst;
    (void)meta;
    (void)timestep;
    (void)bbox;
    throw std::runtime_error("NetCDF_Driver::read: AMIO built without netCDF support");
#endif
}

// ===================================================================
// flush -- sync the file to ensure durability.
// ===================================================================

void NetCDF_Driver::flush() {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (!is_open_) {
        throw std::runtime_error("NetCDF_Driver::flush: driver is not open");
    }

#ifdef AMIO_HAS_NETCDF
    int status = nc_sync(ncid_);
    nc_check(status, "nc_sync");
#else
    throw std::runtime_error("NetCDF_Driver::flush: AMIO built without netCDF support");
#endif
}

// ===================================================================
// close -- close the file handle and release resources.
// ===================================================================

void NetCDF_Driver::close() {
    std::lock_guard<std::mutex> lock(g_nc_driver_mutex);
    if (!is_open_) {
        return;  // Already closed -- idempotent.
    }

#ifdef AMIO_HAS_NETCDF
    int status = nc_close(ncid_);
    nc_check(status, "nc_close('" + file_path_ + "')");

    ncid_ = -1;
    comm_ = MPI_COMM_NULL;
    info_ = MPI_INFO_NULL;
#endif

    is_open_ = false;
    is_write_mode_ = false;
    file_path_.clear();
}

// ===================================================================
// describe_variable -- introspect a variable's dtype, shape, and
// timestep count from the open NetCDF file (Req 4.1, 4.2, 4.5, 9.1).
//
// Steps (design §3):
//   * nc_inq_varid(name)            -> variable id (NC_ENOTVAR => absent)
//   * nc_inq_var / nc_inq_vartype   -> netCDF element type -> amio_dtype_t
//   * nc_inq_varndims / dimids      -> rank + dimension ids
//   * nc_inq_dimlen(dimid)          -> per-dimension extent
//   * nc_inq_unlimdim               -> record dimension; if the variable's
//                                      leading dim is the unlimited dim its
//                                      length becomes total_timesteps and the
//                                      reported shape is the per-timestep
//                                      shape (the leading dim is dropped).
//                                      Otherwise total_timesteps = 1.
//
// Returns VariableInfo{found = false} when the driver is not open for
// reading, the variable is absent, the rank exceeds AMIO_MAX_RANK, or
// the netCDF element type has no AMIO dtype mapping.  A false result
// causes the read path to fail the read with AMIO_ERR_BACKEND_FAILURE.
// ===================================================================

#ifdef AMIO_HAS_NETCDF
namespace {

// Map a netCDF external type constant to an amio_dtype_t.  Returns true
// and writes *out on success; returns false for an unmapped nc type
// (e.g. NC_CHAR, NC_STRING, user-defined compound/enum types).
bool nc_type_to_dtype(int nc_type, amio_dtype_t &out) {
    switch (nc_type) {
        case NC_FLOAT:
            out = AMIO_DTYPE_F32;
            return true;
        case NC_DOUBLE:
            out = AMIO_DTYPE_F64;
            return true;
        case NC_BYTE:
            out = AMIO_DTYPE_I8;
            return true;
        case NC_SHORT:
            out = AMIO_DTYPE_I16;
            return true;
        case NC_INT:
            out = AMIO_DTYPE_I32;
            return true;
        case NC_INT64:
            out = AMIO_DTYPE_I64;
            return true;
        case NC_UBYTE:
            out = AMIO_DTYPE_U8;
            return true;
        case NC_USHORT:
            out = AMIO_DTYPE_U16;
            return true;
        case NC_UINT:
            out = AMIO_DTYPE_U32;
            return true;
        case NC_UINT64:
            out = AMIO_DTYPE_U64;
            return true;
        default:
            return false;
    }
}

}  // namespace
#endif  // AMIO_HAS_NETCDF

std::optional<std::string> NetCDF_Driver::get_text_attribute(const std::string &var_name, const std::string &attr_name) {
    if (!is_open_ || is_write_mode_) {
        return std::nullopt;
    }
#ifdef AMIO_HAS_NETCDF
    int varid = NC_GLOBAL;
    if (!var_name.empty()) {
        if (nc_inq_varid(ncid_, var_name.c_str(), &varid) != NC_NOERR) {
            return std::nullopt;
        }
    }
    nc_type att_type = NC_NAT;
    std::size_t len = 0;
    if (nc_inq_att(ncid_, varid, attr_name.c_str(), &att_type, &len) != NC_NOERR || len == 0) {
        return std::nullopt;
    }

    if (att_type == NC_CHAR) {
        std::string val(len, '\0');
        if (nc_get_att_text(ncid_, varid, attr_name.c_str(), val.data()) != NC_NOERR) {
            return std::nullopt;
        }
        // Trim only trailing NUL padding; interior NULs are preserved.
        const std::size_t last = val.find_last_not_of('\0');
        val.resize(last == std::string::npos ? 0 : last + 1);
        return val;
    }

    if (att_type == NC_STRING) {
        // `len` counts strings, not bytes: a scalar attribute has len == 1 and strs[0] is the whole
        // value; for a string vector only that first string is returned.  Interior NULs need no
        // handling -- the netCDF API takes and returns NUL-terminated C strings, so it cannot store them.
        std::vector<char *> strs(len, nullptr);
        if (nc_get_att_string(ncid_, varid, attr_name.c_str(), strs.data()) != NC_NOERR) {
            return std::nullopt;
        }
        std::string val;
        if (strs[0] != nullptr) {
            val.assign(strs[0]);
        }
        nc_free_string(len, strs.data());
        return val;
    }

    return std::nullopt;  // numeric attribute -- callers want get_numeric_attribute
#else
    (void)var_name;
    (void)attr_name;
    return std::nullopt;
#endif
}

std::optional<double> NetCDF_Driver::get_numeric_attribute(const std::string &var_name, const std::string &attr_name) {
    if (!is_open_ || is_write_mode_) {
        return std::nullopt;
    }
#ifdef AMIO_HAS_NETCDF
    int varid = NC_GLOBAL;
    if (!var_name.empty()) {
        if (nc_inq_varid(ncid_, var_name.c_str(), &varid) != NC_NOERR) {
            return std::nullopt;
        }
    }
    nc_type att_type = NC_NAT;
    std::size_t len = 0;
    if (nc_inq_att(ncid_, varid, attr_name.c_str(), &att_type, &len) != NC_NOERR || len == 0) {
        return std::nullopt;
    }
    if (att_type == NC_CHAR || att_type == NC_STRING) {
        return std::nullopt;
    }
    // nc_get_att_double converts from any numeric on-disk type.
    std::vector<double> vals(len);
    if (nc_get_att_double(ncid_, varid, attr_name.c_str(), vals.data()) != NC_NOERR) {
        return std::nullopt;
    }
    return vals.front();
#else
    (void)var_name;
    (void)attr_name;
    return std::nullopt;
#endif
}

VariableInfo NetCDF_Driver::describe_variable(const std::string &name) {
    VariableInfo info{};  // found == false by default.

    if (!is_open_ || is_write_mode_) {
        return info;
    }

#ifdef AMIO_HAS_NETCDF
    // Resolve the variable id; absence is a soft failure (found = false).
    int varid = -1;
    int status = nc_inq_varid(ncid_, name.c_str(), &varid);
    if (status == NC_ENOTVAR) {
        return info;
    }
    if (status != NC_NOERR) {
        return info;
    }

    // Element type.
    nc_type var_type = NC_NAT;
    status = nc_inq_vartype(ncid_, varid, &var_type);
    if (status != NC_NOERR) {
        return info;
    }
    amio_dtype_t dtype{};
    if (!nc_type_to_dtype(static_cast<int>(var_type), dtype)) {
        // Unmapped element type -> cannot describe robustly.
        return info;
    }

    // Rank + dimension ids.
    int ndims = 0;
    status = nc_inq_varndims(ncid_, varid, &ndims);
    if (status != NC_NOERR || ndims < 1 || ndims > AMIO_MAX_RANK) {
        return info;
    }
    std::vector<int> dimids(static_cast<std::size_t>(ndims), -1);
    status = nc_inq_vardimid(ncid_, varid, dimids.data());
    if (status != NC_NOERR) {
        return info;
    }

    // Identify the unlimited (record) dimension, if any.  netCDF-4 may
    // have multiple unlimited dimensions; query the full set and treat
    // any of them appearing as the variable's leading dimension as the
    // timestep axis.
    int n_unlim = 0;
    std::vector<int> unlim_dimids;
    {
        // nc_inq_unlimdims is available in netCDF-4; fall back to the
        // single-unlimited nc_inq_unlimdim otherwise.
        int probe = 0;
        if (nc_inq_unlimdims(ncid_, &probe, nullptr) == NC_NOERR) {
            n_unlim = probe;
            if (n_unlim > 0) {
                unlim_dimids.resize(static_cast<std::size_t>(n_unlim), -1);
                if (nc_inq_unlimdims(ncid_, &probe, unlim_dimids.data()) != NC_NOERR) {
                    unlim_dimids.clear();
                    n_unlim = 0;
                }
            }
        } else {
            int single = -1;
            if (nc_inq_unlimdim(ncid_, &single) == NC_NOERR && single >= 0) {
                unlim_dimids.push_back(single);
                n_unlim = 1;
            }
        }
    }

    auto is_unlimited = [&](int dimid) {
        for (int u : unlim_dimids) {
            if (u == dimid) {
                return true;
            }
        }
        return false;
    };

    // Identify a CF-style time/record dimension even when it is NOT declared
    // NC_UNLIMITED (common for climatologies and pre-processed inputs such as
    // CAMS-TEMPO, where `time` is a fixed-length dimension). Without this,
    // describe_variable would report the whole [time, ...] array as a single
    // record and read() would ingest every timestep at once. A dimension is
    // treated as the time axis when its name matches a common convention, or
    // its same-named coordinate variable carries CF metadata (axis="T",
    // standard_name="time", or units containing "since").
    auto is_cf_time_dim = [&](int dimid) {
        char dname[NC_MAX_NAME + 1] = {0};
        if (nc_inq_dimname(ncid_, dimid, dname) != NC_NOERR) {
            return false;
        }
        std::string dim_name(dname);
        std::string lower = dim_name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == "time" || lower == "t" || lower == "valid_time" || lower == "forecast_time" || lower == "record" || lower == "hour_index") {
            return true;
        }

        // Inspect a same-named coordinate variable's CF attributes.
        int coord_varid = -1;
        if (nc_inq_varid(ncid_, dim_name.c_str(), &coord_varid) != NC_NOERR) {
            return false;
        }
        auto att_equals = [&](const char *att, const char *expected) {
            std::size_t len = 0;
            if (nc_inq_attlen(ncid_, coord_varid, att, &len) != NC_NOERR || len == 0) {
                return false;
            }
            std::string val(len, '\0');
            if (nc_get_att_text(ncid_, coord_varid, att, val.data()) != NC_NOERR) {
                return false;
            }
            val.resize(std::strlen(val.c_str()));  // trim any trailing NUL
            std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return val == expected;
        };
        auto att_contains = [&](const char *att, const char *needle) {
            std::size_t len = 0;
            if (nc_inq_attlen(ncid_, coord_varid, att, &len) != NC_NOERR || len == 0) {
                return false;
            }
            std::string val(len, '\0');
            if (nc_get_att_text(ncid_, coord_varid, att, val.data()) != NC_NOERR) {
                return false;
            }
            std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return val.find(needle) != std::string::npos;
        };
        return att_equals("axis", "t") || att_equals("standard_name", "time") || att_contains("units", "since");
    };

    auto is_time_axis = [&](int dimid) { return is_unlimited(dimid) || is_cf_time_dim(dimid); };

    // Read per-dimension extents.
    std::vector<std::size_t> extents(static_cast<std::size_t>(ndims), 0);
    for (int d = 0; d < ndims; ++d) {
        std::size_t len = 0;
        status = nc_inq_dimlen(ncid_, dimids[static_cast<std::size_t>(d)], &len);
        if (status != NC_NOERR) {
            return info;
        }
        extents[static_cast<std::size_t>(d)] = len;
    }

    // If the leading dimension is the record/unlimited dimension, its
    // length is total_timesteps and the reported shape is the
    // per-timestep shape (leading dim dropped).  Otherwise the variable
    // is not time-varying (total_timesteps = 1) and the full shape is
    // reported as-is.
    std::int64_t total_timesteps = 1;
    int shape_start = 0;
    if (ndims >= 1 && is_time_axis(dimids[0])) {
        total_timesteps = static_cast<std::int64_t>(extents[0]);
        shape_start = 1;
    }

    int reported_rank = ndims - shape_start;
    if (reported_rank < 1 && ndims >= 1) {
        // The variable is nothing but the record dimension, so it is that
        // axis's coordinate variable rather than a field sampled along it
        // (e.g. `double time(time)`).  Describe it whole, so one read returns
        // the entire axis instead of the variable being undescribable.
        total_timesteps = 1;
        shape_start = 0;
        reported_rank = ndims;
    }
    if (reported_rank < 1 || reported_rank > AMIO_MAX_RANK) {
        return info;
    }

    info.shape.rank = reported_rank;
    for (int d = 0; d < reported_rank; ++d) {
        info.shape.extents[d] = static_cast<std::int64_t>(extents[static_cast<std::size_t>(d + shape_start)]);
        info.shape.strides[d] = 0;  // contiguous / row-major
    }

    info.dtype = dtype;
    info.total_timesteps = total_timesteps > 0 ? total_timesteps : 1;
    info.found = true;
    return info;

#else
    (void)name;
    return info;
#endif
}

// ===================================================================
// Static factory registration.
//
// BackendRegistrar<NetCDF_Driver>("netcdf4") registers the driver
// with the BackendFactory at static initialization time so that
// BackendFactory::build("netcdf4", err) instantiates this driver.
// ===================================================================

namespace {
BackendRegistrar<NetCDF_Driver> reg_netcdf4("netcdf4");
}  // anonymous namespace

}  // namespace amio::detail

extern "C" void amio_register_netcdf_driver() {}
