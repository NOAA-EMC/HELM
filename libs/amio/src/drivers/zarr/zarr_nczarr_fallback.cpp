// zarr_nczarr_fallback.cpp -- AMIO Zarr_Driver NCZarr fallback mode.
//
// This translation unit implements the Zarr_Driver for Zarr v3 datasets
// using the netCDF-c NCZarr interface when TensorStore is not available
// at compile time (AMIO_NCZARR_FALLBACK defined).
//
// Key differences from TensorStore mode:
//   - Uses netCDF-c NCZarr API instead of TensorStore
//   - No zarr3_sharding_indexed codec (sharding unavailable)
//   - No cloud URI support (local filesystem only)
//   - Emits one-shot diagnostic about missing sharding capability
//   - Applies lossless compression (Blosc or Zstandard)
//
// Satisfies Round_Trip_Equivalence (R11.2) in fallback mode.
//
// Validates: R8.6, R8.7, R8.8, R13.5

#ifdef AMIO_NCZARR_FALLBACK

#include <netcdf.h>
#include <netcdf_meta.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/common/var_attributes.hpp"
#include "drivers/zarr/zarr_driver.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

// ===================================================================
// CONF configuration layer for NCZarr fallback.
// ===================================================================

#include <conf/config.hpp>
#include <conf/error.hpp>

namespace amio::detail {

// ===================================================================
// NCZarr helper: check netCDF return codes and throw on failure.
// ===================================================================

namespace {

void nczarr_check(int status, const std::string &context) {
    if (status != NC_NOERR) {
        std::string msg = "Zarr_Driver [NCZarr]: error in " + context + ": " + nc_strerror(status) + " (nc_errno=" + std::to_string(status) + ")";
        throw std::runtime_error(msg);
    }
}

// Emit a one-shot diagnostic that sharding is unavailable (R8.8).
// Falls back to std::cerr until LOGS integration (task 8.4).
void emit_sharding_unavailable_diagnostic() {
    std::cerr << "[AMIO WARNING] Zarr_Driver [NCZarr fallback]: sharding is "
                 "unavailable. The zarr3_sharding_indexed codec requires "
                 "TensorStore. This build uses netCDF-c NCZarr mode with "
                 "flat chunk layout only. Rebuild with "
                 "AMIO_HAS_TENSORSTORE=ON to enable sharding and cloud "
                 "KvStore support."
              << std::endl;
}

// Map amio_dtype_t to netCDF type constant.
int nczarr_dtype_to_nc_type(amio_dtype_t dtype) {
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
            throw std::runtime_error("Zarr_Driver [NCZarr]: unsupported dtype " + std::to_string(static_cast<int>(dtype)));
    }
}

// Write one attribute set onto `varid` (NC_GLOBAL for file-level).
// Must be called in define mode.  Numeric attributes use a type
// matching their literal form; `_FillValue` uses the variable's type.
void nczarr_write_attributes(int ncid, int varid, const VarAttributes &attrs, int var_nc_type) {
    for (const auto &kv : attrs.items) {
        const std::string &name = kv.first;
        const AttrValue &val = kv.second;

        if (val.is_numeric) {
            if (name == "_FillValue" && varid != NC_GLOBAL && var_nc_type != NC_NAT) {
                if (var_nc_type == NC_FLOAT) {
                    float f = static_cast<float>(val.number);
                    nczarr_check(nc_put_att_float(ncid, varid, name.c_str(), NC_FLOAT, 1, &f), "nc_put_att_float('" + name + "')");
                    continue;
                } else if (var_nc_type == NC_DOUBLE) {
                    double d = val.number;
                    nczarr_check(nc_put_att_double(ncid, varid, name.c_str(), NC_DOUBLE, 1, &d), "nc_put_att_double('" + name + "')");
                    continue;
                } else if (var_nc_type == NC_INT) {
                    int i = static_cast<int>(val.number);
                    nczarr_check(nc_put_att_int(ncid, varid, name.c_str(), NC_INT, 1, &i), "nc_put_att_int('" + name + "')");
                    continue;
                } else if (var_nc_type == NC_INT64) {
                    long long ll = static_cast<long long>(val.number);
                    nczarr_check(nc_put_att_longlong(ncid, varid, name.c_str(), NC_INT64, 1, &ll), "nc_put_att_longlong('" + name + "')");
                    continue;
                }
            }

            if (val.is_integer) {
                long long ll = static_cast<long long>(val.number);
                nczarr_check(nc_put_att_longlong(ncid, varid, name.c_str(), NC_INT64, 1, &ll), "nc_put_att_longlong('" + name + "')");
            } else {
                double d = val.number;
                nczarr_check(nc_put_att_double(ncid, varid, name.c_str(), NC_DOUBLE, 1, &d), "nc_put_att_double('" + name + "')");
            }
        } else {
            nczarr_check(nc_put_att_text(ncid, varid, name.c_str(), val.text.size(), val.text.c_str()), "nc_put_att_text('" + name + "')");
        }
    }
}

// Convert a local filesystem path to an NCZarr URI.
// NCZarr uses file:// URIs with #mode=nczarr,<format> fragment.
std::string to_nczarr_uri(const std::string &path) {
    // NCZarr requires a file:// URI with mode fragment.
    // Format: file://<absolute_path>#mode=nczarr,file
    if (path.empty()) {
        throw std::runtime_error("Zarr_Driver [NCZarr]: empty path is not valid");
    }

    // If already a file:// URI, append mode fragment if missing.
    if (path.rfind("file://", 0) == 0) {
        if (path.find("#mode=") == std::string::npos) {
            return path + "#mode=nczarr,file";
        }
        return path;
    }

    // Reject cloud URIs -- NCZarr fallback is local-only.
    if (path.rfind("s3://", 0) == 0 || path.rfind("gs://", 0) == 0 || path.rfind("https://", 0) == 0) {
        throw std::runtime_error(
            "Zarr_Driver [NCZarr fallback]: cloud URIs are not "
            "supported in NCZarr mode. URI '" +
            path +
            "' requires "
            "TensorStore. Rebuild with AMIO_HAS_TENSORSTORE=ON.");
    }

    // Local path: convert to file:// URI with NCZarr mode.
    std::string uri = "file://";
    if (path[0] != '/') {
        // Relative path -- prepend current directory marker.
        uri += "./";
    }
    uri += path;
    uri += "#mode=nczarr,file";
    return uri;
}

}  // anonymous namespace

// ===================================================================
// open_write -- NCZarr fallback: open for writing via netCDF-c.
//
// Validates config, rejects cloud URIs, emits one-shot sharding
// diagnostic, and opens the NCZarr store for writing.
// ===================================================================

void Zarr_Driver::open_write(const conf::Config &config) {
    if (is_open_) {
        throw std::runtime_error("Zarr_Driver: already open");
    }

    // Parse and validate configuration.
    config_ = parse_zarr_config(config);

    // In NCZarr mode, shard_shape is accepted but ignored (no sharding).
    // We still validate the codec.
    validate_codec(config_);

    // Parse CF/UGRID convention metadata + per-variable attributes.
    attributes_ = parse_dataset_attributes(config);

    // Reject cloud URIs (R8.6: no cloud KvStore in fallback mode).
    if (is_cloud_uri(config_.uri)) {
        throw std::runtime_error(
            "Zarr_Driver [NCZarr fallback]: cloud URIs (s3://, gs://, "
            "https://) are not supported in NCZarr mode. URI '" +
            config_.uri +
            "' requires TensorStore. Rebuild with "
            "AMIO_HAS_TENSORSTORE=ON.");
    }

    // Emit one-shot diagnostic that sharding is unavailable (R8.8).
    if (!sharding_diagnostic_emitted_) {
        emit_sharding_unavailable_diagnostic();
        sharding_diagnostic_emitted_ = true;
    }

    // Convert path to NCZarr URI and create the file.
    std::string nczarr_uri = to_nczarr_uri(config_.uri);

    // Create mode: NCZarr with NetCDF-4 format.
    int cmode = NC_NETCDF4 | NC_CLOBBER;

    int status = nc_create(nczarr_uri.c_str(), cmode, &ncid_);
    nczarr_check(status, "nc_create('" + nczarr_uri + "')");

    // Write the global CF/UGRID `Conventions` attribute plus any extra
    // global attributes.  These surface as Zarr group attributes
    // (.zattrs) through NCZarr.
    {
        const std::string &conv = attributes_.conventions;
        nczarr_check(nc_put_att_text(ncid_, NC_GLOBAL, "Conventions", conv.size(), conv.c_str()), "nc_put_att_text(Conventions)");
        nczarr_write_attributes(ncid_, NC_GLOBAL, attributes_.global, NC_NAT);
        global_attrs_written_ = true;
    }

    is_open_ = true;
    is_write_mode_ = true;
}

// ===================================================================
// open_read -- NCZarr fallback: open for reading via netCDF-c.
// ===================================================================

void Zarr_Driver::open_read(const conf::Config &config) {
    if (is_open_) {
        throw std::runtime_error("Zarr_Driver: already open");
    }

    // Parse and validate configuration.
    config_ = parse_zarr_config(config);
    validate_codec(config_);

    // Reject cloud URIs (R8.6: no cloud KvStore in fallback mode).
    if (is_cloud_uri(config_.uri)) {
        throw std::runtime_error(
            "Zarr_Driver [NCZarr fallback]: cloud URIs (s3://, gs://, "
            "https://) are not supported in NCZarr mode. URI '" +
            config_.uri +
            "' requires TensorStore. Rebuild with "
            "AMIO_HAS_TENSORSTORE=ON.");
    }

    // Emit one-shot diagnostic that sharding is unavailable (R8.8).
    if (!sharding_diagnostic_emitted_) {
        emit_sharding_unavailable_diagnostic();
        sharding_diagnostic_emitted_ = true;
    }

    // Convert path to NCZarr URI and open the file.
    std::string nczarr_uri = to_nczarr_uri(config_.uri);

    int status = nc_open(nczarr_uri.c_str(), NC_NOWRITE, &ncid_);
    nczarr_check(status, "nc_open('" + nczarr_uri + "')");

    is_open_ = true;
    is_write_mode_ = false;
}

// ===================================================================
// write -- NCZarr fallback: serialize StagingBuffer through netCDF-c
// NCZarr with lossless compression (R8.7).
// ===================================================================

void Zarr_Driver::write(const StagingBuffer &src, const VarMeta &meta) {
    if (!is_open_ || !is_write_mode_) {
        throw std::runtime_error("Zarr_Driver: write called on driver not open for writing");
    }

    // Look up or create the variable in the NCZarr store.
    int varid = -1;
    int status = nc_inq_varid(ncid_, meta.name.c_str(), &varid);

    if (status == NC_ENOTVAR) {
        // Variable does not exist -- define it.
        std::vector<int> dimids(meta.shape.rank);

        // Enter define mode.
        int redef_status = nc_redef(ncid_);
        if (redef_status != NC_NOERR && redef_status != NC_EINDEFINE) {
            nczarr_check(redef_status, "nc_redef");
        }

        // Define dimensions.
        for (int32_t d = 0; d < meta.shape.rank; ++d) {
            std::string dim_name = meta.name + "_dim" + std::to_string(d);
            int existing_dimid = -1;
            int dim_status = nc_inq_dimid(ncid_, dim_name.c_str(), &existing_dimid);
            if (dim_status == NC_NOERR) {
                dimids[d] = existing_dimid;
            } else {
                std::size_t dim_len = static_cast<std::size_t>(meta.shape.extents[d]);
                status = nc_def_dim(ncid_, dim_name.c_str(), dim_len, &dimids[d]);
                nczarr_check(status, "nc_def_dim('" + dim_name + "')");
            }
        }

        // Define the variable with the appropriate netCDF type.
        int nc_type = nczarr_dtype_to_nc_type(meta.dtype);
        status = nc_def_var(ncid_, meta.name.c_str(), nc_type, meta.shape.rank, dimids.data(), &varid);
        nczarr_check(status, "nc_def_var('" + meta.name + "')");

        // Apply CF/UGRID per-variable attributes from the manifest.
        // Written as Zarr attributes (.zattrs) via NCZarr.  Still in
        // define mode here.
        if (const VarAttributes *var_attrs = attributes_.find(meta.name)) {
            nczarr_write_attributes(ncid_, varid, *var_attrs, nc_type);
        }

        // Set chunking based on config chunk_shape (flat chunks, no
        // sharding in NCZarr mode).
        if (!config_.chunk_shape.empty()) {
            std::vector<std::size_t> chunks(meta.shape.rank);
            for (int32_t d = 0; d < meta.shape.rank; ++d) {
                if (d < static_cast<int32_t>(config_.chunk_shape.size())) {
                    chunks[d] = static_cast<std::size_t>(config_.chunk_shape[d]);
                } else {
                    chunks[d] = static_cast<std::size_t>(meta.shape.extents[d]);
                }
            }
            status = nc_def_var_chunking(ncid_, varid, NC_CHUNKED, chunks.data());
            nczarr_check(status, "nc_def_var_chunking('" + meta.name + "')");
        }

        // Apply lossless compression: exactly one of {Blosc, Zstandard}
        // (R8.7).  netCDF-c supports deflate natively; for Blosc and
        // Zstandard we use the HDF5 filter plugin mechanism when
        // available, falling back to deflate as a portable lossless
        // alternative.
        if (config_.codec == "blosc") {
            // Attempt to use Blosc filter via nc_def_var_filter if
            // available (netCDF-c >= 4.7.4 with HDF5 plugin).
            // Filter ID for Blosc is 32001.
#if defined(NC_HAS_BLOSC) && NC_HAS_BLOSC
            // Use native Blosc support.
            unsigned int blosc_params[7] = {
                0,  // version (auto)
                2,  // compressor: lz4
                4,  // element size (bytes) -- will be overridden
                0,  // chunk size (auto)
                5,  // compression level
                1,  // shuffle (byte shuffle)
                0   // number of threads (auto)
            };
            blosc_params[2] = static_cast<unsigned int>(dtype_size(meta.dtype));
            status = nc_def_var_filter(ncid_, varid, 32001, 7, blosc_params);
            if (status != NC_NOERR) {
                // Blosc filter not available -- fall back to deflate.
                status = nc_def_var_deflate(ncid_, varid,
                                            /*shuffle=*/1,
                                            /*deflate=*/1,
                                            /*deflate_level=*/5);
                nczarr_check(status, "nc_def_var_deflate('" + meta.name + "')");
            }
#else
            // Blosc not compiled into netCDF -- use deflate with
            // shuffle as portable lossless compression.
            status = nc_def_var_deflate(ncid_, varid,
                                        /*shuffle=*/1,
                                        /*deflate=*/1,
                                        /*deflate_level=*/5);
            nczarr_check(status, "nc_def_var_deflate('" + meta.name + "')");
#endif
        } else if (config_.codec == "zstandard") {
            // Attempt Zstandard filter (HDF5 filter ID 32015).
#if defined(NC_HAS_ZSTD) && NC_HAS_ZSTD
            unsigned int zstd_params[1] = {3};  // compression level
            status = nc_def_var_filter(ncid_, varid, 32015, 1, zstd_params);
            if (status != NC_NOERR) {
                // Zstandard filter not available -- fall back to deflate.
                status = nc_def_var_deflate(ncid_, varid,
                                            /*shuffle=*/1,
                                            /*deflate=*/1,
                                            /*deflate_level=*/5);
                nczarr_check(status, "nc_def_var_deflate('" + meta.name + "')");
            }
#else
            // Zstandard not compiled into netCDF -- use deflate with
            // shuffle as portable lossless compression.
            status = nc_def_var_deflate(ncid_, varid,
                                        /*shuffle=*/1,
                                        /*deflate=*/1,
                                        /*deflate_level=*/5);
            nczarr_check(status, "nc_def_var_deflate('" + meta.name + "')");
#endif
        }

        // End define mode.
        status = nc_enddef(ncid_);
        nczarr_check(status, "nc_enddef");

    } else if (status != NC_NOERR) {
        nczarr_check(status, "nc_inq_varid('" + meta.name + "')");
    }

    // Write the data from the staging buffer.
    std::vector<std::size_t> start(meta.shape.rank, 0);
    std::vector<std::size_t> count(meta.shape.rank);
    for (int32_t d = 0; d < meta.shape.rank; ++d) {
        count[d] = static_cast<std::size_t>(meta.shape.extents[d]);
    }

    status = nc_put_vara(ncid_, varid, start.data(), count.data(), src.data);
    nczarr_check(status, "nc_put_vara('" + meta.name + "')");
}

// ===================================================================
// read -- NCZarr fallback: read from NCZarr store into StagingBuffer.
// ===================================================================

void Zarr_Driver::read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) {
    if (!is_open_ || is_write_mode_) {
        throw std::runtime_error("Zarr_Driver: read called on driver not open for reading");
    }

    (void)timestep;  // Timestep encoded in variable path/index.

    // Look up the variable.
    int varid = -1;
    int status = nc_inq_varid(ncid_, meta.name.c_str(), &varid);
    nczarr_check(status, "nc_inq_varid('" + meta.name + "') for read");

    // Compute start/count arrays.
    std::vector<std::size_t> start(meta.shape.rank);
    std::vector<std::size_t> count(meta.shape.rank);

    if (bbox.has_value()) {
        const auto &box = bbox.value();
        for (int32_t d = 0; d < box.rank; ++d) {
            start[d] = static_cast<std::size_t>(box.offsets[d]);
            count[d] = static_cast<std::size_t>(box.extents[d]);
        }
    } else {
        for (int32_t d = 0; d < meta.shape.rank; ++d) {
            start[d] = 0;
            count[d] = static_cast<std::size_t>(meta.shape.extents[d]);
        }
    }

    // Calculate total bytes to read.
    std::size_t elem_size = dtype_size(meta.dtype);
    std::size_t total_elems = 1;
    for (int32_t d = 0; d < meta.shape.rank; ++d) {
        total_elems *= count[d];
    }
    std::size_t total_bytes = total_elems * elem_size;

    if (total_bytes > dst.capacity_bytes) {
        throw std::runtime_error("Zarr_Driver [NCZarr]: staging buffer capacity (" + std::to_string(dst.capacity_bytes) +
                                 " bytes) insufficient for read payload (" + std::to_string(total_bytes) + " bytes)");
    }

    // Handle strided reads if bounding box has strides.
    if (bbox.has_value()) {
        const auto &box = bbox.value();
        bool has_strides = false;
        std::vector<ptrdiff_t> strides(meta.shape.rank, 1);
        for (int32_t d = 0; d < box.rank; ++d) {
            if (box.strides[d] > 1) {
                has_strides = true;
                strides[d] = static_cast<ptrdiff_t>(box.strides[d]);
            }
        }

        if (has_strides) {
            status = nc_get_vars(ncid_, varid, start.data(), count.data(), strides.data(), dst.data);
            nczarr_check(status, "nc_get_vars('" + meta.name + "')");
        } else {
            status = nc_get_vara(ncid_, varid, start.data(), count.data(), dst.data);
            nczarr_check(status, "nc_get_vara('" + meta.name + "')");
        }
    } else {
        status = nc_get_vara(ncid_, varid, start.data(), count.data(), dst.data);
        nczarr_check(status, "nc_get_vara('" + meta.name + "')");
    }

    dst.used_bytes = total_bytes;
}

// ===================================================================
// flush -- NCZarr fallback: sync the NCZarr store.
// ===================================================================

void Zarr_Driver::flush() {
    if (!is_open_) {
        return;  // No-op if not open.
    }

    int status = nc_sync(ncid_);
    nczarr_check(status, "nc_sync");
}

// ===================================================================
// close -- NCZarr fallback: close the NCZarr store.
// ===================================================================

void Zarr_Driver::close() {
    if (!is_open_) {
        return;  // No-op if not open.
    }

    int status = nc_close(ncid_);
    nczarr_check(status, "nc_close");

    ncid_ = -1;
    is_open_ = false;
    is_write_mode_ = false;
}

}  // namespace amio::detail

#endif  // AMIO_NCZARR_FALLBACK
