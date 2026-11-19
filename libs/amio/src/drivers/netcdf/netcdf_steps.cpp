#include "drivers/netcdf/netcdf_steps.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

#include "drivers/common/netcdf_mutex.hpp"

#ifdef AMIO_HAS_NETCDF
#include <netcdf_par.h>
#endif

namespace amio::detail {

bool make_netcdf_step_layout(const std::string &root, std::uint64_t step_id, NetCDFStepLayout &layout, std::string &error) {
    if (root.empty()) {
        error = "NetCDF step collection root cannot be empty";
        return false;
    }
    layout.payload_path = root + "/payloads/" + std::to_string(step_id) + ".nc";
    return true;
}

bool validate_netcdf_step_schema(const Schema &schema, std::string &error) {
    if (!schema.sealed()) {
        error = "NetCDF step schema must be sealed before payload creation";
        return false;
    }
    std::map<std::string, std::uint64_t, std::less<>> dimensions;
    for (const auto &[name, variable] : schema.variables()) {
        const auto codec = variable.attributes.find("amio.codec");
        if (codec != variable.attributes.end()) {
            const auto *selected = std::get_if<std::string>(&codec->second.value);
            if (selected == nullptr || (*selected != "none" && *selected != "deflate")) {
                error = "NetCDF step fields support only the native lossless deflate codec: " + name;
                return false;
            }
        }
        if (name.empty() || name.size() > 255 || variable.dimensions.empty()) {
            error = "NetCDF step fields require a legal variable name and at least one dimension";
            return false;
        }
        std::set<std::string, std::less<>> local_dims;
        for (const auto &dimension : variable.dimensions) {
            if (dimension.name.empty() || dimension.name.size() > 255 || dimension.extent == 0 ||
                dimension.extent > std::numeric_limits<std::size_t>::max()) {
                error = "NetCDF step dimensions require a legal name and positive extent: " + name;
                return false;
            }
            if (!local_dims.insert(dimension.name).second) {
                error = "NetCDF step variable repeats dimension " + dimension.name;
                return false;
            }
            const auto [it, inserted] = dimensions.emplace(dimension.name, dimension.extent);
            if (!inserted && it->second != dimension.extent) {
                error = "NetCDF named dimension has inconsistent extents: " + dimension.name;
                return false;
            }
        }
    }
    return true;
}

bool validate_netcdf_region(const RegionSelection &region, const std::vector<std::uint64_t> &global, std::string &error) {
    if (region.offsets.size() != global.size() || region.extents.size() != global.size()) {
        error = "NetCDF region rank differs from native variable rank";
        return false;
    }
    bool has_zero = false;
    for (std::size_t i = 0; i < global.size(); ++i) {
        const auto offset = region.offsets[i], count = region.extents[i];
        if (count == 0) {
            has_zero = true;
            if (offset > global[i]) {
                error = "empty NetCDF region offset is out of bounds";
                return false;
            }
        } else if (offset >= global[i] || count > global[i] - offset) {
            error = "NetCDF region exceeds global dimension extent";
            return false;
        }
    }
    if (region.empty != has_zero) {
        error = "NetCDF region empty flag disagrees with its count vector";
        return false;
    }
    return true;
}

#ifdef AMIO_HAS_NETCDF
namespace {
void check_nc(int status, const std::string &where) {
    if (status != NC_NOERR) throw std::runtime_error(where + ": " + nc_strerror(status));
}
int nc_type(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return NC_FLOAT;
        case AMIO_DTYPE_F64:
            return NC_DOUBLE;
        case AMIO_DTYPE_I8:
            return NC_BYTE;
        case AMIO_DTYPE_U8:
            return NC_UBYTE;
        case AMIO_DTYPE_I16:
            return NC_SHORT;
        case AMIO_DTYPE_U16:
            return NC_USHORT;
        case AMIO_DTYPE_I32:
            return NC_INT;
        case AMIO_DTYPE_U32:
            return NC_UINT;
        case AMIO_DTYPE_I64:
            return NC_INT64;
        case AMIO_DTYPE_U64:
            return NC_UINT64;
        default:
            throw std::runtime_error("unsupported dtype in NetCDF step schema");
    }
}
void put_attribute(int ncid, int varid, const std::string &key, const Attribute &attr) {
    std::visit(
        [&](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, bool>) {
                unsigned char v = value ? 1 : 0;
                check_nc(nc_put_att_uchar(ncid, varid, key.c_str(), NC_UBYTE, 1, &v), "attribute " + key);
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                long long v = value;
                check_nc(nc_put_att_longlong(ncid, varid, key.c_str(), NC_INT64, 1, &v), "attribute " + key);
            } else if constexpr (std::is_same_v<T, std::uint64_t>) {
                unsigned long long v = value;
                check_nc(nc_put_att_ulonglong(ncid, varid, key.c_str(), NC_UINT64, 1, &v), "attribute " + key);
            } else if constexpr (std::is_same_v<T, double>)
                check_nc(nc_put_att_double(ncid, varid, key.c_str(), NC_DOUBLE, 1, &value), "attribute " + key);
            else if constexpr (std::is_same_v<T, std::string>)
                check_nc(nc_put_att_text(ncid, varid, key.c_str(), value.size(), value.data()), "attribute " + key);
            else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                check_nc(nc_put_att_longlong(ncid, varid, key.c_str(), NC_INT64, value.size(), reinterpret_cast<const long long *>(value.data())),
                         "attribute " + key);
            else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>)
                check_nc(nc_put_att_ulonglong(ncid, varid, key.c_str(), NC_UINT64, value.size(),
                                              reinterpret_cast<const unsigned long long *>(value.data())),
                         "attribute " + key);
            else if constexpr (std::is_same_v<T, std::vector<double>>)
                check_nc(nc_put_att_double(ncid, varid, key.c_str(), NC_DOUBLE, value.size(), value.data()), "attribute " + key);
            else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                std::vector<const char *> strings;
                for (const auto &s : value) strings.push_back(s.c_str());
                check_nc(nc_put_att_string(ncid, varid, key.c_str(), strings.size(), strings.data()), "attribute " + key);
            }
        },
        attr.value);
}
std::vector<std::size_t> to_size(const std::vector<std::uint64_t> &values) {
    std::vector<std::size_t> out;
    out.reserve(values.size());
    for (auto v : values) {
        if (v > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("NetCDF selection exceeds size_t");
        out.push_back(static_cast<std::size_t>(v));
    }
    return out;
}
}  // namespace

bool define_netcdf_step_schema(int ncid, const Schema &schema, double model_time, std::string &error) {
    std::lock_guard<std::mutex> lock(netcdf_mutex());
    try {
        if (!validate_netcdf_step_schema(schema, error)) return false;
        if (!std::isfinite(model_time)) throw std::runtime_error("NetCDF step model time must be finite");
        check_nc(nc_put_att_double(ncid, NC_GLOBAL, "amio_model_time", NC_DOUBLE, 1, &model_time), "write model time");
        check_nc(nc_put_att_text(ncid, NC_GLOBAL, "amio_schema_digest", schema.digest().size(), schema.digest().data()), "write schema digest");
        std::map<std::string, int, std::less<>> dimids;
        for (const auto &[name, var] : schema.variables())
            for (const auto &dim : var.dimensions) {
                if (!dimids.count(dim.name)) {
                    int id = -1;
                    check_nc(nc_def_dim(ncid, dim.name.c_str(), static_cast<std::size_t>(dim.extent), &id), "define dimension " + dim.name);
                    dimids.emplace(dim.name, id);
                }
            }
        for (const auto &[name, var] : schema.variables()) {
            std::vector<int> ids;
            ids.reserve(var.dimensions.size());
            for (const auto &dim : var.dimensions) ids.push_back(dimids.at(dim.name));
            int id = -1;
            check_nc(nc_def_var(ncid, name.c_str(), nc_type(var.stored_dtype), ids.size(), ids.data(), &id), "define variable " + name);
            const auto codec = var.attributes.find("amio.codec");
            if (codec != var.attributes.end()) {
                const auto *selected = std::get_if<std::string>(&codec->second.value);
                if (selected != nullptr && *selected == "deflate")
                    check_nc(nc_def_var_deflate(ncid, id, 0, 1, 1), "enable native deflate for " + name);
            }
            for (const auto &[key, value] : var.attributes) put_attribute(ncid, id, key, value);
        }
        check_nc(nc_enddef(ncid), "finish NetCDF step definition");
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}

bool write_netcdf_step_region(int ncid, const std::string &name, const RegionSelection &region, const void *data, std::string &error,
                              bool independent) {
    std::lock_guard<std::mutex> lock(netcdf_mutex());
    try {
        int varid = -1;
        check_nc(nc_inq_varid(ncid, name.c_str(), &varid), "find variable " + name);
        int rank = 0;
        check_nc(nc_inq_varndims(ncid, varid, &rank), "query variable rank");
        if (rank < 0 || static_cast<std::size_t>(rank) != region.extents.size() || region.offsets.size() != region.extents.size())
            throw std::runtime_error("NetCDF write selection rank mismatch");
        std::vector<int> dimids(static_cast<std::size_t>(rank));
        check_nc(nc_inq_vardimid(ncid, varid, dimids.data()), "query variable dimensions");
        std::vector<std::uint64_t> global(static_cast<std::size_t>(rank));
        for (int d = 0; d < rank; ++d) {
            std::size_t n = 0;
            check_nc(nc_inq_dimlen(ncid, dimids[static_cast<std::size_t>(d)], &n), "query dimension extent");
            global[static_cast<std::size_t>(d)] = n;
        }
        if (!validate_netcdf_region(region, global, error)) return false;
        const auto start = to_size(region.offsets), count = to_size(region.extents);
        if (!region.empty && data == nullptr) throw std::runtime_error("nonempty NetCDF selection has null data");
#ifdef AMIO_HAS_PARALLEL_NETCDF
        if (!independent) check_nc(nc_var_par_access(ncid, varid, NC_COLLECTIVE), "set collective variable access");
#else
        (void)independent;
#endif
        // Collective mode requires every rank, including zero-count ranks.
        // Prepared independent mode permits different numbers of bounded writes.
        check_nc(nc_put_vara(ncid, varid, start.data(), count.data(), data), "hyperslab write " + name);
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}

bool prepare_netcdf_independent_region_writes(int ncid, const std::string &name, std::string &error, bool parallel) {
    std::lock_guard<std::mutex> lock(netcdf_mutex());
    try {
        int varid = -1;
        check_nc(nc_inq_varid(ncid, name.c_str(), &varid), "find variable " + name);
#ifdef AMIO_HAS_PARALLEL_NETCDF
        if (parallel) check_nc(nc_var_par_access(ncid, varid, NC_INDEPENDENT), "set independent variable access " + name);
#else
        (void)parallel;
#endif
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}

bool read_netcdf_step_region(const std::string &path, const std::string &name, const RegionSelection &region, void *data, std::string &error) {
    std::lock_guard<std::mutex> lock(netcdf_mutex());
    int ncid = -1;
    try {
        check_nc(nc_open(path.c_str(), NC_NOWRITE, &ncid), "open closed NetCDF step " + path);
        int varid = -1;
        check_nc(nc_inq_varid(ncid, name.c_str(), &varid), "find variable " + name);
        int rank = 0;
        check_nc(nc_inq_varndims(ncid, varid, &rank), "query variable rank");
        if (static_cast<std::size_t>(rank) != region.extents.size() || region.offsets.size() != region.extents.size())
            throw std::runtime_error("NetCDF read selection rank mismatch");
        const auto start = to_size(region.offsets), count = to_size(region.extents);
        if (!region.empty && data == nullptr) throw std::runtime_error("nonempty NetCDF selection has null data");
        check_nc(nc_get_vara(ncid, varid, start.data(), count.data(), data), "independent hyperslab read " + name);
        check_nc(nc_close(ncid), "close NetCDF step after read");
        ncid = -1;
        return true;
    } catch (const std::exception &e) {
        if (ncid >= 0) nc_close(ncid);
        error = e.what();
        return false;
    }
}
#endif

}  // namespace amio::detail
