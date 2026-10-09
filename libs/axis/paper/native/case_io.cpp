// SPDX-License-Identifier: Apache-2.0
#include "case_io.hpp"

#include <algorithm>
#include <set>
namespace paper {
void nc_check(int rc) {
    if (rc != NC_NOERR)
        throw Error(2, nc_strerror(rc));
}
File::File(const std::filesystem::path &p, bool write) {
    nc_check(write ? nc_create(p.c_str(), NC_NETCDF4 | NC_NOCLOBBER, &id)
                   : nc_open(p.c_str(), NC_NOWRITE, &id));
}
File::~File() {
    if (id >= 0)
        nc_close(id);
}
int group(int id, const char *n) {
    int g;
    nc_check(nc_inq_ncid(id, n, &g));
    return g;
}
std::string attr(int id, const char *n, const std::string &fallback) {
    size_t len;
    int rc = nc_inq_attlen(id, NC_GLOBAL, n, &len);
    if (rc == NC_ENOTATT)
        return fallback;
    nc_check(rc);
    std::string s(len, ' ');
    nc_check(nc_get_att_text(id, NC_GLOBAL, n, s.data()));
    return s;
}
static size_t length(int id, int var) {
    int rank;
    nc_check(nc_inq_varndims(id, var, &rank));
    std::vector<int> dims(rank);
    nc_check(nc_inq_vardimid(id, var, dims.data()));
    size_t n = 1;
    for (int d : dims) {
        size_t x;
        nc_check(nc_inq_dimlen(id, d, &x));
        n *= x;
    }
    return n;
}
std::vector<double> doubles(int id, const char *n) {
    int v;
    nc_check(nc_inq_varid(id, n, &v));
    std::vector<double> x(length(id, v));
    nc_check(nc_get_var_double(id, v, x.data()));
    return x;
}
std::vector<int> integers(int id, const char *n) {
    int v;
    nc_check(nc_inq_varid(id, n, &v));
    std::vector<int> x(length(id, v));
    nc_check(nc_get_var_int(id, v, x.data()));
    return x;
}
void put(int id, const char *n, const std::vector<double> &v) {
    int d, x;
    nc_check(nc_def_dim(id, (std::string(n) + "_n").c_str(), v.size(), &d));
    nc_check(nc_def_var(id, n, NC_DOUBLE, 1, &d, &x));
    nc_check(nc_put_var_double(id, x, v.data()));
}
void put(int id, const char *n, const std::string &s) {
    nc_check(nc_put_att_text(id, NC_GLOBAL, n, s.size(), s.data()));
}
static Mesh mesh(int g) {
    Mesh m;
    m.lon = doubles(g, "lon");
    m.lat = doubles(g, "lat");
    m.center_lon = doubles(g, "center_lon");
    m.center_lat = doubles(g, "center_lat");
    m.area = doubles(g, "area");
    m.offsets = integers(g, "offsets");
    m.indices = integers(g, "indices");
    m.mask = integers(g, "mask");
    m.ids = integers(g, "cell_id");
    int v;
    if (nc_inq_varid(g, "parent_id", &v) == NC_NOERR)
        m.parent = integers(g, "parent_id");
    else
        m.parent = m.ids;
    if (m.offsets.size() < 2 || m.offsets.front() != 0 ||
        m.offsets.back() != int(m.indices.size()) || m.lat.size() != m.lon.size())
        throw Error(2, "Malformed mesh extents");
    const auto n = m.size();
    if (m.mask.size() != n || m.ids.size() != n || m.area.size() != n || m.center_lon.size() != n ||
        m.center_lat.size() != n || m.parent.size() != n)
        throw Error(2, "Cell extents differ");
    if (std::set<int>(m.ids.begin(), m.ids.end()).size() != n)
        throw Error(2, "Duplicate cell IDs");
    for (size_t i = 0; i < n; ++i) {
        if (m.offsets[i + 1] - m.offsets[i] < 3 || !std::isfinite(m.area[i]) || m.area[i] <= 0)
            throw Error(2, "Invalid cell/area");
    }
    for (int k : m.indices)
        if (k < 0 || size_t(k) >= m.lon.size())
            throw Error(2, "Connectivity out of range");
    for (size_t k = 0; k < m.lon.size(); ++k)
        if (!std::isfinite(m.lon[k]) || !std::isfinite(m.lat[k]) || std::abs(m.lat[k]) > 90)
            throw Error(2, "Invalid coordinate");
    return m;
}
Case read_case(const std::filesystem::path &p) {
    File f(p);
    if (attr(f.id, "schema_version") != "1")
        throw Error(2, "Unsupported case schema");
    Case c;
    c.id = attr(f.id, "case_id");
    c.geometry = attr(f.id, "geometry");
    c.field = attr(f.id, "field");
    c.norm = attr(f.id, "norm");
    if (c.geometry != "great_circle" && c.geometry != "constant_latitude")
        throw Error(4, "Unsupported reference geometry");
    if (c.norm != "dstarea" && c.norm != "fracarea")
        throw Error(2, "Invalid normalization");
    c.tolerance = std::stod(attr(f.id, "reference_tolerance", "1e-9"));
    c.area_tolerance = std::stod(attr(f.id, "area_tolerance", "1e-10"));
    c.constant_tolerance = std::stod(attr(f.id, "constant_tolerance", "1e-10"));
    c.conservation_tolerance = std::stod(attr(f.id, "conservation_tolerance", "1e-10"));
    c.near_zero_tolerance = std::stod(attr(f.id, "near_zero_tolerance", "1e-12"));
    c.reproduction_tolerance = std::stod(attr(f.id, "reproduction_tolerance", "1e-10"));
    c.smooth_rms_tolerance = std::stod(attr(f.id, "smooth_rms_tolerance", "0.1"));
    c.depth = std::stoi(attr(f.id, "max_depth", "20"));
    c.budget = std::stoi(attr(f.id, "max_subtriangles", "1000000"));
    c.src = mesh(group(f.id, "source"));
    c.dst = mesh(group(f.id, "destination"));
    return c;
}
Reference read_reference(const std::filesystem::path &p) {
    File f(p);
    if (attr(f.id, "status") != "complete")
        throw Error(6, "Invalid reference");
    Reference r;
    r.src = doubles(f.id, "src");
    r.dst = doubles(f.id, "dst");
    r.src_area = doubles(f.id, "src_area");
    r.dst_area = doubles(f.id, "dst_area");
    r.src_covered = doubles(f.id, "src_covered");
    r.dst_covered = doubles(f.id, "dst_covered");
    r.uncertainty = doubles(f.id, "uncertainty");
    r.precision_difference = doubles(f.id, "precision_difference");
    r.area_uncertainty = doubles(f.id, "area_uncertainty");
    r.roundoff_discarded_area = doubles(f.id, "roundoff_discarded_area");
    return r;
}
Output read_output(const std::filesystem::path &p) {
    File f(p);
    Output r;
    r.src = doubles(f.id, "src");
    r.dst = doubles(f.id, "dst");
    r.src_area = doubles(f.id, "src_area");
    r.dst_area = doubles(f.id, "dst_area");
    r.src_frac = doubles(f.id, "src_frac");
    r.dst_frac = doubles(f.id, "dst_frac");
    r.row_sum = doubles(f.id, "row_sum");
    return r;
}
void write_reference(const std::filesystem::path &p, const Reference &r) {
    auto tmp = p.string() + ".tmp";
    {
        File f(tmp, true);
        put(f.id, "status", "complete");
        put(f.id, "schema_version", "1");
        put(f.id, "src", r.src);
        put(f.id, "dst", r.dst);
        put(f.id, "src_area", r.src_area);
        put(f.id, "dst_area", r.dst_area);
        put(f.id, "src_covered", r.src_covered);
        put(f.id, "dst_covered", r.dst_covered);
        put(f.id, "uncertainty", r.uncertainty);
        put(f.id, "source_uncertainty", r.source_uncertainty);
        put(f.id, "precision_difference", r.precision_difference);
        put(f.id, "area_uncertainty", r.area_uncertainty);
        put(f.id, "roundoff_discarded_area", r.roundoff_discarded_area);
        put(f.id, "precision_evidence",
            "double/long-double agreement, adaptive quadrature for Valcke fields, and roundoff "
            "estimate; not an interval proof");
        put(f.id, "extended_precision_digits",
            std::to_string(std::numeric_limits<long double>::digits));
    }
    if (std::filesystem::exists(p))
        throw Error(2, "Reference exists");
    std::filesystem::rename(tmp, p);
}
void write_output(const std::filesystem::path &p, const Output &r) {
    File f(p, true);
    put(f.id, "src", r.src);
    put(f.id, "dst", r.dst);
    put(f.id, "src_area", r.src_area);
    put(f.id, "dst_area", r.dst_area);
    put(f.id, "src_frac", r.src_frac);
    put(f.id, "dst_frac", r.dst_frac);
    put(f.id, "row_sum", r.row_sum);
}
}  // namespace paper
