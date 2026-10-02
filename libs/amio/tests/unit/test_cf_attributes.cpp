// test_cf_attributes.cpp
//
// Integration test for CF/UGRID convention compliance and per-variable
// attribute support.  Drives the REAL Zarr_Driver in NCZarr fallback
// mode (serial netCDF-c, no MPI) end-to-end:
//
//   parse manifest (eckit YAML) -> open_write -> write var -> close
//
// then re-opens the produced store with netCDF-c and asserts:
//
//   * the global `Conventions` attribute is "CF-1.10 UGRID-1.0"
//     (auto-upgraded because a UGRID mesh role is declared)
//   * a normal variable carries its CF attributes (units,
//     standard_name, long_name, _FillValue with the variable's type)
//   * a mesh variable carries its UGRID attributes (cf_role,
//     topology_dimension)
//   * an extra global attribute (title) is written
//
// This exercises the shared attribute model (var_attributes.cpp) and
// the NCZarr write path.  The NetCDF-4 driver shares the same model
// and helper, differing only in nc_create_par vs nc_create.
//
// Validates: CF-1.10 / UGRID-1.0 convention compliance + variable
// attributes for the netCDF-c-backed drivers.

#include <eckit/config/YAMLConfiguration.h>
#include <netcdf.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "drivers/common/var_attributes.hpp"
#include "drivers/zarr/zarr_driver.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

using amio::detail::StagingBuffer;
using amio::detail::VarMeta;
using amio::detail::Zarr_Driver;

namespace {

int g_passed = 0;
int g_failed = 0;

void report_failure(const char *expr, const char *file, int line, const std::string &ctx) {
    std::fprintf(stderr, "FAIL %s:%d: %s   (%s)\n", file, line, expr, ctx.c_str());
    ++g_failed;
}

#define EXPECT_TRUE(cond, ctx)                                \
    do {                                                      \
        if (!(cond)) {                                        \
            report_failure(#cond, __FILE__, __LINE__, (ctx)); \
        } else {                                              \
            ++g_passed;                                       \
        }                                                     \
    } while (0)

// Read a text attribute from a netCDF variable/group; "" if absent.
std::string read_text_att(int ncid, int varid, const char *name) {
    size_t len = 0;
    if (nc_inq_attlen(ncid, varid, name, &len) != NC_NOERR) {
        return "";
    }
    std::string out(len, '\0');
    if (nc_get_att_text(ncid, varid, name, out.data()) != NC_NOERR) {
        return "";
    }
    // Trim any trailing NUL.
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

}  // namespace

int main() {
    const std::string store_path = "/tmp/amio_test_cf_attrs.zarr";

    // Manifest declaring CF attributes on a data variable plus a UGRID
    // mesh-topology variable.  Conventions is intentionally NOT set so
    // the parser auto-upgrades to "CF-1.10 UGRID-1.0".
    const std::string yaml = "uri: " + store_path +
                             "\n"
                             "chunk_shape: [4, 5]\n"
                             "shard_shape: [4, 5]\n"
                             "array_shape: [4, 5]\n"
                             "codec: blosc\n"
                             "global_attributes:\n"
                             "  title: AMIO CF/UGRID test\n"
                             "variables:\n"
                             "  t2m:\n"
                             "    attributes:\n"
                             "      units: K\n"
                             "      long_name: 2 metre temperature\n"
                             "      standard_name: air_temperature\n"
                             "      _FillValue: -9999.0\n"
                             "  mesh:\n"
                             "    attributes:\n"
                             "      cf_role: mesh_topology\n"
                             "      topology_dimension: 2\n";

    eckit::YAMLConfiguration cfg{yaml};

    // --- Drive the driver end-to-end ---
    try {
        Zarr_Driver driver;
        driver.open_write(cfg);

        // Write the t2m data variable (4x5 floats).
        StagingBuffer buf{};
        float data[20];
        for (int i = 0; i < 20; ++i) data[i] = static_cast<float>(i);
        buf.data = reinterpret_cast<std::byte *>(data);
        buf.capacity_bytes = sizeof(data);
        buf.used_bytes = sizeof(data);

        VarMeta meta{};
        meta.name = "t2m";
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = 4;
        meta.shape.extents[1] = 5;
        driver.write(buf, meta);

        // Write the mesh variable (a small int field carrying UGRID
        // role attributes; content is irrelevant to the test).
        int mesh_data[20];
        for (int i = 0; i < 20; ++i) mesh_data[i] = i;
        StagingBuffer mbuf{};
        mbuf.data = reinterpret_cast<std::byte *>(mesh_data);
        mbuf.capacity_bytes = sizeof(mesh_data);
        mbuf.used_bytes = sizeof(mesh_data);

        VarMeta mmeta{};
        mmeta.name = "mesh";
        mmeta.dtype = AMIO_DTYPE_I32;
        mmeta.shape.rank = 2;
        mmeta.shape.extents[0] = 4;
        mmeta.shape.extents[1] = 5;
        driver.write(mbuf, mmeta);

        driver.flush();
        driver.close();
        ++g_passed;
    } catch (const std::exception &e) {
        report_failure("driver end-to-end", __FILE__, __LINE__, e.what());
        std::fprintf(stdout, "test_cf_attributes: passed=%d failed=%d\n", g_passed, g_failed);
        return 1;
    }

    // --- Re-open the produced NCZarr store and verify attributes ---
    std::string uri = "file://" + store_path + "#mode=nczarr,file";
    int ncid = -1;
    int status = nc_open(uri.c_str(), NC_NOWRITE, &ncid);
    EXPECT_TRUE(status == NC_NOERR, std::string("nc_open: ") + nc_strerror(status));

    if (status == NC_NOERR) {
        // Global Conventions auto-upgraded to CF + UGRID.
        EXPECT_TRUE(read_text_att(ncid, NC_GLOBAL, "Conventions") == "CF-1.10 UGRID-1.0", "global Conventions should be 'CF-1.10 UGRID-1.0'");

        // Extra global attribute.
        EXPECT_TRUE(read_text_att(ncid, NC_GLOBAL, "title") == "AMIO CF/UGRID test", "global title attribute");

        // t2m CF attributes.
        int t2m_id = -1;
        if (nc_inq_varid(ncid, "t2m", &t2m_id) == NC_NOERR) {
            EXPECT_TRUE(read_text_att(ncid, t2m_id, "units") == "K", "t2m:units");
            EXPECT_TRUE(read_text_att(ncid, t2m_id, "standard_name") == "air_temperature", "t2m:standard_name");
            EXPECT_TRUE(read_text_att(ncid, t2m_id, "long_name") == "2 metre temperature", "t2m:long_name");

            // _FillValue should be a float (variable's type), value -9999.
            nc_type att_type;
            size_t att_len = 0;
            int rc1 = nc_inq_att(ncid, t2m_id, "_FillValue", &att_type, &att_len);
            EXPECT_TRUE(rc1 == NC_NOERR && att_type == NC_FLOAT && att_len == 1, "t2m:_FillValue is a scalar float");
            float fill = 0.0f;
            if (rc1 == NC_NOERR) {
                nc_get_att_float(ncid, t2m_id, "_FillValue", &fill);
                EXPECT_TRUE(fill == -9999.0f, "t2m:_FillValue value == -9999");
            }
        } else {
            report_failure("nc_inq_varid t2m", __FILE__, __LINE__, "t2m not found");
        }

        // mesh UGRID attributes.
        int mesh_id = -1;
        if (nc_inq_varid(ncid, "mesh", &mesh_id) == NC_NOERR) {
            EXPECT_TRUE(read_text_att(ncid, mesh_id, "cf_role") == "mesh_topology", "mesh:cf_role");
            // topology_dimension is an integer attribute.
            long long topo = 0;
            int rc2 = nc_get_att_longlong(ncid, mesh_id, "topology_dimension", &topo);
            EXPECT_TRUE(rc2 == NC_NOERR && topo == 2, "mesh:topology_dimension == 2 (integer)");
        } else {
            report_failure("nc_inq_varid mesh", __FILE__, __LINE__, "mesh not found");
        }

        nc_close(ncid);
    }

    std::fprintf(stdout, "test_cf_attributes: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
