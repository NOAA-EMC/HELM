# SPDX-License-Identifier: Apache-2.0
"""C96 cubed-sphere (6 tiles, exact supergrid corners) -> global 1deg conservative.

Measures AXIS error vs ESMF/ESMPy on a smooth cos(lat)*cos(lon) field and
constant-field mass conservation, using the current build (CF-edge fixes +
polar-cap C96 fix). Mirrors the geometry of tests_python/test_c96_benchmarks.py
so the numbers are directly comparable to the benchmark tables.
"""

import os
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis import _core  # noqa: E402
from fetch_c96 import fetch_c96_tiles  # noqa: E402

try:
    import esmpy

    ESMPY = True
except ImportError:
    ESMPY = False

C96 = fetch_c96_tiles()

NLAT, NLON = 180, 360
tlats = np.linspace(-89.5, 89.5, NLAT)
tlons = np.linspace(0.5, 359.5, NLON)
lon2d, lat2d = np.meshgrid(tlons, tlats)


def axis_mesh_and_centers():
    node_list, conn_list = [], []
    clon_c, clat_c = [], []
    off = 0
    for f in C96:
        ds = xr.open_dataset(f)
        clon_t = ds["x"].values[0::2, 0::2]
        clat_t = ds["y"].values[0::2, 0::2]
        node_list.append(np.column_stack([clon_t.ravel(), clat_t.ravel()]))
        clon_c.append(ds["x"].values[1::2, 1::2].ravel())
        clat_c.append(ds["y"].values[1::2, 1::2].ravel())
        ni = nj = 96
        nip1 = ni + 1
        i, j = np.meshgrid(np.arange(ni), np.arange(nj))
        bl = (i + j * nip1 + off).ravel()
        br = ((i + 1) + j * nip1 + off).ravel()
        tr = ((i + 1) + (j + 1) * nip1 + off).ravel()
        tl = (i + (j + 1) * nip1 + off).ravel()
        conn_list.append(np.column_stack([bl, br, tr, tl]).ravel())
        off += len(node_list[-1])
    mesh = _core.make_ugrid_mesh(
        np.asfortranarray(np.concatenate(node_list)),
        np.arange(0, sum(len(c) for c in conn_list) + 1, 4, dtype=np.int64),
        np.concatenate(conn_list).astype(np.int64),
    )
    return mesh, np.concatenate(clon_c), np.concatenate(clat_c)


def esmf_regrid(src_lon, src_lat, smooth_src):
    """Welded-node ESMPy mesh -> 1deg grid, conservative. Returns (const, smooth) dst."""
    node_coords, elem_conn = [], []
    node_map, nxt = {}, 1
    for f in C96:
        ds = xr.open_dataset(f)
        clon = ds["x"].values[0::2, 0::2]
        clat = ds["y"].values[0::2, 0::2]
        tile_ids = np.zeros((97, 97), dtype=int)
        for j in range(97):
            for i in range(97):
                key = (round(float(clon[j, i]) % 360, 5), round(float(clat[j, i]), 5))
                if key not in node_map:
                    node_map[key] = nxt
                    node_coords.append([clon[j, i], clat[j, i]])
                    nxt += 1
                tile_ids[j, i] = node_map[key]
        for j in range(96):
            for i in range(96):
                # ESMPy add_elements wants 0-based indices into the node array.
                elem_conn.extend([tile_ids[j, i] - 1, tile_ids[j, i + 1] - 1, tile_ids[j + 1, i + 1] - 1, tile_ids[j + 1, i] - 1])
    nn = len(node_coords)
    ne = len(elem_conn) // 4
    mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=esmpy.CoordSys.SPH_DEG)
    mesh.add_nodes(
        nn, np.arange(1, nn + 1, dtype=np.int32), np.array(node_coords, dtype=np.float64).flatten(), np.zeros(nn, dtype=np.int32)
    )
    mesh.add_elements(
        ne,
        np.arange(1, ne + 1, dtype=np.int32),
        np.full(ne, esmpy.MeshElemType.QUAD, dtype=np.int32),
        np.array(elem_conn, dtype=np.int32),
    )

    grid = esmpy.Grid(
        max_index=np.array([NLON, NLAT]),
        coord_sys=esmpy.CoordSys.SPH_DEG,
        staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
    )
    cl = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
    ca = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
    cl[...] = lon2d.T
    ca[...] = lat2d.T
    rl = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
    ra = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)
    rl[...] = np.meshgrid(np.linspace(0, 360, NLON + 1), np.linspace(-90, 90, NLAT + 1))[0].T
    ra[...] = np.meshgrid(np.linspace(0, 360, NLON + 1), np.linspace(-90, 90, NLAT + 1))[1].T

    out = {}
    t_esmf = 0.0
    for name, sdata in [("const", np.ones(ne)), ("smooth", smooth_src)]:
        fs = esmpy.Field(mesh, meshloc=esmpy.MeshLoc.ELEMENT)
        fd = esmpy.Field(grid, staggerloc=esmpy.StaggerLoc.CENTER)
        fs.data[...] = sdata
        fd.data[...] = 0.0
        t0 = time.perf_counter()
        rg = esmpy.Regrid(fs, fd, regrid_method=esmpy.RegridMethod.CONSERVE, unmapped_action=esmpy.UnmappedAction.IGNORE)
        rg(fs, fd)
        t_esmf += time.perf_counter() - t0
        out[name] = fd.data.T.copy()
    return out, t_esmf


def main():
    mesh, slon, slat = axis_mesh_and_centers()
    src_const = np.ones(slon.size)
    src_smooth = np.cos(np.radians(slat)) * np.cos(np.radians(slon))
    dst_mesh = _core.make_regular_mesh(NLON, NLAT, 0.0, -90.0, 1.0, 1.0)
    cfg = {
        "method": _core.Method.Conservative,
        "periodic": False,
        "line_type": _core.LineType.GreatCircle,
        "norm_type": _core.NormType.FracArea,
        "unmapped": _core.UnmappedAction.Ignore,
    }
    t0 = time.perf_counter()
    w = _core.generate_weights(mesh, dst_mesh, cfg)
    ax_const = np.array(_core.apply_weights(w, src_const)).reshape(NLAT, NLON)
    ax_smooth = np.array(_core.apply_weights(w, src_smooth)).reshape(NLAT, NLON)
    ax_t = time.perf_counter() - t0

    print(f"AXIS C96->1deg  ({ax_t:.3f}s)")
    print(f"  const  min/max/mean : {ax_const.min():.2e} / {ax_const.max():.2e} / {ax_const.mean():.10f}")
    print(f"  const  max|1-x|     : {np.max(np.abs(ax_const - 1.0)):.2e}   (nans={np.isnan(ax_const).sum()})")
    exact = np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))
    print(f"  smooth vs analytic  : max {np.max(np.abs(ax_smooth - exact)):.2e}")

    if ESMPY:
        es, t_esmf = esmf_regrid(slon, slat, src_smooth)
        dc = np.max(np.abs(ax_const - es["const"]))
        ds_ = np.max(np.abs(ax_smooth - es["smooth"]))
        rs = np.sqrt(np.mean((ax_smooth - es["smooth"]) ** 2))
        print(f"\nESMF reference ({t_esmf:.3f}s, weight gen + apply):")
        print(f"  const max|1-x|    : {np.max(np.abs(es['const'] - 1.0)):.2e}  (nans={np.isnan(es['const']).sum()})")
        print(f"  AXIS vs ESMF const: max {dc:.2e}")
        print(f"  AXIS vs ESMF smooth: max {ds_:.2e}  RMS {rs:.2e}")
    else:
        print("\nesmpy not available in this env")


if __name__ == "__main__":
    main()
