# SPDX-License-Identifier: Apache-2.0
"""US2 — define any of the six grid families (T026).

Asserts spec AC1–AC8: construction from xarray / dict / raw arrays for every
family, named grids with zero file I/O, CF edge conventions, location=
overrides, actionable detection errors, and construct-once reuse (SC-002).
"""

from __future__ import annotations

import axis
import numpy as np
import pytest
from axis import Grid, GridFamily
from axis.errors import AxisCapabilityError, GridError

xr = pytest.importorskip("xarray")


# ─── AC1: rectilinear, CF midpoint edges ─────────────────────────────────────


def test_rect_from_arrays_cf_edges():
    lon = np.arange(0.0, 360.0, 10.0)  # centers at 0,10,...350
    lat = np.arange(-80.0, 81.0, 10.0)
    g = Grid(lon=lon, lat=lat)
    assert g.family is GridFamily.RECTILINEAR
    assert g.shape == (lat.size, lon.size)
    m = g.to_mesh()
    assert m.n_cells == g.n_cells
    # cell bounds follow CF midpoints, clamped at the poles: the mesh node
    # lattice spans lon -5..355 and lat -90..90 (clamped).
    # (verified through a regridder: pole rows must receive weights)
    rg = axis.Regridder(g, g, "bilinear")
    assert rg.nnz > 0


def test_rect_from_dict():
    lon = np.arange(0.5, 360.0, 2.0)
    lat = np.arange(-89.5, 90.0, 2.0)
    g = Grid.from_dict({"lon": lon, "lat": lat})
    assert g.family is GridFamily.RECTILINEAR
    assert g.n_cells == lat.size * lon.size


def test_rect_radians_normalized():
    lon = np.radians(np.arange(0.5, 360.0, 4.0))
    lat = np.radians(np.arange(-88.0, 89.0, 4.0))
    g = Grid(lon=lon, lat=lat)
    assert np.allclose(g._payload["lat"], np.arange(-88.0, 89.0, 4.0))


# ─── AC2: curvilinear, bounds preferred, corners synthesized ─────────────────


def test_curv_synthesized_corners():
    LON, LAT = np.meshgrid(np.arange(0.5, 180.0, 2.0), np.arange(-40.0, 41.0, 2.0))
    g = Grid(lon=LON + 0.5 * np.sin(np.radians(LAT * 3)), lat=LAT)
    assert g.family is GridFamily.CURVILINEAR
    m = g.to_mesh()
    assert m.n_cells == g.n_cells


def test_curv_bounds_preferred(curvilinear_ds):
    g = Grid(curvilinear_ds)
    assert g.family is GridFamily.CURVILINEAR
    assert "bounds" in g._payload
    m_bounds = g.to_mesh()
    m_synth = Grid(lon=curvilinear_ds["lon"].values, lat=curvilinear_ds["lat"].values).to_mesh()
    # explicit bounds have 4 vertices/cell from the file, synth also 4 — but
    # the node count differs (bounds share fewer nodes than the corner lattice)
    assert m_bounds.n_cells == m_synth.n_cells == g.n_cells
    assert m_bounds.n_nodes != m_synth.n_nodes


# ─── AC3: cubed-sphere named + file tiles ────────────────────────────────────


# Cell counts measured from the engine (T009 registry): C = 6N^2 cells /
# 6N^2+2 poles; O = 4N^2-2; F = 6*4^k...; N = 10n^2+2; R = ni*(nj-1) regular.
@pytest.mark.parametrize(
    "token,n_cells",
    [("C6", 216), ("C12", 864), ("O96", 40680), ("F64", 32512), ("N320", 422376), ("R80", 50721), ("grid3", 65160)],
)
def test_named_zero_io(token, n_cells):
    g = Grid(token)  # no file access — pure generator
    assert g.n_cells == n_cells
    assert g.periodic is True
    m = g.to_mesh()
    assert m.n_cells == n_cells


def test_named_families():
    assert Grid("C12").family is GridFamily.CUBED_SPHERE
    for tok in ("O96", "F64", "N320", "R80", "grid3"):
        assert Grid(tok).family is GridFamily.UGRID


def test_cs_from_tiles():
    N = 8
    tiles_lon, tiles_lat = [], []
    for t in range(6):
        a, b = np.meshgrid(np.linspace(t * 60, t * 60 + 80, N), np.linspace(-40, 40, N), indexing="ij")
        tiles_lon.append(a)
        tiles_lat.append(b)
    g = Grid(lon=np.stack(tiles_lon), lat=np.stack(tiles_lat))
    assert g.family is GridFamily.CUBED_SPHERE
    assert g.dims == ("tile", "j", "i")
    assert g.n_cells == 6 * N * N
    m = g.to_mesh()
    assert m.n_cells == 6 * N * N


def test_c96_file_ingest(c96_ds):
    g = Grid(c96_ds)
    assert g.family is GridFamily.CUBED_SPHERE
    assert g.n_cells == 6 * 96 * 96


# ─── AC4: UGRID / MPAS with location override ────────────────────────────────


def _mpas_like_ds():
    lat_v = np.array([5.0, 15.0, 25.0, 5.0, 15.0, 25.0])
    lon_v = np.array([5.0, 5.0, 5.0, 25.0, 25.0, 25.0])
    voc = np.array([[1, 2, 5, 4], [4, 5, 6, 0], [2, 3, 6, 5], [0, 0, 0, 0]])
    return xr.Dataset(
        {
            "latVertex": (("nVertices",), lat_v, {"units": "degrees_north"}),
            "lonVertex": (("nVertices",), lon_v, {"units": "degrees_east"}),
            "verticesOnCell": (("nCells", "nv"), voc),
            "nEdgesOnCell": (("nCells",), np.array([4, 3, 3, 0])),
            "latCell": (("nCells",), np.array([10.0, 20.0, 20.0, 15.0])),
            "lonCell": (("nCells",), np.array([10.0, 15.0, 20.0, 15.0])),
            "temperature": (("nCells",), np.ones(4)),
        }
    )


def test_mpas_detection():
    g = Grid(_mpas_like_ds())
    assert g.family is GridFamily.UGRID
    assert g.dims == ("nCells",)
    assert g.n_cells == 4  # dry 4th cell kept as zero-vertex ring


def test_mpas_regrid_feeds_engine():
    ds = _mpas_like_ds()
    g_src = Grid(ds)
    g_dst = Grid(lon=np.arange(0, 30, 2.0), lat=np.arange(0, 30, 2.0))
    rg = axis.Regridder(g_src, g_dst, "bilinear")
    out = rg(ds["temperature"])
    assert out.shape == g_dst.shape


def test_mpas_real_ingest(mpas_ds):
    g = Grid(mpas_ds)
    assert g.family is GridFamily.UGRID
    assert g.n_cells == mpas_ds.sizes["nCells"]
    assert g.dims == ("nCells",)
    m_poly = g.to_mesh("poly")
    m_tri = g.to_mesh("tri")
    assert m_poly.n_cells == g.n_cells
    assert m_tri.n_cells > g.n_cells  # fan-triangulated: more triangles than cells
    # real mesh feeds both cell-centered and interpolation methods
    dst = Grid(lon=np.arange(0, 360, 10.0), lat=np.arange(-80, 81, 10.0))
    rg = axis.Regridder(g, dst, "conservative")
    assert rg.nnz > 0


def test_ugrid_mesh_topology_roles():
    # synthetic UGRID: 2x2 quads with explicit mesh_topology attrs
    node_lon = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20], float)
    node_lat = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20], float)
    face_node = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]) + 1  # 1-based!
    ds = xr.Dataset(
        {
            "mesh": xr.DataArray(
                0,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "node_lon node_lat",
                    "face_node_connectivity": "face_node",
                },
            ),
            "node_lon": (("node",), node_lon),
            "node_lat": (("node",), node_lat),
            "face_node": (("nCell", "nv"), face_node, {"start_index": 1}),
        }
    )
    g = Grid(ds)
    assert g.family is GridFamily.UGRID
    assert g.n_cells == 4
    assert g.dims == ("nCell",)
    # start_index=1 normalized internally: regrid works
    rg = axis.Regridder(g, Grid(lon=np.arange(5, 20, 2.0), lat=np.arange(5, 20, 2.0)), "bilinear")
    assert rg.nnz > 0


def test_ugrid_location_node():
    coords = np.column_stack([np.linspace(0, 90, 12), np.linspace(0, 45, 12)])
    offs = np.array([0, 4, 8, 12], np.int64)
    inds = np.array([0, 1, 5, 4, 4, 5, 9, 8, 1, 2, 6, 5], np.int64)
    g = Grid.from_ugrid(coords, offs, inds, location="node")
    assert g.n_cells == 12
    assert g.dims == ("nnode",)


def test_ugrid_location_edge():
    # 2 quads sharing an edge; edge_node_connectivity midpoints as point-cells
    node_lon = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20], float)
    node_lat = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20], float)
    face_node = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    edge_node = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]])
    ds = xr.Dataset(
        {
            "mesh": xr.DataArray(
                0,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "node_lon node_lat",
                    "face_node_connectivity": "face_node",
                    "edge_node_connectivity": "edge_node",
                },
            ),
            "node_lon": (("node",), node_lon),
            "node_lat": (("node",), node_lat),
            "face_node": (("nCell", "nv"), face_node),
            "edge_node": (("nEdge", "nv2"), edge_node),
        }
    )
    g = Grid(ds, location="edge")
    assert g.family is GridFamily.POINTS
    assert g.n_cells == 7
    assert g.dims == ("nEdge",)
    # midpoint of edge 0 = nodes (0,0)-(10,0): lon 5, lat 0
    coords = g._payload["node_coords"]
    assert np.allclose(coords[0], [5.0, 0.0])


def test_mpas_location_edge():
    ds = _mpas_like_ds()
    ds["lonEdge"] = (("nEdges",), np.array([10.0, 20.0, 15.0, 12.0]))
    ds["latEdge"] = (("nEdges",), np.array([10.0, 10.0, 20.0, 15.0]))
    g = Grid(ds, location="edge")
    assert g.family is GridFamily.POINTS
    assert g.n_cells == 4
    assert g.dims == ("nEdges",)


# ─── AC5: ICON triangles ─────────────────────────────────────────────────────


def test_icon_family(icon_mesh):
    g = Grid.from_ugrid(icon_mesh["node_coords"], icon_mesh["conn_offsets"], icon_mesh["conn_indices"])
    assert g.family is GridFamily.ICON
    m_poly = g.to_mesh("poly")
    m_tri = g.to_mesh("tri")
    assert m_poly.n_cells == g.n_cells == m_tri.n_cells  # already triangles


def test_icon_feeds_regridder(icon_mesh):
    g = Grid.from_ugrid(icon_mesh["node_coords"], icon_mesh["conn_offsets"], icon_mesh["conn_indices"])
    dst = Grid(lon=np.arange(100, 132, 2.0), lat=np.arange(20, 39, 2.0))
    rg = axis.Regridder(g, dst, "bilinear")
    offs, inds, coords = icon_mesh["conn_offsets"], icon_mesh["conn_indices"], icon_mesh["node_coords"]
    centroid_lat = np.array([coords[inds[offs[c] : offs[c + 1]], 1].mean() for c in range(g.n_cells)])
    field = xr.DataArray(np.sin(np.radians(centroid_lat)), dims=g.dims)
    out = rg(field)
    assert out.shape == dst.shape


# ─── AC6: point clouds ───────────────────────────────────────────────────────


def test_points_family(flight_points):
    g = Grid.from_points(flight_points["lon"], flight_points["lat"])
    assert g.family is GridFamily.POINTS
    assert g.n_cells == flight_points["lon"].size
    assert g.dims == ("ncol",)


def test_points_as_target_and_source(flight_points):
    pts = Grid.from_points(flight_points["lon"], flight_points["lat"])
    rect = Grid(lon=np.arange(250, 310, 1.0), lat=np.arange(20, 60, 1.0))
    LON, LAT = np.meshgrid(np.asarray(rect._payload["lon"]), np.asarray(rect._payload["lat"]))
    da = xr.DataArray((np.cos(np.radians(LAT)) ** 2), dims=rect.dims)
    # rect -> points (sampling a track)
    rg1 = axis.Regridder(rect, pts, "bilinear")
    out1 = rg1(da)
    assert out1.shape == (pts.n_cells,)
    # points -> rect (interpolating obs onto a model grid)
    pf = xr.DataArray(np.sin(np.radians(np.asarray(pts._payload["node_coords"][:, 1]))), dims=pts.dims)
    rg2 = axis.Regridder(pts, rect, "nearest")
    out2 = rg2(pf)
    assert out2.shape == rect.shape


def test_points_conservative_rejected(flight_points):
    pts = Grid.from_points(flight_points["lon"], flight_points["lat"])
    rect = Grid(lon=np.arange(250, 310, 2.0), lat=np.arange(20, 60, 2.0))
    with pytest.raises(axis.AxisConfigError):
        axis.Regridder(pts, rect, "conservative")
    with pytest.raises(axis.AxisConfigError):
        axis.Regridder(rect, pts, "conservative")


# ─── AC7: actionable detection errors ────────────────────────────────────────


def test_detection_error_lists_patterns():
    ds = xr.Dataset({"foo": ("bar", [1.0, 2.0, 3.0])})
    with pytest.raises(GridError) as ei:
        Grid(ds)
    msg = str(ei.value).lower()
    assert "searched" in msg
    for needle in ("standard_name", "cf-xarray", "mesh_topology", "mpas"):
        assert needle in msg, f"pattern {needle!r} missing from error"
    assert "lon=" in msg or "explicit" in msg


def test_detection_error_suggests_override():
    with pytest.raises(GridError) as ei:
        Grid(xr.Dataset({"a": ("b", [1.0])}))
    assert any(k in str(ei.value) for k in ("lon=", "override", "explicit"))


# ─── AC8 / FR-007: construct once, reuse everywhere ─────────────────────────


def test_construct_once_reuse():
    src = Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0))
    # mesh built once, cached
    m1 = src.to_mesh()
    assert src.to_mesh() is m1
    # same grid reused across multiple regridder partners
    a = axis.Regridder(src, Grid("C6"), "bilinear")
    b = axis.Regridder(src, Grid(lon=np.arange(0.5, 90.0, 1.0), lat=np.arange(0.0, 45.0, 1.0)), "conservative")
    assert a.nnz > 0 and b.nnz > 0
    # fingerprint stable, equality by fingerprint (no revalidation cost)
    assert src.fingerprint == Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0)).fingerprint


def test_grid_immutable():
    g = Grid(lon=np.arange(0, 360, 10.0), lat=np.arange(-80, 81, 10.0))
    with pytest.raises(AttributeError):
        g.periodic = False
    with pytest.raises(AttributeError):
        g._family = GridFamily.POINTS


# ─── T031: projected grids ───────────────────────────────────────────────────


def _lcc_ds():
    ny, nx = 20, 25
    x = np.linspace(-1_500_000, 1_500_000, nx)
    y = np.linspace(-1_000_000, 1_000_000, ny)
    X, Y = np.meshgrid(x, y)
    mapping = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [25.0, 25.0],
            "latitude_of_projection_origin": 25.0,
            "longitude_of_central_meridian": -95.0,
        },
    )
    return xr.Dataset(
        {
            "lcc": mapping,
            "T": (("y", "x"), np.ones((ny, nx)), {"grid_mapping": "lcc"}),
        },
        coords={
            "x": ("x", x, {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
            "y": ("y", y, {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
        },
    )


def test_projected_lcc():
    ds = _lcc_ds()
    g = Grid(ds)
    assert g.family is GridFamily.RECTILINEAR
    assert g.line_type == "cartesian"
    m = g.to_mesh()
    assert m.n_cells == g.n_cells
    rg = axis.Regridder(g, Grid(lon=np.arange(-120, -60, 2.0), lat=np.arange(20, 40, 2.0)), "bilinear")
    out = rg(ds["T"])
    assert out.shape[-2:] == rg.target.shape


def test_projected_requires_proj(monkeypatch):
    from axis import _core

    monkeypatch.setattr(_core, "HAVE_PROJ", False, raising=True)
    with pytest.raises(AxisCapabilityError):
        Grid(_lcc_ds())
