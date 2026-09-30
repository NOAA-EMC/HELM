# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr

# Ensure dask and scipy are optionally importable for advanced tests
try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import scipy.sparse  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import axis
from axis import _core

PROJ_AVAILABLE = getattr(_core, "HAVE_PROJ", False)


@pytest.fixture
def sample_grids():
    """Create simple source (10x5) and destination (20x10) xarray regular grids."""
    src_lons = np.linspace(0.0, 360.0, 10, endpoint=False)
    src_lats = np.linspace(-90.0, 90.0, 5)

    dst_lons = np.linspace(0.0, 360.0, 20, endpoint=False)
    dst_lats = np.linspace(-90.0, 90.0, 10)

    ds_in = xr.Dataset(
        {
            "lon": (["lon"], src_lons),
            "lat": (["lat"], src_lats),
        }
    )

    ds_out = xr.Dataset(
        {
            "lon": (["lon"], dst_lons),
            "lat": (["lat"], dst_lats),
        }
    )

    return ds_in, ds_out


def test_regridder_eager_numpy(sample_grids):
    """Verify that axis.Regridder works on eager NumPy-backed DataArrays."""
    ds_in, ds_out = sample_grids

    # Create simple cosine field: cos(lat) * cos(lon)
    lon_grid, lat_grid = np.meshgrid(ds_in["lon"], ds_in["lat"])
    field = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))

    da_in = xr.DataArray(
        field,
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
        name="temperature",
    )

    # Initialize and call regridder
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out = regridder(da_in)

    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (10, 20)
    assert da_out.name == "temperature"
    assert "lat" in da_out.coords
    assert "lon" in da_out.coords

    # Verify constant field is preserved (partition of unity)
    const_da = xr.DataArray(np.full(da_in.shape, 5.0), coords=da_in.coords, dims=da_in.dims)
    const_out = regridder(const_da)
    np.testing.assert_allclose(const_out.values, 5.0, rtol=1e-12)


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask array not available")
def test_regridder_lazy_dask(sample_grids):
    """Verify that axis.Regridder works on lazy Dask-backed DataArrays with batch dimensions."""
    ds_in, ds_out = sample_grids

    # Create a lazy Dask-backed DataArray with 2 time steps
    lazy_data = da.ones((2, 5, 10), chunks=(1, 5, 10))
    da_in = xr.DataArray(
        lazy_data,
        coords={"time": [0, 1], "lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["time", "lat", "lon"],
        name="precipitation",
    )

    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out = regridder(da_in)

    # Verify that the output array is ALSO a lazy Dask array (has not been eagerly computed)
    assert hasattr(da_out.data, "dask")
    assert da_out.dims == ("time", "lat", "lon")
    assert da_out.shape == (2, 10, 20)

    # Eagerly compute the lazy array and verify values
    computed = da_out.compute()
    assert isinstance(computed.data, np.ndarray)
    np.testing.assert_allclose(computed.values, 1.0, rtol=1e-12)


def test_regridder_accessor(sample_grids):
    """Verify that xarray datasets and dataarrays can be regridded via the .axis accessor."""
    ds_in, ds_out = sample_grids

    da_in = xr.DataArray(
        np.ones((5, 10)),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
        name="temperature",
    )

    # 1. Test DataArray accessor: da.axis.to(ds_out)
    da_out = da_in.axis.to(ds_out, method="bilinear")
    assert da_out.shape == (10, 20)
    assert da_out.name == "temperature"
    np.testing.assert_allclose(da_out.values, 1.0, rtol=1e-12)

    # 2. Test Dataset accessor: ds.axis.to(ds_out)
    ds_in_vars = xr.Dataset({"temperature": da_in})
    ds_out_vars = ds_in_vars.axis.to(ds_out, method="bilinear")
    assert "temperature" in ds_out_vars.data_vars
    assert ds_out_vars["temperature"].shape == (10, 20)
    np.testing.assert_allclose(ds_out_vars["temperature"].values, 1.0, rtol=1e-12)


def test_regridder_methods(sample_grids):
    """Verify that the regridder supports all core interpolation methods."""
    ds_in, ds_out = sample_grids

    da_in = xr.DataArray(
        np.ones((5, 10)),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
    )

    methods = ["nearest", "bicubic", "conservative"]
    for method in methods:
        regridder = axis.Regridder(ds_in, ds_out, method=method)
        da_out = regridder(da_in)
        assert da_out.shape == (10, 20)
        # Verify that non-boundary cells are exactly 1.0 (unmapped polar cells may be 0.0)
        np.testing.assert_allclose(da_out.values[1:-1, :], 1.0, rtol=1e-12)


def test_regridder_skipna(sample_grids):
    """Verify NaN-aware re-normalization with skipna=True."""
    ds_in, ds_out = sample_grids

    # Create input field with a NaN cell
    field = np.ones((5, 10))
    field[2, 3] = np.nan  # Insert NaN in the center

    da_in = xr.DataArray(field, coords={"lat": ds_in["lat"], "lon": ds_in["lon"]}, dims=["lat", "lon"])

    # Eager path: bilinear with skipna=True
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear", skipna=True)
    da_out = regridder(da_in)

    # Ensure that cells overlapping the NaN are re-normalized and not masked as NaN
    # (Only cells with 100% NaN contribution will be NaN if na_thres=1.0)
    assert not np.all(np.isnan(da_out.values))

    # Verify that points far from NaN are still exactly 1.0
    np.testing.assert_allclose(da_out.values[0, 0], 1.0, rtol=1e-12)


def test_regridder_unstructured_mpas():
    """Verify that unstructured MPAS-style Voronoi datasets parse and regrid correctly."""
    # Define simple MPAS mesh
    n_cells = 4
    n_vertices = 6  # noqa: F841

    # 4 cell centroids
    cell_lon = np.array([10.0, 20.0, 10.0, 20.0])
    cell_lat = np.array([10.0, 10.0, 20.0, 20.0])

    # 6 vertices
    lat_vertex = np.array([5.0, 15.0, 25.0, 5.0, 15.0, 25.0])
    lon_vertex = np.array([5.0, 5.0, 5.0, 25.0, 25.0, 25.0])

    # 4 triangular/quad cells
    vertices_on_cell = np.array([[1, 2, 5, 4], [4, 5, 6, 0], [2, 3, 6, 5], [0, 0, 0, 0]])

    ds_in = xr.Dataset(
        {
            "latVertex": (["nVertices"], lat_vertex),
            "lonVertex": (["nVertices"], lon_vertex),
            "verticesOnCell": (["nCells", "maxVertices"], vertices_on_cell),
            "latCell": (["nCells"], cell_lat),
            "lonCell": (["nCells"], cell_lon),
            "nEdgesOnCell": (["nCells"], [4, 3, 3, 0]),
            "temperature": (["nCells"], np.ones(n_cells)),
        }
    )

    # Destination regular grid
    ds_out = xr.Dataset(
        {
            "lon": (["lon"], [15.0]),
            "lat": (["lat"], [15.0]),
        }
    )

    regridder = axis.Regridder(ds_in, ds_out, method="nearest")
    da_out = regridder(ds_in["temperature"])

    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (1, 1)
    np.testing.assert_allclose(da_out.values, 1.0, rtol=1e-12)


def test_regridder_mpas_hex():
    """Verify that MPAS meshes with hexagons and pentagons regrid correctly with cell-centered remapping."""
    n_cells = 2  # noqa: F841
    n_vertices = 7  # noqa: F841

    # 7 vertices
    lat_vertex = np.array([10.0, 10.0, 15.0, 20.0, 20.0, 15.0, 12.0])
    lon_vertex = np.array([10.0, 20.0, 25.0, 20.0, 10.0, 5.0, 15.0])

    # 2 cells (hexagon and pentagon)
    vertices_on_cell = np.array(
        [
            [1, 2, 3, 4, 5, 6],  # Hexagon (6 edges)
            [1, 2, 4, 7, 6, 0],  # Pentagon (5 edges, padded with 0)
        ]
    )

    ds_in = xr.Dataset(
        {
            "latVertex": (["nVertices"], lat_vertex),
            "lonVertex": (["nVertices"], lon_vertex),
            "verticesOnCell": (["nCells", "maxVertices"], vertices_on_cell),
            "latCell": (["nCells"], [15.0, 14.0]),
            "lonCell": (["nCells"], [15.0, 13.0]),
            "nEdgesOnCell": (["nCells"], [6, 5]),
            "temperature": (["nCells"], np.array([25.0, 30.0])),
        }
    )

    # Destination regular grid
    ds_out = xr.Dataset(
        {
            "lon": (["lon"], [15.0]),
            "lat": (["lat"], [15.0]),
        }
    )

    regridder = axis.Regridder(ds_in, ds_out, method="nearest")
    da_out = regridder(ds_in["temperature"])

    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (1, 1)
    np.testing.assert_allclose(da_out.values, 25.0, rtol=1e-12)


def test_regridder_curvilinear_2d():
    """Verify that general 2D curvilinear grids (without LCC) parse and regrid correctly."""
    # Build 2D curvilinear coordinates
    lons = np.array([[10.0, 20.0], [10.0, 20.0]])
    lats = np.array([[10.0, 10.0], [20.0, 20.0]])

    ds_in = xr.Dataset(
        {
            "lon": (["y", "x"], lons),
            "lat": (["y", "x"], lats),
            "temperature": (["y", "x"], np.array([[10.0, 20.0], [30.0, 40.0]])),
        }
    )

    # Destination regular grid
    ds_out = xr.Dataset(
        {
            "lon": (["lon"], [11.0]),
            "lat": (["lat"], [11.0]),
        }
    )

    regridder = axis.Regridder(ds_in, ds_out, method="nearest")
    da_out = regridder(ds_in["temperature"])

    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (1, 1)
    np.testing.assert_allclose(da_out.values, 10.0, rtol=1e-12)


def test_regridder_cubed_sphere_3d():
    """Verify that 3D cubed-sphere grids (ntiles, ny, nx) parse and regrid correctly."""
    # Build 3D coordinates representing 6 tiles of size 2x2
    ntiles = 6
    ny, nx = 2, 2
    lons = np.zeros((ntiles, ny, nx))
    lats = np.zeros((ntiles, ny, nx))

    for t in range(ntiles):
        lons[t] = np.array([[10.0, 20.0], [10.0, 20.0]]) + t * 40.0
        lats[t] = np.array([[10.0, 10.0], [20.0, 20.0]])

    ds_in = xr.Dataset(
        {
            "lon": (["tile", "y", "x"], lons),
            "lat": (["tile", "y", "x"], lats),
            "temperature": (["tile", "y", "x"], np.ones((ntiles, ny, nx)) * 42.0),
        }
    )

    # Destination regular grid
    ds_out = xr.Dataset(
        {
            "lon": (["lon"], [55.0]),
            "lat": (["lat"], [15.0]),
        }
    )

    regridder = axis.Regridder(ds_in, ds_out, method="nearest")
    da_out = regridder(ds_in["temperature"])

    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (1, 1)
    np.testing.assert_allclose(da_out.values, 42.0, rtol=1e-12)


@pytest.mark.skipif(not PROJ_AVAILABLE, reason="_core built without PROJ (AXIS_ENABLE_PROJ=OFF)")
def test_regridder_projected_lcc():
    """Verify that regional Lambert Conformal projected datasets parse and regrid correctly."""
    # Build 2D curvilinear coordinates with standard grid_mapping metadata
    lons = np.array([[260.0, 261.0], [260.0, 261.0]])
    lats = np.array([[30.0, 30.0], [31.0, 31.0]])

    ds_in = xr.Dataset(
        {
            "lon": (["y", "x"], lons),
            "lat": (["y", "x"], lats),
            "temperature": (["y", "x"], np.ones((2, 2))),
            "lambert_conformal_conic": ([], 0),
        }
    )
    ds_in["lambert_conformal_conic"].attrs = {"grid_mapping_name": "lambert_conformal_conic"}
    ds_in["temperature"].attrs = {"grid_mapping": "lambert_conformal_conic"}

    ds_out = xr.Dataset(
        {
            "lon": (["lon"], [260.5]),
            "lat": (["lat"], [30.5]),
        }
    )

    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out = regridder(ds_in["temperature"])

    assert da_out.shape == (1, 1)
    np.testing.assert_allclose(da_out.values, 1.0, rtol=1e-12)


def test_regridder_sutherland_approximation(sample_grids):
    """Verify that the regridder supports Sutherland-Hodgman flat planar clipping via line_type='cartesian'."""
    ds_in, ds_out = sample_grids
    da_in = xr.DataArray(
        np.ones((5, 10)),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
    )

    # Initialize conservative regridder with Cartesian flat-plane clipping
    regridder = axis.Regridder(ds_in, ds_out, method="conservative", line_type="cartesian")
    da_out = regridder(da_in)

    assert da_out.shape == (10, 20)
    # Verify that non-boundary cells are exactly 1.0 (unmapped polar cells may be 0.0)
    np.testing.assert_allclose(da_out.values[1:-1, :], 1.0, rtol=1e-12)


def test_regridder_errors(sample_grids):
    """Verify that the regridder rejects invalid inputs with actionable errors."""
    ds_in, ds_out = sample_grids

    # 1. Invalid method name -> AxisConfigError naming the bad input (SC-006)
    with pytest.raises(axis.AxisConfigError, match="invalid_method_name"):
        axis.Regridder(ds_in, ds_out, method="invalid_method_name")

    # 2. Passing invalid types to __call__
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    with pytest.raises(TypeError):
        regridder("not_an_xarray_object")

    # 3. Invalid line_type name
    with pytest.raises(axis.AxisConfigError, match="invalid_line_type"):
        axis.Regridder(ds_in, ds_out, method="conservative", line_type="invalid_line_type")

    # 4. Missing latitude coordinates in dataset -> GridError
    bad_ds = xr.Dataset({"lon": ds_in["lon"]})  # No lat coordinate!
    with pytest.raises(axis.GridError):
        axis.Regridder(bad_ds, ds_out, method="bilinear")


def test_regridder_save_and_reuse_weights(sample_grids, tmp_path):
    """Weights serialize to a native file and reload with identical results."""
    ds_in, ds_out = sample_grids
    da_in = xr.DataArray(
        np.ones((len(ds_in["lat"]), len(ds_in["lon"]))),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
    )

    regridder_gen = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out_gen = regridder_gen(da_in)

    weights_path = tmp_path / "weights.axisw"
    regridder_gen.save_weights(weights_path)
    assert weights_path.exists()

    source = axis.Grid(lon=ds_in["lon"].values, lat=ds_in["lat"].values)
    target = axis.Grid(lon=ds_out["lon"].values, lat=ds_out["lat"].values)
    regridder_loaded = axis.Regridder.load_weights(weights_path, source=source, target=target)
    da_out_loaded = regridder_loaded(da_in)

    np.testing.assert_allclose(da_out_loaded.values, da_out_gen.values, rtol=1e-15, atol=1e-15)


def test_regridder_esmf_weights_roundtrip(sample_grids, tmp_path):
    """Weights write as ESMF NetCDF and reload with identical results."""
    ds_in, ds_out = sample_grids
    da_in = xr.DataArray(
        np.ones((len(ds_in["lat"]), len(ds_in["lon"]))),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
    )

    regridder_gen = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out_gen = regridder_gen(da_in)

    esmf_path = tmp_path / "esmf_weights.nc"
    regridder_gen.to_esmf(esmf_path)
    assert esmf_path.exists()

    source = axis.Grid(lon=ds_in["lon"].values, lat=ds_in["lat"].values)
    target = axis.Grid(lon=ds_out["lon"].values, lat=ds_out["lat"].values)
    regridder_loaded = axis.Regridder.from_esmf(esmf_path, source=source, target=target)
    da_out_loaded = regridder_loaded(da_in)

    np.testing.assert_allclose(da_out_loaded.values, da_out_gen.values, rtol=1e-15, atol=1e-15)


def test_regridder_categorical():
    """Verify that categorical/fractional remapping works perfectly."""
    # Source 2x2 grid
    lons_in = [10.0, 20.0]
    lats_in = [10.0, 20.0]

    # 2x2 cells with category values: 1 (forest), 2 (water), 3 (urban)
    categories = np.array([[1, 2], [1, 3]])

    ds_in = xr.Dataset(
        {
            "land_use": (["lat", "lon"], categories),
            "lat": (["lat"], lats_in),
            "lon": (["lon"], lons_in),
        }
    )

    # Coarse 1x1 destination grid spanning the entire source
    ds_out = xr.Dataset({"lat": (["lat"], [15.0]), "lon": (["lon"], [15.0])})

    regridder = axis.Regridder(ds_in, ds_out, method="nearest")

    # 1. Test auto-discovery of categories
    ds_fraction = regridder.regrid_categorical(ds_in["land_use"])
    assert "fraction_1" in ds_fraction
    assert "fraction_2" in ds_fraction
    assert "fraction_3" in ds_fraction

    # 2. Test dictionary mapping
    mapping = {1: "forest", 2: "water", 3: "urban"}
    ds_mapped = regridder.regrid_categorical(ds_in["land_use"], categories=mapping)
    assert "fraction_forest" in ds_mapped
    assert "fraction_water" in ds_mapped
    assert "fraction_urban" in ds_mapped

    # Verify exact area/centroid fractions
    for var in ds_mapped.data_vars:
        val = ds_mapped[var].values[0, 0]
        assert val == 0.0 or val == 1.0
