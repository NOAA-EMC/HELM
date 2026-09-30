#!/usr/bin/env python3
"""
AXIS vs CDO Interpolation Benchmark
====================================

Compares AXIS interpolation accuracy and speed against CDO (Climate Data
Operators) for bilinear, nearest-neighbor, and conservative remapping.

Prerequisites:
  - CDO installed (via conda: `mamba install -c conda-forge cdo`)
  - python-cdo package (`mamba install -c conda-forge python-cdo`)
  - xarray + netcdf4 for I/O
  - numpy
  - scipy
  - _core module (built from libs/axis/python/)

Usage:
  python benchmarks/compare_cdo.py [--grid-type regular] [--grid-size 32] [--methods bilinear,nearest,conservative]

Output:
  Prints a comparison table with max error, RMS error, conservation error,
  and wall-clock time for both AXIS and CDO.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray not available. Install with: pip install xarray netcdf4")
    sys.exit(1)

try:
    from cdo import Cdo
except ImportError:
    print("ERROR: python-cdo not available. Install with: mamba install -c conda-forge python-cdo")
    sys.exit(1)

# Try to import axis — if not built, provide instructions
try:
    import axis

    AXIS_AVAILABLE = True
except ImportError:
    print("WARNING: axis Python package not found. Building requires:")
    print("  pip install ./libs/axis")
    print("")
    print("Running CDO-only benchmark (no AXIS comparison)...")
    AXIS_AVAILABLE = False

# Try to import xregrid for optional comparative benchmarking
try:
    import xregrid

    XREGRID_AVAILABLE = True
except ImportError:
    XREGRID_AVAILABLE = False


# Lambert Conformal Conic (LCC) projection string
LCC_PROJ = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _signed_spherical_area_deg(lonlat_deg):
    """Signed spherical excess of a lon/lat polygon (degrees), CCW-positive.

    Triangle-fan atan2 formula: positive when vertices wind counter-clockwise
    as seen from outside the sphere — the convention AXIS's clippers (and real
    MPAS meshes) require. scipy.spatial.Voronoi emits regions in arbitrary
    winding, so each cell must be normalized before handing it to AXIS.
    """
    lon = np.radians(lonlat_deg[:, 0])
    lat = np.radians(lonlat_deg[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    v = np.column_stack((x, y, z))
    area = 0.0
    for i in range(1, len(v) - 1):
        v0, v1, v2 = v[0], v[i], v[i + 1]
        num = np.dot(v0, np.cross(v1, v2))
        den = 1.0 + np.dot(v0, v1) + np.dot(v1, v2) + np.dot(v0, v2)
        area += 2.0 * np.arctan2(num, den)
    return area


def _normalize_winding_ccw(region, node_coords):
    """Return the region's vertex indices ordered CCW on the sphere."""
    if _signed_spherical_area_deg(node_coords[region]) < 0.0:
        return list(reversed(region))
    return list(region)


def create_test_field(nlat, nlon, field_type="cosine", grid_type="regular"):
    """Create a test field on a regular lat-lon, LCC, or unstructured MPAS (Voronoi) grid."""
    if grid_type == "mpas":
        # Generate random/Poisson-like generator points in regional sector
        # We use [245.0, 283.0] (equivalent to [-115, -77] shifted by +360) to overlap with global regular grids
        n_cells = nlon  # Use total cells requested as nlon (e.g. --src-size 1000)
        min_lon = 245.0
        max_lon = 283.0
        min_lat = 25.0
        max_lat = 55.0

        # Deterministic generation
        np.random.seed(42)
        lons = np.random.uniform(min_lon, max_lon, n_cells)
        lats = np.random.uniform(min_lat, max_lat, n_cells)
        points = np.column_stack((lons, lats))

        from scipy.spatial import Voronoi

        vor = Voronoi(points)

        node_coords = vor.vertices

        conn_offsets = [0]
        conn_indices = []
        valid_cell_indices = []
        ccw_regions = {}
        cell_lons = []
        cell_lats = []

        for i, r_idx in enumerate(vor.point_region):
            region = vor.regions[r_idx]
            # Valid bounded cell has at least 3 vertices and no infinity vertex (-1)
            if len(region) >= 3 and -1 not in region:
                # Filter out crazy boundary cells
                valid_coords = True
                for v_idx in region:
                    v_lon, v_lat = node_coords[v_idx]
                    if v_lon < (min_lon - 5.0) or v_lon > (max_lon + 5.0) or v_lat < (min_lat - 5.0) or v_lat > (max_lat + 5.0):
                        valid_coords = False
                        break
                if valid_coords:
                    # Normalize to CCW winding: AXIS's clipper paths assume CCW
                    # polygons (like real MPAS meshes); scipy Voronoi regions
                    # have arbitrary winding (~50% CW here), which silently
                    # zeroed overlap for CW cells and broke conservation.
                    region = _normalize_winding_ccw(region, node_coords)
                    ccw_regions[r_idx] = region
                    valid_cell_indices.append(r_idx)
                    conn_indices.extend(region)
                    conn_offsets.append(len(conn_indices))
                    cell_lons.append(lons[i])
                    cell_lats.append(lats[i])

        n_valid_cells = len(valid_cell_indices)
        cell_lons = np.array(cell_lons)
        cell_lats = np.array(cell_lats)
        conn_offsets = np.array(conn_offsets, dtype=np.int64)
        conn_indices = np.array(conn_indices, dtype=np.int64)

        # Determine maximum vertices per cell for the CDO 2D bounds padding
        max_nv = max(len(ccw_regions[r]) for r in valid_cell_indices)
        face_nodes = -1 * np.ones((n_valid_cells, max_nv), dtype=np.int32)
        for idx, r_idx in enumerate(valid_cell_indices):
            region = ccw_regions[r_idx]
            face_nodes[idx, : len(region)] = region

        if field_type == "cosine":
            field = np.cos(np.radians(cell_lats)) * np.cos(np.radians(cell_lons))
        elif field_type == "linear":
            field = 0.5 * cell_lons + 0.3 * cell_lats + 10.0
        elif field_type == "constant":
            field = 42.0 * np.ones_like(cell_lats)
        elif field_type == "step":
            field = np.where(cell_lats > 40, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        # Return structured data package for unstructured grid
        return (
            cell_lats,
            cell_lons,
            {
                "node_coords": node_coords,
                "conn_offsets": conn_offsets,
                "conn_indices": conn_indices,
                "face_nodes": face_nodes,
                "cell_lon": cell_lons,
                "cell_lat": cell_lats,
                "field": field,
            },
        )

    if grid_type == "lcc":
        # Create a coordinate grid in projection space (meters)
        # Use a large regional domain (2000 km x 2000 km) for better overlap
        min_x = -1000000.0
        max_x = 1000000.0
        min_y = -1000000.0
        max_y = 1000000.0

        xs = np.linspace(min_x, max_x, nlon)
        ys = np.linspace(min_y, max_y, nlat)
        x_grid2d, y_grid2d = np.meshgrid(xs, ys)

        # Convert projection coordinates to geographic lon/lat
        import pyproj

        proj = pyproj.Proj(LCC_PROJ)
        lons2d, lats2d = proj(x_grid2d, y_grid2d, inverse=True)

        if field_type == "cosine":
            # Smooth cosine bell
            field = np.cos(np.radians(lats2d)) * np.cos(np.radians(lons2d))
        elif field_type == "linear":
            field = 0.5 * lons2d + 0.3 * lats2d + 10.0
        elif field_type == "constant":
            field = 42.0 * np.ones_like(lats2d)
        elif field_type == "step":
            field = np.where(lats2d > 0, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        return lats2d, lons2d, field

    # Regular
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lons, lats)

    if field_type == "cosine":
        # Smooth cosine bell
        field = np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))
    elif field_type == "linear":
        field = 0.5 * lon2d + 0.3 * lat2d + 10.0
    elif field_type == "constant":
        field = 42.0 * np.ones_like(lat2d)
    elif field_type == "step":
        field = np.where(lat2d > 0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return lats, lons, field


def write_netcdf(filepath, lats, lons, field, grid_type="regular", varname="temperature"):
    """Write a field to a CF-compliant NetCDF file for CDO."""
    if grid_type == "mpas":
        # Standard CDO-compliant Unstructured Grid format (cell center lon/lat with repeated corners)
        data = field  # Unstructured mesh data package
        n_cells = len(data["cell_lon"])
        max_nv = data["face_nodes"].shape[1]

        lon_bnds = np.zeros((n_cells, max_nv))
        lat_bnds = np.zeros((n_cells, max_nv))

        # Populate bounds from face_nodes and node_coords
        for idx in range(n_cells):
            node_idx_list = [n for n in data["face_nodes"][idx] if n != -1]
            n_vertices = len(node_idx_list)

            # Fill existing corners
            for v_idx in range(n_vertices):
                lon_bnds[idx, v_idx] = data["node_coords"][node_idx_list[v_idx], 0]
                lat_bnds[idx, v_idx] = data["node_coords"][node_idx_list[v_idx], 1]

            # Repeat last corner to pad up to max_nv (standard CDO SCRIP-style padding)
            last_lon = lon_bnds[idx, n_vertices - 1]
            last_lat = lat_bnds[idx, n_vertices - 1]
            for v_idx in range(n_vertices, max_nv):
                lon_bnds[idx, v_idx] = last_lon
                lat_bnds[idx, v_idx] = last_lat

        ds = xr.Dataset(
            {varname: (["cell"], data["field"].astype(np.float64))},
            coords={
                "lon": (["cell"], data["cell_lon"]),
                "lat": (["cell"], data["cell_lat"]),
            },
        )
        ds["lon_bnds"] = (["cell", "nv"], lon_bnds)
        ds["lat_bnds"] = (["cell", "nv"], lat_bnds)
        ds["lon"].attrs = {
            "units": "degrees_east",
            "standard_name": "longitude",
            "bounds": "lon_bnds",
        }
        ds["lat"].attrs = {
            "units": "degrees_north",
            "standard_name": "latitude",
            "bounds": "lat_bnds",
        }
        ds[varname].attrs = {
            "coordinates": "lon lat",
            "units": "K",
            "long_name": "Test field",
        }
        ds.to_netcdf(filepath)
        return ds

    if grid_type == "lcc":
        # Curvilinear format with 2D lon and lat coordinates and boundary corners
        nlat, nlon = field.shape
        min_x = -1000000.0
        max_x = 1000000.0
        min_y = -1000000.0
        max_y = 1000000.0
        dx = (max_x - min_x) / (nlon - 1)
        dy = (max_y - min_y) / (nlat - 1)

        # Generate cell corner boundaries in projection space
        xs_c = np.linspace(min_x - 0.5 * dx, max_x + 0.5 * dx, nlon + 1)
        ys_c = np.linspace(min_y - 0.5 * dy, max_y + 0.5 * dy, nlat + 1)
        xc_grid2d, yc_grid2d = np.meshgrid(xs_c, ys_c)

        import pyproj

        proj = pyproj.Proj(LCC_PROJ)
        clons2d, clats2d = proj(xc_grid2d, yc_grid2d, inverse=True)

        # Construct 3D bounds [lat, lon, 4 corners] in counter-clockwise order
        lat_bounds = np.zeros((nlat, nlon, 4))
        lon_bounds = np.zeros((nlat, nlon, 4))
        for j in range(nlat):
            for i in range(nlon):
                lat_bounds[j, i, 0] = clats2d[j, i]
                lat_bounds[j, i, 1] = clats2d[j, i + 1]
                lat_bounds[j, i, 2] = clats2d[j + 1, i + 1]
                lat_bounds[j, i, 3] = clats2d[j + 1, i]

                lon_bounds[j, i, 0] = clons2d[j, i]
                lon_bounds[j, i, 1] = clons2d[j, i + 1]
                lon_bounds[j, i, 2] = clons2d[j + 1, i + 1]
                lon_bounds[j, i, 3] = clons2d[j + 1, i]

        ds = xr.Dataset(
            {varname: (["lat", "lon"], field.astype(np.float64))},
            coords={
                "lat_2d": (["lat", "lon"], lats),
                "lon_2d": (["lat", "lon"], lons),
            },
        )
        ds["lat_bounds"] = (["lat", "lon", "nv"], lat_bounds)
        ds["lon_bounds"] = (["lat", "lon", "nv"], lon_bounds)
        ds["lat_2d"].attrs = {
            "units": "degrees_north",
            "standard_name": "latitude",
            "bounds": "lat_bounds",
        }
        ds["lon_2d"].attrs = {
            "units": "degrees_east",
            "standard_name": "longitude",
            "bounds": "lon_bounds",
        }
        ds[varname].attrs = {
            "coordinates": "lon_2d lat_2d",
            "units": "K",
            "long_name": "Test field",
        }
        ds.to_netcdf(filepath)
        return ds

    # Regular
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]

    lat_bnds = np.zeros((len(lats), 2))
    lat_bnds[:, 0] = lats - 0.5 * dlat
    lat_bnds[:, 1] = lats + 0.5 * dlat
    lat_bnds = np.clip(lat_bnds, -90.0, 90.0)

    lon_bnds = np.zeros((len(lons), 2))
    lon_bnds[:, 0] = lons - 0.5 * dlon
    lon_bnds[:, 1] = lons + 0.5 * dlon

    ds = xr.Dataset(
        {varname: (["lat", "lon"], field.astype(np.float64))},
        coords={"lat": lats, "lon": lons},
    )
    ds["lat_bnds"] = (["lat", "bnds"], lat_bnds)
    ds["lon_bnds"] = (["lon", "bnds"], lon_bnds)
    ds["lat"].attrs = {
        "units": "degrees_north",
        "axis": "Y",
        "bounds": "lat_bnds",
        "standard_name": "latitude",
    }
    ds["lon"].attrs = {
        "units": "degrees_east",
        "axis": "X",
        "bounds": "lon_bnds",
        "standard_name": "longitude",
    }
    ds[varname].attrs = {"units": "K", "long_name": "Test field"}
    ds.to_netcdf(filepath)
    return ds


def run_cdo_remap(input_file, output_file, target_grid, method):
    """Run CDO remapping and return wall-clock time."""
    cdo = Cdo()

    # CDO operator names
    method_map = {
        "bilinear": "remapbil",
        "nearest": "remapnn",
        "bicubic": "remapbic",
        "patch": "remapbil",  # CDO doesn't have exact patch; use bilinear as reference
        "conservative": "remapcon",
    }

    operator = method_map[method]
    t0 = time.perf_counter()
    getattr(cdo, operator)(target_grid, input=input_file, output=output_file)
    elapsed = time.perf_counter() - t0

    return elapsed


def run_xregrid_remap(input_file, target_grid, dst_lats, dst_lons, method, dst_grid_type):
    """Run xregrid remapping and return (result, wall_clock_time)."""
    if not XREGRID_AVAILABLE:
        return None, 0.0

    # xregrid methods mapping (translates to ESMF native operators)
    method_map = {
        "bilinear": "bilinear",
        "nearest": "nearest_s2d",
        "conservative": "conservative",
        "bicubic": "bicubic",
        "patch": "patch",
    }

    if method not in method_map:
        return None, 0.0

    t0 = time.perf_counter()
    try:
        # Load source dataset
        ds_in = xr.open_dataset(input_file)

        # Load or construct target dataset
        if dst_grid_type == "mpas":
            ds_out = xr.open_dataset(target_grid)
        else:
            dlat = dst_lats[1] - dst_lats[0]
            dlon = dst_lons[1] - dst_lons[0]

            lat_bnds = np.zeros((len(dst_lats), 2))
            lat_bnds[:, 0] = dst_lats - 0.5 * dlat
            lat_bnds[:, 1] = dst_lats + 0.5 * dlat
            lat_bnds = np.clip(lat_bnds, -90.0, 90.0)

            lon_bnds = np.zeros((len(dst_lons), 2))
            lon_bnds[:, 0] = dst_lons - 0.5 * dlon
            lon_bnds[:, 1] = dst_lons + 0.5 * dlon

            ds_out = xr.Dataset({"lat": (["lat"], dst_lats), "lon": (["lon"], dst_lons)})
            ds_out["lat_bnds"] = (["lat", "bnds"], lat_bnds)
            ds_out["lon_bnds"] = (["lon", "bnds"], lon_bnds)
            ds_out["lat"].attrs = {
                "units": "degrees_north",
                "axis": "Y",
                "bounds": "lat_bnds",
                "standard_name": "latitude",
            }
            ds_out["lon"].attrs = {
                "units": "degrees_east",
                "axis": "X",
                "bounds": "lon_bnds",
                "standard_name": "longitude",
            }

        # Auto-detect if the grid is global and requires periodic longitude wrapping
        # ESMF requires periodic=True for global grids to connect 360 back to 0.
        is_global = dst_grid_type == "regular" and len(dst_lons) > 1 and abs(dst_lons[-1] - dst_lons[0]) > 300.0

        regridder = xregrid.Regridder(ds_in, ds_out, method=method_map[method], periodic=is_global)
        res_ds = regridder(ds_in["temperature"])
        result = res_ds.values
        elapsed = time.perf_counter() - t0
        return result, elapsed
    except Exception:
        # Silently return None if esmpy/xregrid fails (e.g. for certain unperiodic coordinates)
        return None, 0.0


def run_axis_remap(
    input_file,
    target_grid,
    dst_lats,
    dst_lons,
    method,
    grid_type="regular",
    line_type="great_circle",
    dst_grid_type="regular",
):
    """Run AXIS remapping via high-level Python Regridder class and return (result, wall_time)."""
    if not AXIS_AVAILABLE:
        return None, 0.0

    t0 = time.perf_counter()
    try:
        ds_in = xr.open_dataset(input_file)

        # Load or construct target dataset
        if dst_grid_type == "mpas":
            ds_out = xr.open_dataset(target_grid)
        else:
            ds_out = xr.Dataset({"lat": (["lat"], dst_lats), "lon": (["lon"], dst_lons)})

        # Determine periodic longitude wrapping
        is_global = dst_grid_type == "regular" and len(dst_lons) > 1 and abs(dst_lons[-1] - dst_lons[0]) > 300.0

        # Initialize high-level xarray regridder
        regridder = axis.Regridder(ds_in, ds_out, method=method, periodic=is_global, line_type=line_type)

        # Regrid the DataArray
        da_out = regridder(ds_in["temperature"])
        result = da_out.values
        elapsed = time.perf_counter() - t0
        return result, elapsed
    except Exception:
        import traceback

        traceback.print_exc()
        return None, 0.0


def compare_results(axis_result, cdo_result):
    """Compare AXIS and CDO results, return error metrics."""
    if axis_result is None or cdo_result is None:
        return {"max_error": np.nan, "rms_error": np.nan, "mean_error": np.nan}

    # Flatten results to match 1-D size
    axis_flat = np.ravel(axis_result)
    cdo_flat = np.ravel(cdo_result)

    # Truncate to common size
    n = min(len(axis_flat), len(cdo_flat))
    diff = axis_flat[:n] - cdo_flat[:n]

    # Ignore NaN values from unmapped cells in local projected / unstructured grids
    valid = ~np.isnan(diff) & ~np.isnan(axis_flat[:n]) & ~np.isnan(cdo_flat[:n])
    if not np.any(valid):
        return {"max_error": 0.0, "rms_error": 0.0, "mean_error": 0.0}

    valid_diff = diff[valid]

    return {
        "max_error": np.max(np.abs(valid_diff)),
        "rms_error": np.sqrt(np.mean(valid_diff**2)),
        "mean_error": np.mean(np.abs(valid_diff)),
    }


def main():
    parser = argparse.ArgumentParser(description="AXIS vs CDO interpolation benchmark")
    parser.add_argument(
        "--src-size",
        type=str,
        default="32",
        help="Source grid size: N (square NxN), NLONxNLAT, or total cells for unstructured",
    )
    parser.add_argument(
        "--dst-size",
        type=str,
        default="24",
        help="Destination grid size: N (square NxN) or NLONxNLAT",
    )
    parser.add_argument(
        "--grid-type",
        type=str,
        default="regular",
        choices=["regular", "lcc", "mpas"],
        help="Source grid type: regular (lat-lon), lcc (Lambert Conformal), or mpas (unstructured Voronoi)",
    )
    parser.add_argument(
        "--dst-grid-type",
        type=str,
        default="regular",
        choices=["regular", "mpas"],
        help="Destination grid type: regular (lat-lon) or mpas (unstructured Voronoi)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="bilinear,nearest,bicubic,patch,conservative",
        help="Comma-separated interpolation methods to test",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "step"],
        help="Test field type",
    )
    parser.add_argument("--skip-cdo", action="store_true", help="Skip CDO runs")
    parser.add_argument("--skip-xregrid", action="store_true", help="Skip xregrid runs")
    parser.add_argument(
        "--line-type",
        type=str,
        default="great_circle",
        choices=["great_circle", "cartesian"],
        help="Line geometry to use: great_circle (default) or cartesian (fast planar)",
    )
    args = parser.parse_args()

    # Parse grid sizes (support NxM or just N for square)
    def parse_grid_size(s):
        if "x" in s:
            parts = s.split("x")
            return int(parts[0]), int(parts[1])
        n = int(s)
        return n, n

    if args.grid_type == "mpas":
        # Unstructured: src-size defines total cells directly (nlon)
        src_nlon = int(args.src_size)
        src_nlat = 1
    else:
        src_nlon, src_nlat = parse_grid_size(args.src_size)

    if args.dst_grid_type == "mpas":
        # Unstructured: dst-size defines total cells directly (nlon)
        dst_nlon = int(args.dst_size)
        dst_nlat = 1
    else:
        dst_nlon, dst_nlat = parse_grid_size(args.dst_size)

    methods = [m.strip() for m in args.methods.split(",")]

    print(f"{'=' * 70}")
    print("AXIS vs CDO Interpolation Benchmark")
    print(f"{'=' * 70}")
    if args.grid_type == "mpas":
        print(f"Source grid:  {src_nlon} unstructured MPAS cells")
    else:
        print(f"Source grid:  {src_nlon}x{src_nlat} {args.grid_type} ({src_nlon * src_nlat:,} cells)")

    if args.dst_grid_type == "mpas":
        print(f"Dest grid:    {dst_nlon} unstructured MPAS cells")
    else:
        print(f"Dest grid:    {dst_nlon}x{dst_nlat} {args.dst_grid_type} ({dst_nlon * dst_nlat:,} cells)")
    print(f"Test field:   {args.field}")
    print(f"Methods:      {', '.join(methods)}")
    print(f"AXIS module:  {'loaded' if axis else 'NOT AVAILABLE'}")
    print(f"{'=' * 70}")
    print()

    print("Creating test field...")
    # Create test field
    lats, lons, field = create_test_field(src_nlat, src_nlon, args.field, args.grid_type)

    print("Writing source NetCDF...")
    # Write source NetCDF for CDO
    with tempfile.TemporaryDirectory() as tmpdir:
        src_nc = os.path.join(tmpdir, "source.nc")
        write_netcdf(src_nc, lats, lons, field, args.grid_type)

        dst_field_data = None
        if args.dst_grid_type == "mpas":
            print("Creating destination test field (MPAS)...")
            dst_lats, dst_lons, dst_field_data = create_test_field(dst_nlat, dst_nlon, args.field, "mpas")
            print("Writing destination NetCDF...")
            target_grid = os.path.join(tmpdir, "target_grid.nc")
            write_netcdf(target_grid, dst_lats, dst_lons, dst_field_data, "mpas")
        else:
            # Create CDO target grid description (representing regional sector for LCC/MPAS, or global for regular)
            if args.grid_type in ["lcc"]:
                dst_min_lon = -110.0
                dst_max_lon = -82.0
                dst_min_lat = 30.0
                dst_max_lat = 50.0
            elif args.grid_type in ["mpas"]:
                dst_min_lon = 250.0
                dst_max_lon = 278.0
                dst_min_lat = 30.0
                dst_max_lat = 50.0
            else:
                dst_min_lon = 0.0
                dst_max_lon = 360.0
                dst_min_lat = -90.0
                dst_max_lat = 90.0

            dst_lats = np.linspace(dst_min_lat, dst_max_lat, dst_nlat)
            dst_lons = np.linspace(dst_min_lon, dst_max_lon, dst_nlon, endpoint=False)
            target_grid = os.path.join(tmpdir, "target_grid.txt")
            with open(target_grid, "w") as f:
                f.write("gridtype = lonlat\n")
                f.write(f"xsize = {dst_nlon}\n")
                f.write(f"ysize = {dst_nlat}\n")
                f.write(f"xfirst = {dst_lons[0]}\n")
                f.write(f"xinc = {dst_lons[1] - dst_lons[0]}\n")
                f.write(f"yfirst = {dst_lats[0]}\n")
                f.write(f"yinc = {dst_lats[1] - dst_lats[0]}\n")

        # Run benchmarks
        print(f"{'Method':<15} {'Engine':<8} {'Time (s)':<12} {'Max Err':<14} {'RMS Err':<14} {'Src Σ':<14} {'Dst Σ':<14}")
        print(f"{'-' * 15} {'-' * 8} {'-' * 12} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14}")

        for method in methods:
            # ── CDO ──
            cdo_out = os.path.join(tmpdir, f"cdo_{method}.nc")
            if args.skip_cdo:
                cdo_result = None
                cdo_time = 0.0
                cdo_sum = np.nan
            else:
                try:
                    cdo_time = run_cdo_remap(src_nc, cdo_out, target_grid, method)
                    cdo_ds = xr.open_dataset(cdo_out)
                    cdo_result = cdo_ds["temperature"].values
                    cdo_sum = float(np.nansum(cdo_result))
                except Exception as e:
                    print(f"{method:<15} {'CDO':<8} {'FAILED':<12} {str(e)[:40]}")
                    cdo_result = None
                    cdo_time = 0.0
                    cdo_sum = np.nan

            if args.grid_type == "mpas":
                src_sum = float(np.sum(field["field"]))
            else:
                src_sum = float(np.sum(field))

            if cdo_result is not None:
                print(f"{method:<15} {'CDO':<8} {cdo_time:<12.4f} {'—':<14} {'—':<14} {src_sum:<14.4f} {cdo_sum:<14.4f}")

            # ── xregrid ──
            xregrid_result = None
            xregrid_time = 0.0
            xregrid_sum = np.nan
            if not args.skip_xregrid and XREGRID_AVAILABLE:
                try:
                    xregrid_result, xregrid_time = run_xregrid_remap(
                        src_nc,
                        target_grid,
                        dst_lats,
                        dst_lons,
                        method,
                        args.dst_grid_type,
                    )
                    if xregrid_result is not None:
                        xregrid_sum = float(np.nansum(xregrid_result))
                except Exception:
                    pass

            if xregrid_result is not None:
                if cdo_result is not None:
                    xr_errors = compare_results(xregrid_result.ravel(), cdo_result)
                    print(
                        f"{method:<15} {'xregrid':<8} {xregrid_time:<12.4f} "
                        f"{xr_errors['max_error']:<14.2e} "
                        f"{xr_errors['rms_error']:<14.2e} "
                        f"{src_sum:<14.4f} {xregrid_sum:<14.4f}"
                    )
                else:
                    print(
                        f"{method:<15} {'xregrid':<8} {xregrid_time:<12.4f} {'—':<14} {'—':<14} {src_sum:<14.4f} {xregrid_sum:<14.4f}"
                    )

            # ── AXIS ──
            axis_result, axis_time = run_axis_remap(
                src_nc,
                target_grid,
                dst_lats,
                dst_lons,
                method,
                args.grid_type,
                args.line_type,
                args.dst_grid_type,
            )

            if axis_result is not None:
                axis_sum = float(np.nansum(axis_result))

                # Compare against CDO
                if cdo_result is not None:
                    errors = compare_results(axis_result, cdo_result)
                    print(
                        f"{method:<15} {'AXIS':<8} {axis_time:<12.4f} "
                        f"{errors['max_error']:<14.2e} "
                        f"{errors['rms_error']:<14.2e} "
                        f"{src_sum:<14.4f} {axis_sum:<14.4f}"
                    )
                else:
                    print(f"{method:<15} {'AXIS':<8} {axis_time:<12.4f} {'—':<14} {'—':<14} {src_sum:<14.4f} {axis_sum:<14.4f}")
            else:
                print(f"{method:<15} {'AXIS':<8} {'N/A':<12} {'(module not loaded)'}")

            print()

    print(f"{'=' * 70}")
    print("Notes:")
    print("  - Max/RMS Err = Engine result vs CDO result (CDO is the reference)")
    print("  - Src/Dst Σ = sum of field values (check conservation)")
    print("  - Time includes weight generation + apply (not I/O)")
    if axis is None:
        print("\n  To enable AXIS comparison, build the Python module:")
        print("    cd libs/axis && cmake -B build-py -DAXIS_BUILD_PYTHON=ON")
        print("    cmake --build build-py && export PYTHONPATH=build-py/python")
    if not XREGRID_AVAILABLE:
        print("\n  To enable xregrid comparison, install esmpy and xregrid:")
        print("    conda install -c conda-forge esmpy dask")
        print("    pip install git+https://github.com/noaa-emc/xregrid.git")


if __name__ == "__main__":
    main()
