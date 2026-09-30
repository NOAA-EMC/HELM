import os
import sys

import esmpy
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_c96 import fetch_c96_tiles  # noqa: E402


def build_esmf_c96_mesh():
    """Build a unified ESMPy Mesh for all 6 tiles of C96 using exact supergrid corners."""
    tile_files = fetch_c96_tiles()
    ds_list = [xr.open_dataset(f) for f in tile_files]

    # Total cells: 6 * 96 * 96 = 55296
    # For each tile t in 0..5, corner lons/lats are at x[0::2, 0::2] and y[0::2, 0::2] (97, 97)
    # Cell centers are at x[1::2, 1::2] and y[1::2, 1::2] (96, 96)

    node_coords = []
    element_conn = []
    element_types = []
    element_ids = []

    node_id_map = {}
    next_node_id = 1

    for t, ds in enumerate(ds_list):
        clon = ds["x"].values[0::2, 0::2]
        clat = ds["y"].values[0::2, 0::2]
        ny, nx = 96, 96

        # We can create nodes for tile t
        tile_node_ids = np.zeros((97, 97), dtype=int)
        for j in range(97):
            for i in range(97):
                # Identify rounded coords to weld shared nodes across tile boundaries
                key = (round(float(clon[j, i]) % 360, 5), round(float(clat[j, i]), 5))
                if key in node_id_map:
                    tile_node_ids[j, i] = node_id_map[key]
                else:
                    node_id_map[key] = next_node_id
                    tile_node_ids[j, i] = next_node_id
                    node_coords.append([clon[j, i], clat[j, i]])
                    next_node_id += 1

        for j in range(ny):
            for i in range(nx):
                elem_id = t * ny * nx + j * nx + i + 1
                element_ids.append(elem_id)
                element_types.append(esmpy.MeshElemType.QUAD)
                # Counter-clockwise corner indices: (j, i), (j, i+1), (j+1, i+1), (j+1, i)
                n1 = tile_node_ids[j, i]
                n2 = tile_node_ids[j, i + 1]
                n3 = tile_node_ids[j + 1, i + 1]
                n4 = tile_node_ids[j + 1, i]
                element_conn.extend([n1, n2, n3, n4])

    node_coords = np.array(node_coords, dtype=np.float64)
    num_nodes = len(node_coords)
    num_elements = len(element_ids)

    mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=esmpy.CoordSys.SPH_DEG)

    node_ids = np.arange(1, num_nodes + 1, dtype=np.int32)
    node_owners = np.zeros(num_nodes, dtype=np.int32)
    mesh.add_nodes(num_nodes, node_ids, node_coords.flatten(), node_owners)

    element_ids = np.array(element_ids, dtype=np.int32)
    element_types = np.array(element_types, dtype=np.int32)
    element_conn = np.array(element_conn, dtype=np.int32)

    mesh.add_elements(num_elements, element_ids, element_types, element_conn)
    return mesh, ds_list


def main():
    print("Building ESMF C96 Mesh...")
    mesh_src, ds_list = build_esmf_c96_mesh()
    print("ESMF C96 Mesh built successfully.")

    # Build ESMF target grid (180x360 regular lat-lon)
    target_lats = np.linspace(-89.5, 89.5, 180)
    target_lons = np.linspace(0.5, 359.5, 360)

    grid_dst = esmpy.Grid(
        max_index=np.array([360, 180]),
        coord_sys=esmpy.CoordSys.SPH_DEG,
        staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
    )

    grid_dst_center_lon = grid_dst.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
    grid_dst_center_lat = grid_dst.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)

    lon_2d, lat_2d = np.meshgrid(target_lons, target_lats)
    grid_dst_center_lon[...] = lon_2d.T
    grid_dst_center_lat[...] = lat_2d.T

    # Corner coords for target grid
    grid_dst_corner_lon = grid_dst.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
    grid_dst_corner_lat = grid_dst.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)

    target_clats = np.linspace(-90.0, 90.0, 181)
    target_clons = np.linspace(0.0, 360.0, 361)
    clon_2d, clat_2d = np.meshgrid(target_clons, target_clats)
    grid_dst_corner_lon[...] = clon_2d.T
    grid_dst_corner_lat[...] = clat_2d.T

    field_src = esmpy.Field(mesh_src, meshloc=esmpy.MeshLoc.ELEMENT)
    field_dst = esmpy.Field(grid_dst, staggerloc=esmpy.StaggerLoc.CENTER)

    # Fill source field with 1.0 (constant)
    field_src.data[...] = 1.0
    field_dst.data[...] = 0.0

    # Create Regrid object with ESMF Conservative
    regrid_esmf = esmpy.Regrid(
        field_src, field_dst, regrid_method=esmpy.RegridMethod.CONSERVATIVE, unmapped_action=esmpy.UnmappedAction.IGNORE
    )

    regrid_esmf(field_src, field_dst)

    esmf_out_const = field_dst.data.T  # shape (180, 360)
    print("ESMF Conservative output min/max/mean for constant 1.0 field:")
    print("  Min :", esmf_out_const.min())
    print("  Max :", esmf_out_const.max())
    print("  Mean:", esmf_out_const.mean())
    print("  NaNs:", np.isnan(esmf_out_const).sum())

    # Now let's test smooth field cos(lat)*cos(lon)
    cell_lons = np.array([ds["x"].values[1::2, 1::2] for ds in ds_list])  # (6, 96, 96)
    cell_lats = np.array([ds["y"].values[1::2, 1::2] for ds in ds_list])  # (6, 96, 96)
    rad_lat = np.radians(cell_lats)
    rad_lon = np.radians(cell_lons)
    field_smooth_flat = (np.cos(rad_lat) * np.cos(rad_lon)).flatten()

    field_src.data[...] = field_smooth_flat
    field_dst.data[...] = 0.0
    regrid_esmf(field_src, field_dst)
    esmf_out_smooth = field_dst.data.T
    print("ESMF Conservative output min/max/mean for smooth field:")
    print("  Min :", esmf_out_smooth.min())
    print("  Max :", esmf_out_smooth.max())
    print("  Mean:", esmf_out_smooth.mean())


if __name__ == "__main__":
    main()
