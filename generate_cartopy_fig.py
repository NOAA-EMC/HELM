#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import axis
from axis import _core

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, sans-serif'

# Load C96 tiles
tile_files = sorted(glob.glob('/workspace/helm-project/libs/axis/C96_grid.tile*.nc'))
ds_list = [xr.open_dataset(f) for f in tile_files]

# Global regular target grid (180x360)
mesh_src_regular = _core.make_regular_mesh(360, 180, 0.0, -90.0, 1.0, 1.0)
src_lats_reg = np.linspace(-89.5, 89.5, 180)
src_lons_reg = np.linspace(0.5, 359.5, 360)
lon2d_reg, lat2d_reg = np.meshgrid(src_lons_reg, src_lats_reg)
field_src_smooth_reg = (np.cos(np.radians(lat2d_reg)) * np.cos(np.radians(lon2d_reg))).ravel()

config = {
    'method': _core.Method.Conservative,
    'periodic': False,
    'line_type': _core.LineType.GreatCircle,
    'norm_type': _core.NormType.FracArea,
    'unmapped': _core.UnmappedAction.Ignore,
}

fig = plt.figure(figsize=(15, 10), dpi=300)
# We will use Robinson projection for global view
proj = ccrs.Robinson(central_longitude=0)

tile_names = ["Tile 1 (Equatorial)", "Tile 2 (Equatorial)", "Tile 3 (North Pole)",
              "Tile 4 (Equatorial)", "Tile 5 (Equatorial)", "Tile 6 (South Pole)"]

for t in range(6):
    ax = fig.add_subplot(2, 3, t+1, projection=proj)
    
    ds = ds_list[t]
    clon_t = ds['x'].values[0::2, 0::2]
    clat_t = ds['y'].values[0::2, 0::2]
    coords_t = np.column_stack([clon_t.ravel(), clat_t.ravel()])
    ni, nj = 96, 96
    nip1 = ni + 1
    i_grid, j_grid = np.meshgrid(np.arange(ni), np.arange(nj))
    bl = (i_grid + j_grid * nip1).ravel()
    br = ((i_grid + 1) + j_grid * nip1).ravel()
    tr = ((i_grid + 1) + (j_grid + 1) * nip1).ravel()
    tl = (i_grid + (j_grid + 1) * nip1).ravel()
    indices_t = np.column_stack([bl, br, tr, tl]).astype(np.int64).ravel()
    offsets_t = np.arange(0, len(indices_t) + 1, 4, dtype=np.int64)
    mesh_tile = _core.make_ugrid_mesh(np.asfortranarray(coords_t), offsets_t, indices_t)
    
    weights = _core.generate_weights(mesh_src_regular, mesh_tile, config)
    out_tile_smooth = np.array(_core.apply_weights(weights, field_src_smooth_reg)).reshape((96, 96))
    
    cell_lon = ds['x'].values[1::2, 1::2]
    cell_lat = ds['y'].values[1::2, 1::2]
    exact_tile_smooth = np.cos(np.radians(cell_lat)) * np.cos(np.radians(cell_lon))
    
    err_tile = (out_tile_smooth - exact_tile_smooth) * 1e3  # Scale by 10^3 for readability
    
    # Fix the wrap-around artifact in pcolormesh by making longitudes continuous
    # If the max difference in longitude within the tile is > 180, we shift >180 to negative.
    lon_plot = cell_lon.copy()
    if np.ptp(lon_plot) > 180:
        lon_plot[lon_plot > 180] -= 360

    # Draw coastlines and gridlines for reference
    ax.coastlines(color='black', linewidth=0.5, alpha=0.5)
    ax.gridlines(color='gray', alpha=0.3, linestyle='--')
    ax.set_global()

    # Use scatter to plot the exact cell centers to completely avoid any pcolormesh topological rendering bugs
    # This guarantees we see exactly what the data is without matplotlib connecting edges incorrectly
    sc = ax.scatter(lon_plot.ravel(), cell_lat.ravel(), c=err_tile.ravel(), 
                    cmap='RdBu_r', vmin=-2.0, vmax=2.0, s=8, transform=ccrs.PlateCarree(),
                    edgecolors='none')
                    
    ax.set_title(f"{tile_names[t]}\nMax Abs Err: {np.max(np.abs(out_tile_smooth - exact_tile_smooth)):.4e}", fontsize=10, fontweight='bold')

plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Interpolation Truncation Error (× 10⁻³)", fontsize=12, fontweight='bold')

plt.suptitle("AXIS True Error Distribution Mapped on Globe (Avoiding Plotting Artifacts)", fontsize=14, fontweight='bold')
output_fig = "/workspace/helm-project/c96_cartopy_error_map.png"
plt.savefig(output_fig, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved Map Figure to {output_fig}")
