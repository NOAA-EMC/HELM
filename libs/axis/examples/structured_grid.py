#!/usr/bin/env python3
"""Build structured meshes from center coordinates with AXIS."""

import axis
import numpy as np


def main() -> None:
    lon_centers = np.array([45.0, 135.0, 225.0, 315.0])
    lat_centers = np.array([-45.0, 45.0])

    rect = axis.RectilinearGrid(lon_centers, lat_centers, longitude_periodic=True)
    rect_mesh = rect.to_mesh()
    print(f"Rectilinear mesh: {rect_mesh.n_cells} cells, {rect_mesh.n_nodes} nodes")

    # Gaussian-Legendre latitude centers derive exact latitude bands by default.
    mu, _ = np.polynomial.legendre.leggauss(4)
    gaussian_latitudes = np.degrees(np.arcsin(mu))
    gaussian = axis.GaussianGrid(lon_centers, gaussian_latitudes)
    gaussian_mesh = gaussian.to_mesh()
    print(f"Gaussian mesh: {gaussian_mesh.n_cells} cells, {gaussian_mesh.n_nodes} nodes")

    # Lower-level API for explicit corner and seam policy control.
    ni, nj = 2, 2
    center_lon = np.array([0.0, 1.0, 0.0, 1.0])
    center_lat = np.array([0.0, 0.0, 1.0, 1.0])
    grid = axis.axis_py.StructuredGrid(ni, nj, center_lon, center_lat)
    mesh = grid.to_unstructured(
        axis.CornerPolicy.CurvilinearApproximate,
        axis.LongitudePeriodicity(periodic=True, period=360.0, increasing=True),
    )
    print(f"Explicit mesh: {mesh.n_cells} cells, {mesh.n_nodes} nodes")


if __name__ == "__main__":
    main()
