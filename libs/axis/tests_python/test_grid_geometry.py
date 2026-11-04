# test_grid_geometry.py
#
# Verifies explicit coordinate layouts (Rectilinear, Curvilinear, Unstructured).

import numpy as np
from axis.grid import CurvilinearGrid, GaussianGrid, MultiFaceGrid, RectilinearGrid, ReducedGaussianGrid, UnstructuredMesh


def test_rectilinear_grid_generation():
    lons = np.linspace(-180, 180, 36)
    lats = np.linspace(-90, 90, 18)

    grid = RectilinearGrid(lons, lats)
    mesh = grid.to_mesh()

    # 36 * 18 cell mesh has (36+1)*(18+1) = 37 * 19 = 703 nodes, and 36 * 18 = 648 cells
    assert mesh.n_nodes == 703
    assert mesh.n_cells == 648


def test_curvilinear_grid_generation():
    # 5x5 grid centers
    x = np.arange(5)
    y = np.arange(5)
    lons, lats = np.meshgrid(x, y)

    grid = CurvilinearGrid(lons, lats)
    mesh = grid.to_mesh()

    # Curvilinear corner synthesis pads (ny, nx) cells to (ny+1, nx+1) corners
    # For a 5x5 center array: NY=5, NX=5 -> Corners: (NY+1)*(NX+1) = 36 nodes, NY*NX = 25 cells
    assert mesh.n_nodes == 36
    assert mesh.n_cells == 25


def test_curvilinear_grid_can_require_authoritative_corners():
    lons, lats = np.meshgrid(np.array([0.25, 0.75]), np.array([0.25, 0.75]))
    corner_lons, corner_lats = np.meshgrid(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    grid = CurvilinearGrid(
        lons,
        lats,
        corner_policy="require_explicit",
        corner_lons=corner_lons,
        corner_lats=corner_lats,
    )
    mesh = grid.to_mesh()
    assert mesh.n_cells == 4
    assert mesh.n_nodes == 9


def test_rectilinear_center_conversion_uses_native_explicit_bounds():
    grid = RectilinearGrid(np.array([45.0, 135.0, 225.0, 315.0]), np.array([-45.0, 45.0]), longitude_periodic=True)
    mesh = grid.to_mesh()
    assert mesh.n_cells == 8
    assert mesh.n_nodes == 15


def test_gaussian_grid_constructs_quadrature_bands_from_centers():
    mu, _ = np.polynomial.legendre.leggauss(4)
    lats = np.degrees(np.arcsin(mu))
    lons = np.array([45.0, 135.0, 225.0, 315.0])
    mesh = GaussianGrid(lons, lats).to_mesh()
    assert mesh.n_cells == 16
    assert mesh.n_nodes == 25


def test_reduced_gaussian_grid_preserves_ragged_cell_count():
    grid = ReducedGaussianGrid(
        np.array([-60.0, 0.0, 60.0]),
        [np.array([-180.0, -60.0, 60.0, 180.0]), np.array([-180.0, 0.0, 180.0])],
        longitude_periodic=True,
    )
    mesh = grid.to_mesh()
    assert mesh.n_cells == 5


def test_multiface_grid_welds_explicit_shared_edge():
    faces = [
        {"ni": 1, "nj": 1, "node_coords": np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)},
        {"ni": 1, "nj": 1, "node_coords": np.array([[1, 0], [2, 0], [1, 1], [2, 1]], dtype=np.float64)},
    ]
    connections = [{"face_a": 0, "edge_a": "east", "face_b": 1, "edge_b": "west", "reversed": False}]
    mesh = MultiFaceGrid(faces, connections).to_mesh()
    assert mesh.n_cells == 2
    assert mesh.n_nodes == 6


def test_unstructured_mesh_generation():
    # Define a single triangular cell with 3 nodes
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    offsets = np.array([0, 3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)

    mesh_geom = UnstructuredMesh(coords, offsets, indices)
    mesh = mesh_geom.to_mesh()

    assert mesh.n_nodes == 3
    assert mesh.n_cells == 1
