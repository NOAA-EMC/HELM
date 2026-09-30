# SPDX-License-Identifier: Apache-2.0
"""
AXIS Python package — stateless, high-performance spatial regridding for xarray.
"""
# ruff: noqa: I001

# Import axis_py FIRST using relative import to completely avoid partially initialized circular issues
from . import axis_py  # noqa: F401

# Register the custom .axis xarray accessor
from . import accessors  # noqa: F401
from .grid import CurvilinearGrid, Geometry, GridFactory, RectilinearGrid, UnstructuredMesh, RuleGeometry
from .regridder import Regridder
from .vector import VectorRegridder
from .vertical import VerticalRegridder, regrid_3d

# Expose C++ Mesh construction and Matrix serialization APIs directly on the axis package
from .axis_py import (
    LineType,
    Matrix,
    Mesh,
    Method,
    NormType,
    UnmappedAction,
    adjust_by_fraction,
    apply_weights,
    batch_apply,
    detect_rectilinear_grid,
    detect_regular_grid,
    detect_tripolar_grid,
    generate_vector_weights,
    make_named_mesh,
    make_projected_mesh,
    make_regular_mesh,
    make_ugrid_mesh,
    reconstruct_gradient,
    write_gmsh,
)

__all__ = [
    "Regridder",
    "Geometry",
    "RectilinearGrid",
    "CurvilinearGrid",
    "UnstructuredMesh",
    "RuleGeometry",
    "GridFactory",
    "VectorRegridder",
    "VerticalRegridder",
    "regrid_3d",
    "Mesh",
    "Matrix",
    "make_regular_mesh",
    "make_projected_mesh",
    "make_ugrid_mesh",
    "make_named_mesh",
    "apply_weights",
    "batch_apply",
    "write_gmsh",
    "reconstruct_gradient",
    "detect_tripolar_grid",
    "detect_regular_grid",
    "detect_rectilinear_grid",
    "adjust_by_fraction",
    "generate_vector_weights",
    "Method",
    "NormType",
    "UnmappedAction",
    "LineType",
]
