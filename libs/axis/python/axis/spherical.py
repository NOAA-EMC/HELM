# SPDX-License-Identifier: Apache-2.0
"""Spherical geometry, 3D vector conversions, and arc operations."""

from . import axis_py

Vec3 = axis_py.Vec3
lonlat_to_xyz = axis_py.lonlat_to_xyz
xyz_to_lonlat = axis_py.xyz_to_lonlat
robust_orient_sphere = axis_py.robust_orient_sphere
great_circle_arc_intersection = axis_py.great_circle_arc_intersection

__all__ = [
    "Vec3",
    "lonlat_to_xyz",
    "xyz_to_lonlat",
    "robust_orient_sphere",
    "great_circle_arc_intersection",
]
