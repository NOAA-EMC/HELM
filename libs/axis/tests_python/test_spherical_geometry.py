# SPDX-License-Identifier: Apache-2.0

import numpy as np
from axis import axis_py

Vec3 = axis_py.Vec3
great_circle_arc_intersection = axis_py.great_circle_arc_intersection
lonlat_to_xyz = axis_py.lonlat_to_xyz
robust_orient_sphere = axis_py.robust_orient_sphere
xyz_to_lonlat = axis_py.xyz_to_lonlat


def test_spherical_conversions():
    # Convert 45 lon, 45 lat in radians to Cartesian
    p = lonlat_to_xyz(np.pi / 4.0, np.pi / 4.0)
    assert isinstance(p, Vec3)
    assert np.isclose(p.x, 0.5)
    assert np.isclose(p.y, 0.5)
    assert np.isclose(p.z, np.sqrt(2.0) / 2.0)

    lon, lat = xyz_to_lonlat(p)
    assert np.isclose(lon, np.pi / 4.0)
    assert np.isclose(lat, np.pi / 4.0)


def test_robust_orientation():
    a = Vec3(1.0, 0.0, 0.0)
    b = Vec3(0.0, 1.0, 0.0)
    # Point c is CCW (to the left)
    c_left = Vec3(0.5, 0.5, 0.5)
    val = robust_orient_sphere(a, b, c_left)
    assert val > 0.0


def test_arc_intersection():
    # Arc 1: Along equator (0 to 90 East)
    a1 = Vec3(1.0, 0.0, 0.0)
    a2 = Vec3(0.0, 1.0, 0.0)
    # Arc 2: Along prime meridian (-45 to 45 North)
    b1 = lonlat_to_xyz(0.0, -np.pi / 4.0)
    b2 = lonlat_to_xyz(0.0, np.pi / 4.0)

    p_inter = great_circle_arc_intersection(a1, a2, b1, b2)
    assert p_inter is not None
    assert np.isclose(p_inter.x, 1.0)
    assert np.isclose(p_inter.y, 0.0)
    assert np.isclose(p_inter.z, 0.0)
