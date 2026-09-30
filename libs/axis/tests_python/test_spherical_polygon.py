# SPDX-License-Identifier: Apache-2.0

import numpy as np
from axis import axis_py

SphericalPolygon = axis_py.SphericalPolygon
lonlat_to_xyz = axis_py.lonlat_to_xyz


def test_spherical_excess_area():
    # Create an octant on the unit sphere (longitude 0 to 90 East, equator to North Pole)
    # Area must be 1/8 of total sphere area = (4 * pi) / 8 = pi / 2 ≈ 1.570796
    poly = SphericalPolygon()

    # Vertices
    v0 = lonlat_to_xyz(0.0, 0.0)
    v1 = lonlat_to_xyz(np.pi / 2.0, 0.0)
    v2 = lonlat_to_xyz(0.0, np.pi / 2.0)

    poly.add_vertex(v0)
    poly.add_vertex(v1)
    poly.add_vertex(v2)

    assert poly.n == 3
    area_calc = poly.area()
    assert np.isclose(area_calc, np.pi / 2.0, rtol=1e-12)
