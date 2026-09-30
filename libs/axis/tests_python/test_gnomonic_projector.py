# SPDX-License-Identifier: Apache-2.0

import numpy as np
from axis import axis_py

Vec3 = axis_py.Vec3
lonlat_to_xyz = axis_py.lonlat_to_xyz
forward = axis_py.gnomonic_forward
inverse = axis_py.gnomonic_inverse
bilinear_weights = axis_py.bilinear_weights


def test_gnomonic_projections():
    center = Vec3(1.0, 0.0, 0.0)
    pt = lonlat_to_xyz(np.radians(10.0), np.radians(10.0))

    u, v = forward(center, pt)
    assert np.isclose(u, np.tan(np.radians(10.0)))

    reconstructed = inverse(center, u, v)
    assert np.isclose(reconstructed.x, pt.x)
    assert np.isclose(reconstructed.y, pt.y)
    assert np.isclose(reconstructed.z, pt.z)


def test_bilinear_solver():
    # Simple projected quad with resolution 2.0 (from -1 to 1)
    quad_u = np.array([-1.0, 1.0, 1.0, -1.0])
    quad_v = np.array([-1.0, -1.0, 1.0, 1.0])

    # Point at exact center
    w = bilinear_weights(quad_u, quad_v, 0.0, 0.0)
    assert w is not None
    assert np.allclose(w, 0.25)
