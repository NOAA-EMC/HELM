# SPDX-License-Identifier: Apache-2.0
# test_vector_regridder.py
#
# Coupled wind-vector interpolation with local grid-frame rotation (ported to
# the redesigned Grid/VectorRegridder API — T035).

import numpy as np
from axis import Grid, VectorRegridder


def test_vector_regridder_numerical_precision():
    # Source grid lons/lats well within bounds
    lons_in = np.linspace(-170, 170, 4)
    lats_in = np.linspace(-80, 80, 4)

    # Target grid lons/lats nested inside source domain
    lons_out = np.linspace(-90, 90, 2)
    lats_out = np.linspace(-45, 45, 2)

    # Generate flat eastward and northward wind components
    u_in = np.ones((4, 4), dtype=np.float64) * 10.0
    v_in = np.ones((4, 4), dtype=np.float64) * 5.0

    # Instantiate coupled vector regridder (engine auto-computes rotation angles)
    regridder = VectorRegridder(
        Grid(lon=lons_in, lat=lats_in),
        Grid(lon=lons_out, lat=lats_out),
        method="bilinear",
    )

    u_out, v_out = regridder.transform(u_in, v_in)

    assert u_out.shape == (2, 2)
    assert v_out.shape == (2, 2)
    np.testing.assert_allclose(u_out, 10.0, rtol=1e-12)
    np.testing.assert_allclose(v_out, 5.0, rtol=1e-12)


def test_vector_regridder_accepts_dict_geometry():
    # Same setup as above, but pass grids as coordinate dicts to exercise the
    # dict-normalization path (mirrors Regridder's accepted input types).
    lons_in = np.linspace(-170, 170, 4)
    lats_in = np.linspace(-80, 80, 4)
    lons_out = np.linspace(-90, 90, 2)
    lats_out = np.linspace(-45, 45, 2)

    u_in = np.ones((4, 4), dtype=np.float64) * 10.0
    v_in = np.ones((4, 4), dtype=np.float64) * 5.0

    regridder = VectorRegridder(
        {"lon": lons_in, "lat": lats_in},
        {"lon": lons_out, "lat": lats_out},
        method="bilinear",
    )

    u_out, v_out = regridder.transform(u_in, v_in)

    assert u_out.shape == (2, 2)
    assert v_out.shape == (2, 2)
    np.testing.assert_allclose(u_out, 10.0, rtol=1e-12)
    np.testing.assert_allclose(v_out, 5.0, rtol=1e-12)
