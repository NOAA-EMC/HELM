# SPDX-License-Identifier: Apache-2.0
"""Coastal (source-cell) masking with conservative weight renormalization.

Ported to the new fit-on-construction API (FR-040): ``src_mask`` rides on the
constructor, and a flat wet field must remap to 1.0 under frac-area norm once
dry cells are renormalized away.
"""

import numpy as np
import xarray as xr
from axis import Regridder


def test_coastal_mask_application():
    # 4x4 source grid
    lons_in = np.linspace(-10, 10, 4)
    lats_in = np.linspace(-10, 10, 4)
    ds_in = xr.Dataset(coords={"lat": lats_in, "lon": lons_in})

    # 2x2 target grid
    lons_out = np.linspace(-10, 10, 2)
    lats_out = np.linspace(-10, 10, 2)
    ds_out = xr.Dataset(coords={"lat": lats_out, "lon": lons_out})

    # Source land mask: 0 = dry land, 1 = ocean. Top-left quadrant is land.
    src_mask = np.ones((4, 4), dtype=np.int32)
    src_mask[0:2, 0:2] = 0

    regridder = Regridder(ds_in, ds_out, method="conservative", src_mask=src_mask)

    # A flat wet field must map to 1.0 under frac-area renormalization over the
    # active (wet) source cells.
    src_data = np.ones((4, 4), dtype=np.float64)
    res = regridder(src_data)

    assert res.shape == (2, 2)
    assert not np.any(np.isnan(res))
    np.testing.assert_allclose(res, 1.0, rtol=1e-10, atol=1e-10)
