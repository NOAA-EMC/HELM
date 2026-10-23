"""Exercise the installed wheel outside the source package."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import axis
import numpy as np
import xarray as xr

assert "site-packages" in str(Path(axis.__file__).resolve()), axis.__file__
mesh = axis.make_regular_mesh(4, 4, 0.0, 0.0, 1.0, 1.0)
assert mesh is not None
source = xr.Dataset(coords={"lat": [-30.0, 0.0, 30.0], "lon": [0.0, 30.0, 60.0]})
target = xr.Dataset(coords={"lat": [-15.0, 15.0], "lon": [15.0, 45.0]})
field = xr.DataArray(np.ones((3, 3)), dims=("lat", "lon"), coords=source.coords)
regridder = axis.Regridder(source, target, method="nearest")
try:
    result = regridder(field)
    assert result.shape == (2, 2), result.shape
    np.testing.assert_allclose(result.values, 1.0)
finally:
    regridder.close()
assert importlib.metadata.version("axis-regrid")
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
subprocess.run(["axis-regrid", "--help"], check=True)
print("Installed AXIS native wheel passed smoke tests")
