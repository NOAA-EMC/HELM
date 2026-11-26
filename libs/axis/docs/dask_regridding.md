# Distributed conservative regridding with Dask

Use `axis.DistributedRegridder` with a Dask distributed client and spherical
UGRID mesh tiles in degrees. Each tile supplies a zero-argument loader that
opens its own storage and returns nodes, local CSR offsets and connectivity,
and an optional integer cell mask. Avoid capturing the full mesh in a loader.
Use storage chunks sized for the worker memory limit. Tile order defines the
global cell order, and the source field cell dimension must follow that order.

```python
from axis import DistributedRegridder, MeshTile

sources = [MeshTile(size, loader) for size, loader in source_chunks]
targets = [MeshTile(size, loader) for size, loader in destination_chunks]
regridder = DistributedRegridder(client).fit(sources, targets)
result = regridder.apply(source_field, dim="cell")
```

The field must be a Dask-backed xarray DataArray. The result is lazy and
preserves non-cell coordinates. Attach destination cell coordinates as needed.
Keep the client alive while computing results. Reuse the fitted regridder for
additional fields with the same source cell order.

This path supports first-order conservative remapping with great-circle edges.
The default normalization is destination area. Use `norm="fracarea"` for
coverage normalization. Unmapped cells have zero weight. Fit inspects tile
bounds on workers and builds one sparse block per destination tile. It splits destination tiles in cell order when the estimated geometry and
worst-case weight entries exceed `memory_budget`. If a single cell still exceeds
the estimate, fit raises an error. Individual storage tiles must fit on a worker
before splitting. This estimate applies per task rather than to total
worker memory. Apply passes only referenced source values to each weight task, including reconstruction neighbors. Storage reads still follow Dask chunk boundaries.

## Other methods

Specify `method="nearest_neighbor"`, `"bilinear"`, `"bicubic"`, or `"patch"`
to select source stencils across tiles. Each source tile returns its best
candidates for each destination cell. Dask merges those lists using distance
and global cell order, then sends only selected geometry to the weight task.
Bicubic considers up to 64 active source cells per destination. Application
passes only selected cell values to the weight block. Storage loaders and
Dask source chunks still read data at tile granularity.

For second-order conservative remapping, pass `method="conservative2nd"`.
Source tile loaders must return a fifth array of global node IDs after the
cell mask. Reuse node IDs on shared boundaries and use consistent coordinates
for those nodes. Dask expands candidate geometry by two rings of active cells
sharing those IDs and preserves the native reconstruction payload with each
weight block. The current ring lookup scans tile metadata on workers for each
destination partition.
