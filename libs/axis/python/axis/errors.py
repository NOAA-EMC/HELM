# SPDX-License-Identifier: Apache-2.0
"""AXIS error hierarchy.

Every error names the offending input and a concrete fix (spec SC-006).
All fit/construction-time failures surface here — never deferred to apply.
"""

from __future__ import annotations


class AxisError(Exception):
    """Base class for all AXIS errors."""


class GridError(AxisError):
    """Grid detection failed or was ambiguous.

    The message enumerates every pattern searched (CF standard_name/axis
    attributes first, then common-name heuristics, then UGRID mesh roles)
    and shows the explicit-override path (FR-005).
    """

    def __init__(self, message: str, *, searched: list[str] | None = None) -> None:
        if searched:
            message = (
                f"{message}\n\nPatterns searched (in order):\n"
                + "\n".join(f"  - {p}" for p in searched)
                + "\n\nOverride explicitly, e.g.:\n"
                "  axis.Grid(ds, lon='my_lon', lat='my_lat')\n"
                "  axis.Grid(ds, mesh='mesh', location='cell')"
            )
        super().__init__(message)


class AxisConfigError(AxisError):
    """Structurally impossible configuration raised at fit/construction time.

    Examples: unknown method name, bicubic onto a point cloud, conservative
    vector regridding (FR-010, FR-023).
    """


class AxisShapeError(AxisError):
    """Input field's trailing dimensions do not match the source grid (FR-016)."""


class AxisWeightMismatchError(AxisError):
    """Reloaded weights do not match the declared source/target grids (FR-029).

    Raised before any data is touched.
    """


class AxisCapabilityError(AxisError):
    """The compiled engine lacks a capability the API needs.

    Examples: PROJ required for projected grids but built without
    ``AXIS_ENABLE_PROJ``; NetCDF weight I/O but built without
    ``AXIS_HAVE_NETCDF``.
    """


class AxisUnmappedError(AxisError):
    """Destination points had no source coverage with ``unmapped='error'``.

    The message lists the first offending indices (FR-017).
    """

    def __init__(self, message: str, *, unmapped_indices: list[int] | None = None) -> None:
        if unmapped_indices:
            shown = ", ".join(str(i) for i in unmapped_indices[:10])
            more = f" (+{len(unmapped_indices) - 10} more)" if len(unmapped_indices) > 10 else ""
            message = f"{message} Unmapped destination indices: [{shown}{more}]"
        super().__init__(message)
