# SPDX-License-Identifier: Apache-2.0
"""Public enums for the AXIS API.

All option enums are ``StrEnum`` so callers may pass plain strings
(``method="bilinear"``) or the enum members (``method=Method.BILINEAR``)
interchangeably.
"""

from __future__ import annotations

from enum import Enum, StrEnum


class GridFamily(Enum):
    """The six supported spatial grid families (FR-001)."""

    RECTILINEAR = "rectilinear"
    CURVILINEAR = "curvilinear"
    CUBED_SPHERE = "cubed_sphere"
    UGRID = "ugrid"
    ICON = "icon"
    POINTS = "points"


class Method(StrEnum):
    """Horizontal interpolation methods (FR-010)."""

    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    PATCH = "patch"
    NEAREST = "nearest"
    CONSERVATIVE = "conservative"
    CONSERVATIVE_2ND = "conservative2nd"


class Norm(StrEnum):
    """Conservative weight normalization semantics (FR-011)."""

    FRAC_AREA = "frac_area"
    DST_AREA = "dst_area"


class Unmapped(StrEnum):
    """Policy for destination points with no source coverage (FR-017).

    - ``NAN`` (default): fill unmapped points with NaN.
    - ``MASK``: mark unmapped points invalid (NaN) via the engine mask path.
    - ``ERROR``: raise :class:`axis.AxisUnmappedError` at fit time.
    """

    NAN = "nan"
    MASK = "mask"
    ERROR = "error"


class LineType(StrEnum):
    """Geometry of paths between grid points (FR-011)."""

    GREAT_CIRCLE = "great_circle"
    CARTESIAN = "cartesian"


class Location(StrEnum):
    """UGRID mesh element location for unstructured grids (FR-004)."""

    AUTO = "auto"
    CELL = "cell"
    NODE = "node"
    EDGE = "edge"


class OutOfRange(StrEnum):
    """Vertical interpolation out-of-range behavior (FR-026)."""

    NAN = "nan"
    CLIP = "clip"
    EXTRAPOLATE = "extrapolate"


# Methods whose stencils require triangulated source cells on
# polygon-based unstructured meshes (vs. raw Voronoi polygons).
TRIANGULATED_METHODS: frozenset[str] = frozenset({Method.BILINEAR, Method.BICUBIC, Method.PATCH})
POLYGON_METHODS: frozenset[str] = frozenset({Method.NEAREST, Method.CONSERVATIVE, Method.CONSERVATIVE_2ND})
