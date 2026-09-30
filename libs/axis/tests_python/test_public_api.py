# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the public AXIS Python API surface.

These assert the *shape* of ``import axis`` exactly as specified in
``specs/001-axis-python-api-redesign/contracts/python-api.md``: the curated
``__all__``, the constructor signatures of ``Grid``/``Regridder``/
``VectorRegridder``/``VerticalRegridder``/``regrid_3d``, the string-coercible
enums, the error hierarchy, and the retirement of the legacy surface (FR-038,
FR-040).

They are intentionally *importable but red* until the rewrite (T019+) lands:
optional third-party deps are importorskip-guarded, but the API-shape
assertions run unconditionally so this file is the TDD tripwire for the
public contract.
"""

from __future__ import annotations

import inspect
from enum import Enum, StrEnum

import pytest

axis = pytest.importorskip("axis")

# The exact public surface from contracts/python-api.md.
EXPECTED_ALL = [
    "Grid",
    "GridFamily",
    "Regridder",
    "VectorRegridder",
    "VerticalRegridder",
    "regrid_3d",
    "Method",
    "Norm",
    "Unmapped",
    "LineType",
    "AxisError",
    "GridError",
    "AxisConfigError",
    "AxisShapeError",
    "AxisWeightMismatchError",
    "AxisCapabilityError",
    "AxisUnmappedError",
]

# Legacy names that MUST NOT be reachable from the top-level package (FR-040).
LEGACY_RETIRED = [
    "GridFactory",
    "Geometry",
    "RectilinearGrid",
    "CurvilinearGrid",
    "UnstructuredMesh",
    "make_regular_mesh",
    "make_projected_mesh",
    "make_ugrid_mesh",
    "make_named_mesh",
    "apply_weights",
    "batch_apply",
    "detect_tripolar_grid",
    "generate_vector_weights",
    "Mesh",
    "Matrix",
]


# ─── __all__ curation ────────────────────────────────────────────────────────


def test_all_matches_contract():
    assert set(axis.__all__) == set(EXPECTED_ALL), (
        f"axis.__all__ mismatch\nmissing: {set(EXPECTED_ALL) - set(axis.__all__)}\nextra:   {set(axis.__all__) - set(EXPECTED_ALL)}"
    )


def test_all_symbols_exist_and_ordered():
    for name in axis.__all__:
        assert hasattr(axis, name), f"{name!r} listed in __all__ but missing from module"


def test_engine_boundary_is_internal():
    # axis._core is the compiled engine module: reachable but NOT public.
    assert "_core" not in axis.__all__
    assert hasattr(axis, "_core")


@pytest.mark.parametrize("name", LEGACY_RETIRED)
def test_legacy_surface_retired(name):
    assert not hasattr(axis, name), f"legacy {name!r} must not be exported (FR-040)"


# ─── Enums ───────────────────────────────────────────────────────────────────


def test_method_enum():
    assert issubclass(axis.Method, StrEnum)
    assert {m.value for m in axis.Method} == {
        "bilinear",
        "bicubic",
        "patch",
        "nearest",
        "conservative",
        "conservative2nd",
    }
    # string-coercible: a plain string equals the member
    assert axis.Method("bilinear") is axis.Method.BILINEAR


def test_norm_enum():
    assert issubclass(axis.Norm, StrEnum)
    assert {m.value for m in axis.Norm} == {"frac_area", "dst_area"}


def test_unmapped_enum():
    assert issubclass(axis.Unmapped, StrEnum)
    assert {m.value for m in axis.Unmapped} == {"nan", "mask", "error"}


def test_linetype_enum():
    assert issubclass(axis.LineType, StrEnum)
    assert {m.value for m in axis.LineType} == {"great_circle", "cartesian"}


def test_gridfamily_enum():
    assert issubclass(axis.GridFamily, Enum)
    assert {m.value for m in axis.GridFamily} == {
        "rectilinear",
        "curvilinear",
        "cubed_sphere",
        "ugrid",
        "icon",
        "points",
    }


# ─── Error hierarchy ─────────────────────────────────────────────────────────


def test_error_hierarchy():
    assert issubclass(axis.AxisError, Exception)
    for name in [
        "GridError",
        "AxisConfigError",
        "AxisShapeError",
        "AxisWeightMismatchError",
        "AxisCapabilityError",
        "AxisUnmappedError",
    ]:
        assert issubclass(getattr(axis, name), axis.AxisError), f"{name} must derive from AxisError"


# ─── Constructor signatures ──────────────────────────────────────────────────


def _params(func) -> set[str]:
    return set(inspect.signature(func).parameters)


def test_grid_init_signature():
    p = _params(axis.Grid.__init__)
    assert {"obj", "lon", "lat", "bounds", "mesh", "location", "periodic", "line_type", "tripolar"} <= p


@pytest.mark.parametrize(
    "factory",
    ["from_xarray", "from_arrays", "from_named", "from_ugrid", "from_points", "from_dict"],
)
def test_grid_alternate_constructors(factory):
    assert callable(getattr(axis.Grid, factory))


def test_grid_readonly_properties():
    for prop in ["family", "dims", "shape", "n_cells", "periodic", "line_type", "fingerprint"]:
        assert isinstance(getattr(axis.Grid, prop), property), f"Grid.{prop} must be a read-only property"


def test_regridder_init_signature():
    p = _params(axis.Regridder.__init__)
    assert {
        "source",
        "target",
        "method",
        "norm",
        "unmapped",
        "skipna",
        "na_thres",
        "periodic",
        "line_type",
        "src_mask",
        "dst_mask",
    } <= p


def test_regridder_call_and_aliases():
    for m in [
        "__call__",
        "transform",
        "regrid",
        "save_weights",
        "load_weights",
        "to_esmf",
        "from_esmf",
        "regrid_categorical",
        "summary",
    ]:
        assert hasattr(axis.Regridder, m), f"Regridder.{m} missing"
    assert isinstance(axis.Regridder.nnz, property)
    assert isinstance(axis.Regridder.fitted, property)


def test_vector_regridder_init_signature():
    p = _params(axis.VectorRegridder.__init__)
    assert {"source", "target", "method", "src_alpha", "dst_alpha"} <= p


def test_vertical_regridder_init_signature():
    p = _params(axis.VerticalRegridder.__init__)
    assert {"tension", "out_of_range"} <= p


def test_regrid_3d_signature():
    p = _params(axis.regrid_3d)
    assert {"field", "source", "target", "src_levels", "dst_levels", "method", "tension"} <= p
