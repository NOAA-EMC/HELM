# SPDX-License-Identifier: Apache-2.0
"""FR-040 / SC-010 — the pre-redesign Python surface is fully retired.

Every legacy symbol must be (a) absent from ``axis.__all__``, (b) unreachable
as an attribute of the ``axis`` package, and (c) unimportable from the
submodules that used to define it. The compiled engine stays internal at
``axis._core`` and is never re-exported.
"""

from __future__ import annotations

import importlib

import axis
import pytest

# FR-040 disposition inventory — legacy names that must be gone from the
# public package surface.
RETIRED_PACKAGE_ATTRS = [
    # legacy grid abstraction
    "GridFactory",
    "Geometry",
    "RectilinearGrid",
    "CurvilinearGrid",
    "UnstructuredMesh",
    "XarrayGeometry",
    "create_axis_mesh",
    # raw _core re-exports
    "Mesh",
    "Matrix",
    "make_regular_mesh",
    "make_projected_mesh",
    "make_ugrid_mesh",
    "make_named_mesh",
    "apply_weights",
    "batch_apply",
    "detect_tripolar_grid",
    "generate_vector_weights",
    "NormType",
    "UnmappedAction",
]


def test_none_retired_in_all():
    leaked = [n for n in RETIRED_PACKAGE_ATTRS if n in axis.__all__]
    assert not leaked, f"retired symbols leaked into __all__: {leaked}"


def test_none_retired_is_package_attr():
    leaked = [n for n in RETIRED_PACKAGE_ATTRS if hasattr(axis, n)]
    assert not leaked, f"retired symbols reachable on axis: {leaked}"


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("axis.grid", "GridFactory"),
        ("axis.grid", "Geometry"),
        ("axis.grid", "RectilinearGrid"),
        ("axis.grid", "CurvilinearGrid"),
        ("axis.grid", "UnstructuredMesh"),
        ("axis.grid", "XarrayGeometry"),
        ("axis.grid", "create_axis_mesh"),
    ],
)
def test_retired_symbols_unimportable_from_submodule(module, name):
    """FR-040: no compatibility shims — the symbols are removed, not aliased."""
    mod = importlib.import_module(module)
    assert not hasattr(mod, name), f"{module}.{name} must be removed (FR-040)"


def test_core_not_reexported():
    assert "_core" not in axis.__all__
    # reachable as the internal engine module, but never a public name
    assert axis._core is importlib.import_module("axis._core")


def test_new_surface_is_present():
    for name in ["Grid", "Regridder", "VectorRegridder", "VerticalRegridder", "regrid_3d"]:
        assert hasattr(axis, name), f"new public symbol {name!r} missing"
        assert name in axis.__all__
