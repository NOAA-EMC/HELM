# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the AXIS Python API (axis_py nanobind module).

These tests exercise the main binding surface:
  - Mesh construction (regular lat-lon, named grids)
  - Weight generation with RegridConfig
  - Single-field apply and multi-field batch_apply
  - Weight cache serialize/deserialize round-trip

Requirements validated: 12.3, 12.5, 12.6
"""

from typing import Any

import numpy as np
import pytest

# axis_py is the nanobind extension module built with AXIS_BUILD_PYTHON=ON.
# These tests are written to be runnable once the module is available.
axis = pytest.importorskip("axis")
axis_py = axis.axis_py


# ─── Test: generate + apply produces correct results (Req 12.3) ───────────────


class TestGenerateAndApply:
    """Validates Requirement 12.3: apply() with numpy arrays."""

    def test_generate_and_apply_bilinear_constant_field(
        self, small_src_mesh: Any, small_dst_mesh: Any, bilinear_matrix: Any
    ) -> None:
        """
        A spatially constant field interpolated via bilinear weights
        must produce the same constant on the destination mesh.

        This validates the full pipeline: mesh → generate → apply → numpy result.
        """
        n_src: int = bilinear_matrix.n_src
        n_dst: int = bilinear_matrix.n_dst

        # Constant field: every source cell = 42.0
        src_field: np.ndarray = np.full(n_src, 42.0, dtype=np.float64)

        # Apply interpolation
        dst_field: np.ndarray = axis_py.apply_weights(bilinear_matrix, src_field)

        # Result should be a numpy array of length n_dst
        assert isinstance(dst_field, np.ndarray)
        assert dst_field.shape == (n_dst,)
        assert dst_field.dtype == np.float64

        # All destination values should equal the constant (partition of unity)
        np.testing.assert_allclose(dst_field, 42.0, rtol=1e-12)

    def test_apply_returns_correct_shape(self, bilinear_matrix: Any) -> None:
        """apply_weights returns array with shape (n_dst,)."""
        n_src: int = bilinear_matrix.n_src
        src_field: np.ndarray = np.ones(n_src, dtype=np.float64)

        dst_field: np.ndarray = axis_py.apply_weights(bilinear_matrix, src_field)

        assert dst_field.shape == (bilinear_matrix.n_dst,)

    def test_apply_rejects_wrong_size(self, bilinear_matrix: Any) -> None:
        """apply_weights raises on source array size mismatch."""
        wrong_size: np.ndarray = np.ones(bilinear_matrix.n_src + 5, dtype=np.float64)

        with pytest.raises((ValueError, RuntimeError)):
            axis_py.apply_weights(bilinear_matrix, wrong_size)

    def test_generate_with_dict_config(self, small_src_mesh: Any, small_dst_mesh: Any) -> None:
        """generate_weights accepts a Python dict as RegridConfig."""
        config: dict[str, str] = {"method": "bilinear", "unmapped": "ignore"}
        matrix: Any = axis_py.generate_weights(small_src_mesh, small_dst_mesh, config)

        assert matrix.n_src > 0
        assert matrix.n_dst > 0
        assert matrix.nnz > 0


# ─── Test: batch_apply with 2-D numpy arrays (Req 12.6) ──────────────────────


class TestBatchApply:
    """Validates Requirement 12.6: batch_apply() with 2-D numpy arrays."""

    def test_batch_apply_matches_individual(self, bilinear_matrix: Any) -> None:
        """
        batch_apply on a (n_src, n_vars) array must produce the same result
        as calling apply_weights individually for each variable column.
        """
        n_src: int = bilinear_matrix.n_src
        n_dst: int = bilinear_matrix.n_dst
        n_vars: int = 3

        # Create source data: 3 distinct fields
        rng = np.random.default_rng(seed=12345)
        src_2d: np.ndarray = rng.standard_normal((n_src, n_vars))

        # Batch apply
        dst_batch: np.ndarray = axis_py.batch_apply(bilinear_matrix, src_2d)

        # Verify shape
        assert dst_batch.shape == (n_dst, n_vars)

        # Compare against individual applies
        for v in range(n_vars):
            dst_single: np.ndarray = axis_py.apply_weights(bilinear_matrix, src_2d[:, v].copy())
            np.testing.assert_allclose(
                dst_batch[:, v],
                dst_single,
                rtol=1e-12,
                err_msg=f"batch_apply mismatch for variable {v}",
            )

    def test_batch_apply_constant_fields(self, bilinear_matrix: Any) -> None:
        """
        batch_apply on constant fields must produce constant results
        (validates partition-of-unity across all variables).
        """
        n_src: int = bilinear_matrix.n_src
        n_dst: int = bilinear_matrix.n_dst  # noqa: F841

        # Each variable is a different constant
        constants: list[float] = [1.0, -7.5, 100.0, 0.0]
        src_2d: np.ndarray = np.column_stack([np.full(n_src, c, dtype=np.float64) for c in constants])

        dst_batch: np.ndarray = axis_py.batch_apply(bilinear_matrix, src_2d)

        for v, c in enumerate(constants):
            np.testing.assert_allclose(
                dst_batch[:, v],
                c,
                rtol=1e-12,
                err_msg=f"batch_apply constant field {v} (={c}) not preserved",
            )

    def test_batch_apply_rejects_wrong_extent(self, bilinear_matrix: Any) -> None:
        """batch_apply raises on source array shape[0] mismatch."""
        wrong_shape: np.ndarray = np.ones((bilinear_matrix.n_src + 3, 2), dtype=np.float64)

        with pytest.raises((ValueError, RuntimeError)):
            axis_py.batch_apply(bilinear_matrix, wrong_shape)


# ─── Test: weight cache to_bytes/from_bytes round-trip (Req 12.5) ─────────────


class TestWeightCacheRoundTrip:
    """Validates Requirement 12.5: weight cache serialize/deserialize."""

    def test_roundtrip_preserves_matrix(self, bilinear_matrix: Any) -> None:
        """
        Serializing a matrix with to_bytes() and deserializing with
        Matrix.from_bytes() must produce an equivalent matrix that
        yields identical apply results.
        """
        n_src: int = bilinear_matrix.n_src

        # Serialize
        blob: bytes = bilinear_matrix.to_bytes()
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        # Deserialize
        restored: Any = axis_py.Matrix.from_bytes(blob)

        # Structural equality
        assert restored.n_src == bilinear_matrix.n_src
        assert restored.n_dst == bilinear_matrix.n_dst
        assert restored.nnz == bilinear_matrix.nnz

        # Functional equality: apply both to same field, compare results
        rng = np.random.default_rng(seed=99)
        src_field: np.ndarray = rng.standard_normal(n_src)

        dst_original: np.ndarray = axis_py.apply_weights(bilinear_matrix, src_field)
        dst_restored: np.ndarray = axis_py.apply_weights(restored, src_field)

        # Should be identical within double precision machine tolerance (same weights)
        np.testing.assert_allclose(
            dst_original,
            dst_restored,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Round-tripped matrix produces different apply results",
        )

    def test_to_bytes_returns_bytes(self, bilinear_matrix: Any) -> None:
        """to_bytes() returns a Python bytes object."""
        blob: bytes = bilinear_matrix.to_bytes()
        assert isinstance(blob, bytes)

    def test_from_bytes_invalid_data_raises(self) -> None:
        """from_bytes() raises on garbage input."""
        garbage: bytes = b"\x00\x01\x02\x03" * 10
        with pytest.raises(RuntimeError):
            axis_py.Matrix.from_bytes(garbage)


# ─── Test: named mesh generation ──────────────────────────────────────────────


class TestNamedMeshGeneration:
    """Test make_named_mesh() for well-known grid names."""

    def test_make_named_mesh_o32(self):
        """
        make_named_mesh("O32") produces an octahedral reduced Gaussian
        grid with expected cell count.

        O32 has 32 latitude bands with variable longitude points per band.
        Total cells ≈ 2 * 32 * (20 + 64) / 2 = ~2688 (approximate).
        We just verify it's a non-trivial mesh with > 0 cells.
        """
        mesh = axis_py.make_named_mesh("O32")

        assert mesh.n_cells > 0
        assert mesh.n_nodes > 0
        # Octahedral O32 should have meaningful resolution
        assert mesh.n_cells > 100

    def test_make_named_mesh_properties(self):
        """Named meshes expose n_cells and n_nodes properties."""
        mesh = axis_py.make_named_mesh("O32")

        # n_nodes should be >= n_cells for a well-formed mesh
        # (each cell has at least one node, most are shared)
        assert mesh.n_nodes >= 1
        assert mesh.n_cells >= 1


# ─── Test: Exposed Matrix properties and Core utilities ───────────────────────


class TestMatrixPropertiesAndCoreUtilities:
    """Validates exposed C++ properties and utility functions."""

    def test_matrix_exposed_properties(self, bilinear_matrix):
        """Verify that Matrix exposes factor_list, index lists, areas, and fractions."""
        assert isinstance(bilinear_matrix.factor_list, np.ndarray)
        assert isinstance(bilinear_matrix.factor_col, np.ndarray)
        assert isinstance(bilinear_matrix.factor_row, np.ndarray)
        assert isinstance(bilinear_matrix.frac_a, np.ndarray)
        assert isinstance(bilinear_matrix.frac_b, np.ndarray)
        assert isinstance(bilinear_matrix.area_a, np.ndarray)
        assert isinstance(bilinear_matrix.area_b, np.ndarray)

        assert len(bilinear_matrix.factor_list) == bilinear_matrix.nnz
        assert len(bilinear_matrix.factor_col) == bilinear_matrix.nnz
        assert len(bilinear_matrix.factor_row) == bilinear_matrix.nnz
        assert len(bilinear_matrix.frac_a) == bilinear_matrix.n_src
        assert len(bilinear_matrix.frac_b) == bilinear_matrix.n_dst
        assert len(bilinear_matrix.area_a) == bilinear_matrix.n_src
        assert len(bilinear_matrix.area_b) == bilinear_matrix.n_dst

    def test_grid_detectors(self):
        """Verify detect_regular_grid and detect_rectilinear_grid correctly identify grids."""
        # Create a 10x5 regular lat-lon mesh
        mesh = axis_py.make_regular_mesh(10, 5, 0.0, -90.0, 36.0, 36.0)

        # 1. Regular grid detection
        reg_info = axis_py.detect_regular_grid(mesh)
        assert isinstance(reg_info, dict)
        assert reg_info["is_regular"] is True
        assert reg_info["ni"] == 10
        assert reg_info["nj"] == 5
        assert np.isclose(reg_info["delta_lon"], 36.0)
        assert np.isclose(reg_info["delta_lat"], 36.0)

        # 2. Rectilinear grid detection
        rect_info = axis_py.detect_rectilinear_grid(mesh)
        assert isinstance(rect_info, dict)
        assert rect_info["is_rectilinear"] is True
        assert rect_info["ni"] == 10
        assert rect_info["nj"] == 5
        assert isinstance(rect_info["unique_lons"], list)
        assert len(rect_info["unique_lons"]) == 11
        assert isinstance(rect_info["unique_lats"], list)
        assert len(rect_info["unique_lats"]) == 6

    def test_adjust_by_fraction(self):
        """Verify adjust_by_fraction divides in-place."""
        dst = np.array([10.0, 20.0, 0.0, 5.0], dtype=np.float64)
        frac = np.array([0.5, 1.0, 0.0, 0.25], dtype=np.float64)

        # Perform in-place adjustment
        axis_py.adjust_by_fraction(dst, frac)

        # Expect:
        # 10.0 / 0.5  = 20.0
        # 20.0 / 1.0  = 20.0
        # 0.0 / 0.0   = 0.0 or nan (depending on C++ handling, usually leaves alone if frac is 0, let's verify)
        # 5.0 / 0.25  = 20.0
        assert dst[0] == 20.0
        assert dst[1] == 20.0
        assert dst[3] == 20.0
