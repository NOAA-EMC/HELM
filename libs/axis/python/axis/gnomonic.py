# SPDX-License-Identifier: Apache-2.0
"""Gnomonic tangent plane projection utilities."""

from . import axis_py

forward = axis_py.gnomonic_forward
inverse = axis_py.gnomonic_inverse
bilinear_weights = axis_py.bilinear_weights

__all__ = ["forward", "inverse", "bilinear_weights"]
