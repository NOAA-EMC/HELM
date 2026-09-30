# SPDX-License-Identifier: Apache-2.0
"""AXIS command-line interface (FR-035/FR-036, contracts/cli.md).

Subcommands
-----------
``axis-regrid regrid  -s SRC -t DST -o OUT [-v VAR ...] [options]``
``axis-regrid weights -s SRC -t DST -o OUT [--format esmf|native]``
``axis-regrid list    methods|grids``

Exit codes (FR-036): ``0`` success, ``1`` runtime error (an actionable
``AxisError`` message on stderr, no traceback unless ``--debug``), ``2``
argument-parsing misuse (argparse's own convention).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from importlib import metadata
from typing import TYPE_CHECKING

from .errors import AxisError
from .types import Method

if TYPE_CHECKING:
    from .grid import Grid
    from .regridder import Regridder

__all__ = ["main"]

# The six public regridding methods, in canonical order (FR-002).
_METHODS = [m.value for m in Method]


# ─── grid loading ────────────────────────────────────────────────────────────


def _require_file(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise AxisError(f"{label} file does not exist: {path!r}")
    return path


def _load_grid(path: str, label: str) -> Grid:
    """Resolve a path to a Grid: a named-grid string (e.g. 'C48') or NetCDF."""
    from .grid import Grid

    if not path.endswith((".nc", ".nc4", ".netcdf")) and not os.path.exists(path):
        # Not a file path — treat as a named grid identifier.
        try:
            return Grid.from_named(path)
        except AxisError:
            raise AxisError(
                f"{label} {path!r} is neither an existing NetCDF file nor a recognised named grid (e.g. 'C48', 'O96')"
            ) from None
    _require_file(path, label)
    import xarray as xr

    ds = xr.open_dataset(path)
    return Grid.from_xarray(ds)


# ─── regrid ──────────────────────────────────────────────────────────────────


def _load_precomputed(path: str, source: Grid, target: Grid) -> Regridder:
    """Rebuild a fitted Regridder from a weight file written by either the
    Python API or the CLI itself (US7-AC2 round-trip, both directions)."""
    from .regridder import Regridder

    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic == b"AXISW1":
        return Regridder.load_weights(path, source=source, target=target)
    return Regridder.from_esmf(path, source=source, target=target)


def _cmd_regrid(args: argparse.Namespace) -> int:
    import xarray as xr

    from .regridder import Regridder

    source = _load_grid(args.source, "source")
    target = _load_grid(args.target, "target")

    if args.weights:
        rg = _load_precomputed(_require_file(args.weights, "weights"), source, target)
    else:
        rg = Regridder(
            source,
            target,
            method=args.method,
            norm=args.norm,
            unmapped=args.unmapped,
            skipna=args.skipna,
            na_thres=args.na_thres,
            periodic=args.periodic,
            line_type=args.line_type,
        )

    ds = xr.open_dataset(_require_file(args.source, "source"))
    if args.variables:
        missing = [v for v in args.variables if v not in ds.variables]
        if missing:
            available = ", ".join(sorted(str(v) for v in ds.data_vars)) or "<none>"
            raise AxisError(f"variable(s) {', '.join(missing)} not found in {args.source!r}; available data variables: {available}")
        to_regrid = ds[args.variables]
    else:
        names = list(ds.data_vars)
        if not names:
            raise AxisError(f"source {args.source!r} contains no data variables to regrid")
        to_regrid = ds[names]

    out = rg(to_regrid, keep_attrs=args.keep_attrs)
    if not args.keep_attrs:
        for name in out.data_vars:
            out[name].attrs = {}
    out.to_netcdf(args.output)
    return 0


# ─── weights ─────────────────────────────────────────────────────────────────


def _cmd_weights(args: argparse.Namespace) -> int:
    from .regridder import Regridder

    source = _load_grid(args.source, "source")
    target = _load_grid(args.target, "target")
    rg = Regridder(
        source,
        target,
        method=args.method,
        norm=args.norm,
        unmapped=args.unmapped,
        skipna=args.skipna,
        na_thres=args.na_thres,
        periodic=args.periodic,
        line_type=args.line_type,
    )
    if args.format == "native":
        rg.save_weights(args.output)
    else:
        rg.to_esmf(args.output)
    return 0


# ─── list ────────────────────────────────────────────────────────────────────


def _cmd_list(args: argparse.Namespace) -> int:
    if args.what == "methods":
        sys.stdout.write("\n".join(_METHODS) + "\n")
        return 0
    # grids: enumerate the named-grid family prefixes from the engine (FR-039).
    from ._core import list_named_grid_families

    families = list_named_grid_families()
    meaning = {
        "C": "cubed-sphere",
        "F": "regular Gaussian",
        "G": "NOAA GRIB",
        "N": "reduced Gaussian (ECMWF octahedral)",
        "O": "octahedral reduced Gaussian",
        "R": "regular lat-lon",
    }
    for fam in families:
        sys.stdout.write(f"{fam}  {meaning.get(fam, 'named grid family')}\n")
    return 0


# ─── parser ──────────────────────────────────────────────────────────────────


def _add_regrid_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-s", "--source", required=True, help="Source NetCDF file or named grid (e.g. C48).")
    p.add_argument("-t", "--target", required=True, help="Target NetCDF file or named grid (e.g. O96).")
    p.add_argument("-o", "--output", required=True, help="Output path.")
    p.add_argument("-v", "--variables", action="append", metavar="VAR", help="Variable to regrid (repeatable).")
    p.add_argument("-m", "--method", default="bilinear", choices=_METHODS, help="Regridding method.")
    p.add_argument("--norm", default="frac-area", choices=["frac-area", "dst-area"], help="Conservative norm.")
    p.add_argument("--unmapped", default="nan", choices=["nan", "mask", "error"], help="Unmapped-cell policy.")
    p.add_argument("--periodic", action="store_true", help="Treat longitude as periodic.")
    p.add_argument("--skipna", action="store_true", help="Renormalise weights over valid (non-NaN) sources.")
    p.add_argument("--na-thres", type=float, default=1.0, dest="na_thres", help="NaN-fraction threshold [0,1].")
    p.add_argument("--line-type", default=None, choices=["great-circle", "cartesian"], dest="line_type")
    p.add_argument("--debug", action="store_true", help="Print full tracebacks on error.")


def _build_parser() -> argparse.ArgumentParser:
    try:
        version = metadata.version("axis")
    except metadata.PackageNotFoundError:  # pragma: no cover
        version = "0.0.0"
    parser = argparse.ArgumentParser(prog="axis-regrid", description="AXIS regridding command-line tool.")
    parser.add_argument("--version", action="version", version=f"axis-regrid {version}")
    sub = parser.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("regrid", help="Regrid a NetCDF file onto a target grid.")
    _add_regrid_options(pr)
    pr.add_argument(
        "--weights", default=None, metavar="FILE", help="Apply precomputed weights (native or ESMF file) instead of regenerating."
    )
    pr.add_argument("--keep-attrs", action=argparse.BooleanOptionalAction, default=True, dest="keep_attrs")
    pr.set_defaults(func=_cmd_regrid)

    pw = sub.add_parser("weights", help="Generate and save regridding weights.")
    _add_regrid_options(pw)
    pw.add_argument("--format", default="esmf", choices=["esmf", "native"], help="Weight file format.")
    pw.set_defaults(func=_cmd_weights)

    pl = sub.add_parser("list", help="List available methods or named-grid families.")
    pl.add_argument("what", choices=["methods", "grids"])
    pl.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AxisError as exc:
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            sys.stderr.write(f"axis-regrid: error: {exc}\n")
        return 1
    except (FileNotFoundError, OSError) as exc:
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            sys.stderr.write(f"axis-regrid: error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
