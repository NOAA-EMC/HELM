"""Compare independently generated analytical references using xarray.

Generate matching NetCDF files with the production reference worker and a
separate worker built with doubled Gauss orders before invoking this script.
This checks observed convergence; it does not establish interval bounds.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from artifacts import digest, save


def compare_reference(original, refined, tolerance):
    """Check both source and destination averages, retaining failed evidence."""
    record = {
        "case_id": original.stem,
        "original_reference_sha256": digest(original),
        "refined_reference_sha256": digest(refined),
        "passed": True,
    }
    with xr.open_dataset(original) as first, xr.open_dataset(refined) as second:
        for variable, uncertainty in (("src", "source_uncertainty"), ("dst", "uncertainty")):
            left, right = xr.align(first[variable], second[variable], join="exact")
            if left.shape != right.shape:
                raise ValueError(f"Reference shape differs: {original.name}:{variable}")
            difference = np.abs(left.values - right.values)
            if not np.all(np.isfinite(difference)):
                raise ValueError(f"Nonfinite reference comparison: {original.name}:{variable}")
            bound = first[uncertainty].values + second[uncertainty].values
            maximum = float(np.max(difference))
            record[variable] = {
                "maximum_absolute_difference": maximum,
                "within_combined_empirical_estimates": bool(np.all(difference <= bound + 1e-12)),
            }
            record["passed"] &= maximum <= tolerance
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--refined-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=5e-7)
    arguments = parser.parse_args()
    originals = sorted(arguments.reference_dir.glob("*.nc"))
    refined = sorted(arguments.refined_dir.glob("*.nc"))
    if not originals or {path.name for path in originals} != {path.name for path in refined}:
        raise ValueError("Reference folders must contain identical, nonempty NetCDF inventories")
    records = [
        compare_reference(path, arguments.refined_dir / path.name, arguments.absolute_tolerance)
        for path in originals
    ]
    passed = all(record["passed"] for record in records)
    save(
        arguments.output,
        {
            "schema_version": 1,
            "status": "pass" if passed else "fail",
            "description": "Observed reference convergence under independently doubled Gauss orders",
            "absolute_tolerance": arguments.absolute_tolerance,
            "generator_sha256": digest(Path(__file__)),
            "records": records,
        },
    )
    print(f"Reference order comparison: {'pass' if passed else 'fail'} ({len(records)} cases)")
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
