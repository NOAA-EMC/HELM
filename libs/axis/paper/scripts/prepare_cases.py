"""Prepare common explicit meshes; numerical reference fields are generated natively."""

import argparse
import math
import sys
from pathlib import Path
import xarray as xr
import numpy as np
from artifacts import ROOT, digest, identity, profile, save


def xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def assemble(lon, lat, cells, geometry, parents=None, mask_half=False):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    offsets, indices, areas, centers = [0], [], [], []
    for cell in cells:
        v = xyz(lon[cell], lat[cell])
        center = v.sum(axis=0)
        center /= np.linalg.norm(center)
        orientation = sum(
            np.dot(center, np.cross(v[i], v[(i + 1) % len(v)])) for i in range(len(v))
        )
        if orientation < 0:
            cell = list(reversed(cell))
            v = v[::-1]
        indices.extend(cell)
        offsets.append(len(indices))
        centers.append(
            [math.degrees(math.atan2(center[1], center[0])), math.degrees(math.asin(center[2]))]
        )
        if geometry == "constant_latitude":
            unwrapped = np.unwrap(np.deg2rad(lon[cell]))
            area = np.ptp(unwrapped) * np.ptp(np.sin(np.deg2rad(lat[cell])))
        else:
            area = sum(
                2
                * math.atan2(
                    np.dot(v[0], np.cross(v[i], v[i + 1])),
                    1 + np.dot(v[0], v[i]) + np.dot(v[i], v[i + 1]) + np.dot(v[i + 1], v[0]),
                )
                for i in range(1, len(v) - 1)
            )
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Degenerate cell")
        areas.append(area)
    centers = np.array(centers)
    mask = np.ones(len(cells), np.int32)
    if mask_half:
        mask[np.arange(len(cells)) % 3 == 0] = 0
    return dict(
        lon=lon,
        lat=lat,
        offsets=np.asarray(offsets, np.int32),
        indices=np.asarray(indices, np.int32),
        center_lon=centers[:, 0],
        center_lat=centers[:, 1],
        area=np.asarray(areas),
        mask=mask,
        cell_id=np.arange(len(cells), dtype=np.int32) + 1,
        parent_id=np.asarray(
            parents if parents is not None else np.arange(len(cells)) + 1, np.int32
        ),
    )


def generate(desc, geometry, fetch):
    kind = desc["kind"]
    provenance = {"generator": desc}
    if kind == "polar":
        from polar_mesh import polar_coordinates

        if geometry != "great_circle":
            raise ValueError("Polar cap meshes require great-circle geometry")
        lon, lat, cells, parents = polar_coordinates(desc)
        return assemble(lon, lat, cells, geometry, parents), provenance
    if kind == "mpas":
        path = Path(desc["path"])
        if not path.exists() and fetch:
            sys.path.insert(0, str(ROOT / "libs/axis/benchmarks"))
            from fetch_mpas import fetch_mpas_grid

            path = Path(fetch_mpas_grid(desc.get("mesh", "x1.2562")))
        if not path.exists():
            raise FileNotFoundError(
                f"MPAS input unavailable: {path}; use --fetch or provide the file"
            )
        provenance.update(input_path=str(path.resolve()), sha256=digest(path))
        with xr.open_dataset(path) as f:
            lon = np.rad2deg(f["lonVertex"].values)
            lat = np.rad2deg(f["latVertex"].values)
            cells = [
                list(map(int, row[: int(n)] - 1))
                for row, n in zip(f["verticesOnCell"].values, f["nEdgesOnCell"].values)
            ]
        parents = None
        if desc.get("triangulate"):
            expanded = []
            parents = []
            for parent, cell in enumerate(cells, 1):
                # Convex MPAS cells: shared centroid fan; transformed case identity is explicit.
                v = xyz(lon[cell], lat[cell]).sum(axis=0)
                v /= np.linalg.norm(v)
                k = len(lon)
                lon = np.append(lon, math.degrees(math.atan2(v[1], v[0])))
                lat = np.append(lat, math.degrees(math.asin(v[2])))
                for i in range(len(cell)):
                    expanded.append([k, cell[i], cell[(i + 1) % len(cell)]])
                    parents.append(parent)
            cells = expanded
        return assemble(lon, lat, cells, geometry, parents), provenance
    n = desc["n"]
    m = desc.get("m", n)
    if kind == "lcc":
        import pyproj

        projection = desc.get(
            "projection", "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-96 +datum=WGS84 +units=m"
        )
        transform = pyproj.Transformer.from_crs(projection, "EPSG:4326", always_xy=True)
        x, y = np.meshgrid(np.linspace(-500000, 500000, n + 1), np.linspace(-500000, 500000, m + 1))
        lon, lat = transform.transform(x.ravel(), y.ravel())
        provenance["projection"] = projection
    else:
        x, y = np.meshgrid(
            np.linspace(*desc.get("lon", [-20, 20]), n + 1),
            np.linspace(*desc.get("lat", [-20, 20]), m + 1),
        )
        lon, lat = x.ravel(), y.ravel()
    cells = []
    parents = []
    for j in range(m):
        for i in range(n):
            arguments = j * (n + 1) + i
            quad = [arguments, arguments + 1, arguments + n + 2, arguments + n + 1]
            parent = j * n + i + 1
            if desc.get("triangulate", geometry == "great_circle"):
                cells.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
                parents.extend([parent, parent])
            else:
                cells.append(quad)
                parents.append(parent)
    return assemble(lon, lat, cells, geometry, parents, desc.get("masked", False)), provenance


def write_case(path, case, policy, src, dst):
    attributes = dict(
        schema_version="1",
        case_id=case["id"],
        geometry=case["geometry"],
        field=case["field"],
        norm=case["norm"],
        reference_tolerance=str(policy["reference"]),
        area_tolerance=str(policy["area"]),
        constant_tolerance=str(policy["constant"]),
        conservation_tolerance=str(policy["conservation"]),
        near_zero_tolerance=str(policy["near_zero"]),
        reproduction_tolerance=str(policy["reproduction"]),
        smooth_rms_tolerance=str(policy["smooth_rms"]),
        max_depth=str(policy["max_depth"]),
        max_subtriangles=str(policy["max_subtriangles"]),
    )
    xr.Dataset(attrs=attributes).to_netcdf(path, engine="netcdf4")
    for name, mesh in [("source", src), ("destination", dst)]:
        data = xr.Dataset(
            {key: ((key + "_n",), v) for key, v in mesh.items()},
            attrs={"coordinate_units": "degrees", "winding": "CCW"},
        )
        data.to_netcdf(path, group=name, mode="a", engine="netcdf4")
    for name in ["fields", "options", "reference_policy"]:
        attrs = {"extrapolation": "none", "unmapped": "ignore"} if name == "options" else {}
        xr.Dataset(attrs=attrs).to_netcdf(path, group=name, mode="a", engine="netcdf4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fetch", action="store_true")
    arguments = ap.parse_args()
    settings = profile(arguments.profile)
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for case in settings["cases"]:
        try:
            src, sp = generate(case["source"], case["geometry"], arguments.fetch)
            dst, dp = generate(case["destination"], case["geometry"], arguments.fetch)
            path = out / (case["id"] + ".nc")
            write_case(path, case, settings["tolerances"], src, dst)
            entry = dict(
                case,
                status="prepared",
                fixture=path.name,
                fixture_sha256=digest(path),
                reference_policy=settings["tolerances"],
                provenance={"source": sp, "destination": dp},
                source_cells=len(src["cell_id"]),
                destination_cells=len(dst["cell_id"]),
            )
        except (FileNotFoundError, ValueError) as e:
            entry = dict(
                case,
                status="unavailable" if isinstance(e, FileNotFoundError) else "unsupported",
                cause=str(e),
            )
        entries.append(entry)
    save(
        out / "cases.json",
        {"schema_version": 1, "profile_sha256": identity(settings), "cases": entries},
    )
    print(
        f"Prepared {sum(c['status'] == 'prepared' for c in entries)}/{len(entries)} cases in {out}"
    )
    return 0 if all(c["status"] == "prepared" for c in entries) else 3


if __name__ == "__main__":
    raise SystemExit(main())
