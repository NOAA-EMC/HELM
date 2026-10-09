"""Export the focused Atlantic experiment's paired field and percentage-error map."""

import argparse
from pathlib import Path
from artifacts import load, save
from publication import bundle
from make_comparison_figures import relative_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, records = bundle(args.results)
    quality = [r for r in records if r["mode"] == "quality"]
    if len(quality) != 2 or any(r["matching"] != "matched" for r in quality):
        raise ValueError("Detail figure requires exactly one matched AXIS/ESMC pair")
    axis = next(r for r in quality if r["engine"] == "axis")
    esmc = next(r for r in quality if r["engine"] == "esmc")
    products = relative_maps(axis, esmc, output, atlantic_detail=True)
    path = output / "artifacts.json"
    manifest = load(path)
    manifest["products"]["gulfstream_detail"] = products
    save(path, manifest)
    print("Generated focused Atlantic Gulf Stream detail in PDF/SVG/PNG")


if __name__ == "__main__":
    main()
