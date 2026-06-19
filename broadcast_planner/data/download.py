"""Download EU geodata for the case study (optional, requires network)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from broadcast_planner.core.crs import ANALYSIS_CRS
from broadcast_planner.core.regions import CACHE_DIR, DEFAULT_REGION
from broadcast_planner.data.prepare_examples import generate_example_rasters


def download_copernicus_dem(region=DEFAULT_REGION, output_dir: Path | None = None) -> Path:
    """Placeholder for Copernicus DEM download — generates synthetic data if API unavailable."""
    output_dir = output_dir or CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_example_rasters(output_dir, region)
    return output_dir / "dem.tif"


def download_land_cover(region=DEFAULT_REGION, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "clutter.tif").exists():
        generate_example_rasters(output_dir, region)
    return output_dir / "clutter.tif"


def download_population(region=DEFAULT_REGION, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "population.tif").exists():
        generate_example_rasters(output_dir, region)
    return output_dir / "population.tif"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download or prepare EU case-study geodata")
    parser.add_argument("--region", default="brussels_antwerp")
    parser.add_argument("--output", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    from broadcast_planner.core.regions import get_region

    region = get_region(args.region)
    args.output.mkdir(parents=True, exist_ok=True)
    generate_example_rasters(args.output, region)
    print(f"Prepared geodata in {args.output}")
    print("Note: For production Copernicus/Eurostat tiles, replace generate_example_rasters with STAC/API fetch.")


if __name__ == "__main__":
    main()
