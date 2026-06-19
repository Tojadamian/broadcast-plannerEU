"""Generate bundled synthetic example rasters for offline labs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from broadcast_planner.core.crs import ANALYSIS_CRS
from broadcast_planner.core.regions import DEFAULT_REGION, EXAMPLES_DIR, RegionConfig


def generate_example_rasters(output_dir: Path, region: RegionConfig | None = None) -> None:
    region = region or DEFAULT_REGION
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = region.bounds
    res = region.resolution_m
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)

    xs = np.linspace(minx, maxx, width)
    ys = np.linspace(maxy, miny, height)
    xx, yy = np.meshgrid(xs, ys)

    # Synthetic terrain: rolling hills + urban bump near center
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    dem = (
        40.0
        + 25.0 * np.sin((xx - minx) / 25_000.0)
        + 15.0 * np.cos((yy - miny) / 18_000.0)
        + 30.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (25_000.0**2)))
    )

    clutter = np.full((height, width), 10, dtype=np.int16)
    clutter[dem > np.percentile(dem, 70)] = 20
    clutter[dem > np.percentile(dem, 88)] = 30

    pop = (
        50.0
        + 400.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (20_000.0**2)))
        + 80.0 * np.exp(-(((xx - (cx + 30_000)) ** 2 + (yy - (cy + 20_000)) ** 2) / (15_000.0**2)))
    )
    pop = np.maximum(pop, 5.0)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": ANALYSIS_CRS,
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(output_dir / "dem.tif", "w", dtype="float32", **profile) as dst:
        dst.write(dem.astype(np.float32), 1)

    with rasterio.open(output_dir / "clutter.tif", "w", dtype="int16", **profile) as dst:
        dst.write(clutter, 1)

    with rasterio.open(output_dir / "population.tif", "w", dtype="float32", **profile) as dst:
        dst.write(pop.astype(np.float32), 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic example geodata")
    parser.add_argument("--output", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args(argv)
    generate_example_rasters(args.output)
    print(f"Wrote example rasters to {args.output}")


if __name__ == "__main__":
    main()
