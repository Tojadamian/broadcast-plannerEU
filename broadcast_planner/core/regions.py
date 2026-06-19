"""Region definitions for EU case studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


@dataclass(frozen=True)
class RegionConfig:
    name: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy in EPSG:3035
    resolution_m: float = 250.0

    @property
    def width_m(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height_m(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def dem_path(self, cached: bool = True) -> Path:
        if cached and (CACHE_DIR / "dem.tif").exists():
            return CACHE_DIR / "dem.tif"
        return EXAMPLES_DIR / "dem.tif"

    def clutter_path(self, cached: bool = True) -> Path:
        if cached and (CACHE_DIR / "clutter.tif").exists():
            return CACHE_DIR / "clutter.tif"
        return EXAMPLES_DIR / "clutter.tif"

    def population_path(self, cached: bool = True) -> Path:
        if cached and (CACHE_DIR / "population.tif").exists():
            return CACHE_DIR / "population.tif"
        return EXAMPLES_DIR / "population.tif"

    def sites_path(self) -> Path:
        return EXAMPLES_DIR / "sites.yaml"


DEFAULT_REGION = RegionConfig(
    name="Brussels-Antwerp corridor",
    bounds=(4_720_000.0, 2_260_000.0, 4_820_000.0, 2_360_000.0),
    resolution_m=250.0,
)

REGIONS: Dict[str, RegionConfig] = {
    "brussels_antwerp": DEFAULT_REGION,
    "upper_bavaria": RegionConfig(
        name="Upper Bavaria",
        bounds=(4_440_000.0, 5_300_000.0, 4_540_000.0, 5_400_000.0),
    ),
    "tuscany": RegionConfig(
        name="Tuscany",
        bounds=(4_550_000.0, 4_750_000.0, 4_650_000.0, 4_850_000.0),
    ),
}


def get_region(key: str = "brussels_antwerp") -> RegionConfig:
    return REGIONS.get(key, DEFAULT_REGION)


def load_region_bounds_yaml(path: Path | None = None) -> RegionConfig:
    path = path or EXAMPLES_DIR / "region_bounds.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return RegionConfig(
        name=data["name"],
        bounds=tuple(data["bounds"]),
        resolution_m=data.get("resolution_m", 250.0),
    )
