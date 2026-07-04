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
    sites_filename: str = "sites.yaml"  # Region-specific default sites
    test_points: Tuple[Tuple[float, float, str], ...] = ()  # (x, y, name) for coverage testing

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
        return EXAMPLES_DIR / self.sites_filename


DEFAULT_REGION = RegionConfig(
    name="Brussels-Antwerp corridor",
    bounds=(3_900_000.0, 3_080_000.0, 3_950_000.0, 3_160_000.0),
    resolution_m=250.0,
    sites_filename="sites.yaml",
    test_points=(
        (3915000.0, 3100000.0, "Downtown Brussels"),
        (3935000.0, 3120000.0, "Central Antwerp"),
        (3920000.0, 3130000.0, "Mechelen Test"),
    ),
)

REGIONS: Dict[str, RegionConfig] = {
    "brussels_antwerp": DEFAULT_REGION,
    "upper_bavaria": RegionConfig(
        name="Upper Bavaria",
        bounds=(4_400_000.0, 2_740_000.0, 4_480_000.0, 2_820_000.0),
        sites_filename="sites_upper_bavaria.yaml",
        test_points=(
            (4438298.0, 2781422.0, "Munich City Center"),
            (4425000.0, 2765000.0, "Augsburg Test"),
            (4450000.0, 2800000.0, "Ingolstadt Test"),
        ),
    ),
    "tuscany": RegionConfig(
        name="Tuscany",
        bounds=(4_380_000.0, 2_250_000.0, 4_460_000.0, 2_340_000.0),
        sites_filename="sites_tuscany.yaml",
        test_points=(
            (4422352.0, 2296442.0, "Florence City Center"),
            (4410000.0, 2270000.0, "Siena Test"),
            (4400000.0, 2310000.0, "Pisa Test"),
        ),
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
