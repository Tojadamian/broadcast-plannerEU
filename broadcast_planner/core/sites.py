"""Transmitter site models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point

from broadcast_planner.core.crs import ANALYSIS_CRS


@dataclass
class Site:
    id: str
    name: str
    x: float
    y: float
    frequency_mhz: float = 474.0
    erp_kw: float = 100.0
    antenna_height_m: float = 300.0
    topology: str = "HPHT"
    is_pilot: bool = False
    launch_delay_us: float = 0.0
    channel: str = "CH21"
    polarization: str = "vertical"
    antenna_gain_dbi: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def erp_dbm(self) -> float:
        return 10.0 * __import__("math").log10(self.erp_kw * 1_000_000.0)

    def to_point(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class SiteCollection:
    sites: List[Site]

    def __len__(self) -> int:
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    def by_id(self, site_id: str) -> Site:
        for site in self.sites:
            if site.id == site_id:
                return site
        raise KeyError(site_id)

    def pilot(self) -> Site:
        for site in self.sites:
            if site.is_pilot:
                return site
        return self.sites[0]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        records = []
        for site in self.sites:
            rec = asdict(site)
            rec["geometry"] = site.to_point()
            records.append(rec)
        return gpd.GeoDataFrame(records, crs=ANALYSIS_CRS)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.sites])

    @classmethod
    def from_geodataframe(cls, gdf: gpd.GeoDataFrame) -> "SiteCollection":
        sites = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            sites.append(
                Site(
                    id=str(row["id"]),
                    name=str(row.get("name", row["id"])),
                    x=float(geom.x),
                    y=float(geom.y),
                    frequency_mhz=float(row.get("frequency_mhz", 474.0)),
                    erp_kw=float(row.get("erp_kw", 100.0)),
                    antenna_height_m=float(row.get("antenna_height_m", 300.0)),
                    topology=str(row.get("topology", "HPHT")),
                    is_pilot=bool(row.get("is_pilot", False)),
                    launch_delay_us=float(row.get("launch_delay_us", 0.0)),
                    channel=str(row.get("channel", "CH21")),
                )
            )
        return cls(sites=sites)


def load_sites(path: Path | str) -> SiteCollection:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    sites = [Site(**entry) for entry in data["sites"]]
    return SiteCollection(sites=sites)


def save_sites(sites: SiteCollection, path: Path | str) -> None:
    path = Path(path)
    payload = {"sites": [asdict(s) for s in sites.sites]}
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
