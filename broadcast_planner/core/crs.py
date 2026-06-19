"""Coordinate reference helpers."""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point

ANALYSIS_CRS = "EPSG:3035"
DISPLAY_CRS = "EPSG:4326"

_to_analysis = Transformer.from_crs(DISPLAY_CRS, ANALYSIS_CRS, always_xy=True)
_to_display = Transformer.from_crs(ANALYSIS_CRS, DISPLAY_CRS, always_xy=True)


def to_analysis(lon: float, lat: float) -> Tuple[float, float]:
    return _to_analysis.transform(lon, lat)


def to_display(x: float, y: float) -> Tuple[float, float]:
    return _to_display.transform(x, y)


def gdf_to_display(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have CRS set")
    return gdf.to_crs(DISPLAY_CRS)


def point_display(x: float, y: float) -> Point:
    lon, lat = to_display(x, y)
    return Point(lon, lat)
