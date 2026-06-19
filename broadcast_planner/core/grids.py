"""Analysis grid specification and construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.transform import from_origin

from broadcast_planner.core.crs import ANALYSIS_CRS
from broadcast_planner.core.regions import RegionConfig


@dataclass(frozen=True)
class GridSpec:
    transform: rasterio.Affine
    width: int
    height: int
    crs: str = ANALYSIS_CRS
    resolution_m: float = 250.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def xs(self) -> np.ndarray:
        cols = np.arange(self.width)
        xs = self.transform.c + cols * self.transform.a + self.transform.a / 2.0
        return xs

    def ys(self) -> np.ndarray:
        rows = np.arange(self.height)
        ys = self.transform.f + rows * self.transform.e + self.transform.e / 2.0
        return ys

    def mesh_xy(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.xs()
        ys = self.ys()
        return np.meshgrid(xs, ys)

    def write_geotiff(self, path, array: np.ndarray, nodata: float = -9999.0) -> None:
        profile = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 1,
            "dtype": array.dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "compress": "deflate",
        }
        data = np.where(np.isfinite(array), array, nodata).astype(array.dtype)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)


def make_grid(region: RegionConfig, resolution_m: float | None = None) -> GridSpec:
    res = resolution_m or region.resolution_m
    minx, miny, maxx, maxy = region.bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    return GridSpec(transform=transform, width=width, height=height, resolution_m=res)


def read_raster(path) -> tuple[np.ndarray, GridSpec]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        spec = GridSpec(
            transform=src.transform,
            width=src.width,
            height=src.height,
            crs=src.crs.to_string() if src.crs else ANALYSIS_CRS,
            resolution_m=abs(src.transform.a),
        )
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
    return data, spec


def align_to_grid(source: np.ndarray, source_spec: GridSpec, target: GridSpec) -> np.ndarray:
    if source_spec.shape == target.shape and source_spec.transform == target.transform:
        return source
    from rasterio.warp import reproject, Resampling

    dest = np.full(target.shape, np.nan, dtype=np.float64)
    reproject(
        source=source,
        destination=dest,
        src_transform=source_spec.transform,
        src_crs=source_spec.crs,
        dst_transform=target.transform,
        dst_crs=target.crs,
        resampling=Resampling.bilinear,
    )
    return dest


def maybe_align_layer(
    data: np.ndarray,
    target: GridSpec,
    source: GridSpec | None = None,
) -> np.ndarray:
    """Resample a raster layer onto the analysis grid when shapes differ."""
    if data.shape == target.shape:
        return data
    if source is None:
        raise ValueError(
            f"Layer shape {data.shape} does not match grid shape {target.shape}. "
            "Pass the source GridSpec returned by read_raster()."
        )
    return align_to_grid(data, source, target)


def align_layers_to_grid(
    grid: GridSpec,
    dem: np.ndarray,
    clutter: np.ndarray,
    population: np.ndarray,
    source_grid: GridSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align DEM, clutter, and population rasters to the analysis grid."""
    return (
        maybe_align_layer(dem, grid, source_grid),
        maybe_align_layer(clutter, grid, source_grid),
        maybe_align_layer(population, grid, source_grid),
    )
