"""ITU-R P.1546 propagation grid engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from broadcast_planner.core.grids import GridSpec
from broadcast_planner.core.sites import Site
from broadcast_planner.propagation.clutter import area_from_class, r2_from_class

try:
    from Py1546 import P1546

    HAS_PY1546 = True
except ImportError:
    HAS_PY1546 = False


EARTH_RADIUS_M = 6_371_000.0
DTT_SIGMA_DB = 5.5


@dataclass
class PropagationResult:
    field_strength_dbuv_m: np.ndarray
    transmission_loss_db: np.ndarray
    site_id: str


def haversine_m(x1: float, y1: float, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    dx = x2 - x1
    dy = y2 - y1
    return np.hypot(dx, dy)


def effective_height(site: Site, dem_at_site: float) -> float:
    ground = dem_at_site if np.isfinite(dem_at_site) else 0.0
    return site.antenna_height_m + ground


def _p1546_point(
    frequency_mhz: float,
    distance_km: float,
    heff: float,
    h2: float,
    r2: float,
    area: str,
    time_percent: float = 50.0,
) -> float:
    if distance_km < 0.001:
        distance_km = 0.001
    if HAS_PY1546:
        d_v = np.array([distance_km])
        path_c = np.array(["Land"])
        lb = P1546.bt_loss(
            f=frequency_mhz,
            t=time_percent,
            heff=heff,
            h2=h2,
            R2=r2,
            area=area,
            d_v=d_v,
            path_c=path_c,
            pathinfo=0,
        )
        return float(lb[0])
    return _fallback_loss_db(frequency_mhz, distance_km, heff, h2, r2, area)


def _fallback_loss_db(
    frequency_mhz: float,
    distance_km: float,
    heff: float,
    h2: float,
    r2: float,
    area: str,
) -> float:
    """Simplified P.1546-like curve for offline / fast grid use."""
    del r2, area
    fspl = 32.45 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(max(distance_km, 0.001))
    horizon_factor = max(0.0, (heff + h2) / 1000.0)
    extra = 10.0 * math.log10(1.0 + distance_km / max(horizon_factor, 0.5))
    return fspl + extra


def erp_to_field_strength_dbuv_m(erp_kw: float, loss_db: float) -> float:
    erp_w = erp_kw * 1000.0
    e_mv_m = math.sqrt(30.0 * erp_w) / 1.0
    e_dbuv_m = 20.0 * math.log10(e_mv_m * 1e6)
    return e_dbuv_m - loss_db


@lru_cache(maxsize=256)
def _area_label_at_index(class_code: int) -> str:
    from broadcast_planner.propagation.clutter import CLUTTER_MAP

    return CLUTTER_MAP.get(class_code, ("Rural", 10.0))[0]


def compute_site_field_strength(
    site: Site,
    grid: GridSpec,
    dem: np.ndarray,
    clutter_classes: np.ndarray,
    time_percent: float = 50.0,
    max_distance_km: float = 120.0,
) -> PropagationResult:
    xx, yy = grid.mesh_xy()
    dist_m = haversine_m(site.x, site.y, xx, yy)
    dist_km = dist_m / 1000.0

    dem_site = float(dem.flat[np.argmin((xx - site.x) ** 2 + (yy - site.y) ** 2)])
    heff = effective_height(site, dem_site)
    r2_grid = r2_from_class(clutter_classes)
    area_grid = area_from_class(clutter_classes)

    loss = np.full(grid.shape, np.nan, dtype=np.float64)
    field = np.full(grid.shape, np.nan, dtype=np.float64)

    mask = dist_km <= max_distance_km
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        d_km = dist_km[r, c]
        h2 = float(dem[r, c]) if np.isfinite(dem[r, c]) else 0.0
        r2 = float(r2_grid[r, c])
        area = str(area_grid[r, c])
        lb = _p1546_point(site.frequency_mhz, float(d_km), heff, h2, r2, area, time_percent)
        loss[r, c] = lb
        field[r, c] = erp_to_field_strength_dbuv_m(site.erp_kw, lb)

    return PropagationResult(
        field_strength_dbuv_m=field,
        transmission_loss_db=loss,
        site_id=site.id,
    )


def compute_site_field_strength_fast(
    site: Site,
    grid: GridSpec,
    dem: np.ndarray,
    clutter_classes: np.ndarray,
    time_percent: float = 50.0,
    max_distance_km: float = 120.0,
) -> PropagationResult:
    """Vectorized fast path using fallback model for whole grid."""
    xx, yy = grid.mesh_xy()
    dist_m = haversine_m(site.x, site.y, xx, yy)
    dist_km = np.maximum(dist_m / 1000.0, 0.001)

    dem_site_idx = np.unravel_index(
        np.argmin((xx - site.x) ** 2 + (yy - site.y) ** 2), xx.shape
    )
    heff = effective_height(site, float(dem[dem_site_idx]))

    fspl = 32.45 + 20.0 * np.log10(site.frequency_mhz) + 20.0 * np.log10(dist_km)
    horizon = np.maximum((heff + np.nan_to_num(dem, nan=0.0)) / 1000.0, 0.5)
    extra = 10.0 * np.log10(1.0 + dist_km / horizon)
    loss = fspl + extra

    erp_w = site.erp_kw * 1000.0
    e_dbuv = 20.0 * np.log10(np.sqrt(30.0 * erp_w) * 1e6) - loss

    mask = dist_km * 1000.0 <= max_distance_km * 1000.0
    loss = np.where(mask, loss, np.nan)
    e_dbuv = np.where(mask, e_dbuv, np.nan)

    return PropagationResult(field_strength_dbuv_m=e_dbuv, transmission_loss_db=loss, site_id=site.id)


def compute_multi_site_fields(
    sites,
    grid: GridSpec,
    dem: np.ndarray,
    clutter_classes: np.ndarray,
    fast: bool = True,
    **kwargs,
) -> dict[str, PropagationResult]:
    fn = compute_site_field_strength_fast if fast else compute_site_field_strength
    return {site.id: fn(site, grid, dem, clutter_classes, **kwargs) for site in sites}


def best_server_map(results: dict[str, PropagationResult]) -> tuple[np.ndarray, np.ndarray]:
    ids = list(results.keys())
    stack = np.stack([results[sid].field_strength_dbuv_m for sid in ids], axis=0)
    with np.errstate(all="ignore"):
        best_idx = np.nanargmax(stack, axis=0)
    best_field = np.take_along_axis(stack, best_idx[None, ...], axis=0)[0]
    server_map = np.full(stack.shape[1:], -1, dtype=np.int16)
    valid = np.any(np.isfinite(stack), axis=0)
    server_map[valid] = best_idx[valid]
    id_lookup = {i: ids[i] for i in range(len(ids))}
    return best_field, server_map, id_lookup


def location_probability_coverage(
    field_dbuv_m: np.ndarray,
    threshold_dbuv_m: float,
    sigma_db: float = DTT_SIGMA_DB,
    target_lp_percent: float = 95.0,
) -> np.ndarray:
    """Fraction of locations above threshold given log-normal variability."""
    from scipy.stats import norm

    z = (threshold_dbuv_m - field_dbuv_m) / sigma_db
    lp = (1.0 - norm.cdf(z)) * 100.0
    return lp >= target_lp_percent
