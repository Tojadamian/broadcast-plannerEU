"""Candidate site generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union

from broadcast_planner.core.regions import RegionConfig
from broadcast_planner.core.sites import Site, SiteCollection

@dataclass
class CandidateConfig:
    spacing_m: float = 10_000.0
    min_distance_from_existing_m: float = 15_000.0
    max_candidates: int = 200


def generate_candidates(
    region: RegionConfig,
    existing: SiteCollection,
    config: CandidateConfig | None = None,
    exclusion_zones: List[Tuple[float, float, float]] | None = None,
) -> List[Tuple[float, float]]:
    cfg = config or CandidateConfig()
    minx, miny, maxx, maxy = region.bounds
    xs = np.arange(minx + cfg.spacing_m / 2, maxx, cfg.spacing_m)
    ys = np.arange(miny + cfg.spacing_m / 2, maxy, cfg.spacing_m)
    region_box = box(minx, miny, maxx, maxy)
    exclusions = []
    if exclusion_zones:
        for x, y, radius in exclusion_zones:
            exclusions.append(Point(x, y).buffer(radius))
    exclusion_geom = unary_union(exclusions) if exclusions else None

    existing_coords = [(s.x, s.y) for s in existing.sites]
    candidates: List[Tuple[float, float]] = []
    for x in xs:
        for y in ys:
            if len(candidates) >= cfg.max_candidates:
                return candidates
            p = Point(x, y)
            if not region_box.contains(p):
                continue
            if exclusion_geom is not None and exclusion_geom.contains(p):
                continue
            too_close = any(
                np.hypot(x - ex, y - ey) < cfg.min_distance_from_existing_m
                for ex, ey in existing_coords
            )
            if too_close:
                continue
            candidates.append((float(x), float(y)))
    return candidates


def candidate_to_site(x: float, y: float, site_id: str, topology: str = "MPMT") -> Site:
    from broadcast_planner.fiveg_bc.models import TOPOLOGY_PRESETS

    preset = TOPOLOGY_PRESETS.get(topology, TOPOLOGY_PRESETS["MPMT"])
    return Site(
        id=site_id,
        name=f"Candidate {site_id}",
        x=x,
        y=y,
        erp_kw=preset["erp_kw"],
        antenna_height_m=preset["height_m"],
        topology=topology,
    )