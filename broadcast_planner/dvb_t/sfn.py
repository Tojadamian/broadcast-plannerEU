"""DVB-T SFN delay and guard interval analysis."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List

import numpy as np

from broadcast_planner.core.sites import Site, SiteCollection
from broadcast_planner.dvb_t.models import DvbTConfig

C_M_PER_US = 299.792458  # metres per microsecond


def propagation_delay_us(x1: float, y1: float, x2: float, y2: float) -> float:
    dist_m = math.hypot(x2 - x1, y2 - y1)
    return dist_m / C_M_PER_US


def assign_sfn_delays(sites: SiteCollection, pilot_id: str | None = None) -> SiteCollection:
    pilot = sites.pilot() if pilot_id is None else sites.by_id(pilot_id)
    updated: List[Site] = []
    for site in sites.sites:
        if site.id == pilot.id:
            delay = 0.0
        else:
            delay = propagation_delay_us(pilot.x, pilot.y, site.x, site.y)
        updated.append(
            replace(
                site,
                launch_delay_us=delay,
                is_pilot=site.id == pilot.id,
            )
        )
    return SiteCollection(sites=updated)


def sfn_path_delay_map(
    site: Site,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    launch_delay_us: float,
) -> np.ndarray:
    dist_m = np.hypot(grid_x - site.x, grid_y - site.y)
    prop_us = dist_m / C_M_PER_US
    return np.abs(prop_us - launch_delay_us)


def sfn_feasibility_mask(
    sites: SiteCollection,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    config: DvbTConfig | None = None,
    max_delay_us: float | None = None,
) -> np.ndarray:
    """True where all contributing SFN delays fit within guard interval."""
    if max_delay_us is None:
        if config is None:
            raise ValueError("Provide config or max_delay_us")
        max_delay = config.max_sfn_delay_us
    else:
        max_delay = max_delay_us
    min_margin = np.full(grid_x.shape, np.inf)
    for site in sites:
        delay_map = sfn_path_delay_map(site, grid_x, grid_y, site.launch_delay_us)
        min_margin = np.minimum(min_margin, max_delay - delay_map)
    return min_margin >= 0.0


def sfn_delay_table(sites: SiteCollection) -> List[Dict]:
    pilot = sites.pilot()
    rows = []
    for site in sites.sites:
        rows.append(
            {
                "site_id": site.id,
                "name": site.name,
                "is_pilot": site.id == pilot.id,
                "launch_delay_us": round(site.launch_delay_us, 2),
                "distance_from_pilot_km": round(
                    math.hypot(site.x - pilot.x, site.y - pilot.y) / 1000.0, 2
                ),
            }
        )
    return rows
