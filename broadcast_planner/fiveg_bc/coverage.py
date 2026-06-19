"""5G broadcast coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from broadcast_planner.core.grids import GridSpec, align_layers_to_grid
from broadcast_planner.core.sites import SiteCollection
from broadcast_planner.dvb_t.sfn import sfn_feasibility_mask
from broadcast_planner.fiveg_bc.link_budget import (
    erp_kw_to_rsrp_dbm,
    rx_antenna_gain,
    sinr_db,
    thermal_noise_dbm,
)
from broadcast_planner.fiveg_bc.models import FiveGBroadcastConfig
from broadcast_planner.planning.population import population_coverage_stats
from broadcast_planner.propagation.p1546_engine import best_server_map, compute_multi_site_fields


@dataclass
class FiveGCoverageResult:
    rsrp_dbm: np.ndarray
    sinr_db_map: np.ndarray
    service_coverage: np.ndarray
    sfn_feasible: np.ndarray
    isd_feasible: bool
    isd_km_observed: float
    population_stats: Dict
    server_id_lookup: Dict[int, str]


def _observed_isd_km(sites: SiteCollection) -> float:
    coords = [(s.x, s.y) for s in sites.sites]
    if len(coords) < 2:
        return 0.0
    dists: List[float] = []
    for i, (x1, y1) in enumerate(coords):
        for x2, y2 in coords[i + 1 :]:
            dists.append(np.hypot(x2 - x1, y2 - y1) / 1000.0)
    return float(np.mean(dists))


def run_fiveg_broadcast_coverage(
    sites: SiteCollection,
    grid: GridSpec,
    dem: np.ndarray,
    clutter: np.ndarray,
    population: np.ndarray,
    config: FiveGBroadcastConfig,
    source_grid: GridSpec | None = None,
) -> FiveGCoverageResult:
    dem, clutter, population = align_layers_to_grid(
        grid, dem, clutter, population, source_grid=source_grid
    )
    fields = compute_multi_site_fields(sites, grid, dem, clutter, fast=True)
    best_field, server_map, id_lookup = best_server_map(fields)
    site_ids = list(fields.keys())
    site_by_id = {s.id: s for s in sites.sites}

    noise = thermal_noise_dbm(config.bandwidth_mhz, config.noise_figure_db)
    rx_gain = rx_antenna_gain(config)

    rsrp = np.full(grid.shape, np.nan, dtype=np.float64)
    sinr_map = np.full(grid.shape, np.nan, dtype=np.float64)

    site_ids = list(fields.keys())
    stack_loss = np.stack([fields[sid].transmission_loss_db for sid in site_ids], axis=0)

    for idx, sid in enumerate(site_ids):
        site = site_by_id[sid]
        loss = stack_loss[idx]
        wanted_dbm = erp_kw_to_rsrp_dbm(site.erp_kw, np.nan_to_num(loss, nan=200.0), rx_gain)
        rsrp = np.fmax(rsrp, wanted_dbm)

    for r in range(grid.height):
        for c in range(grid.width):
            if not np.isfinite(stack_loss[0, r, c]):
                continue
            best_i = server_map[r, c]
            if best_i < 0:
                continue
            best_site = site_by_id[id_lookup[best_i]]
            wanted = erp_kw_to_rsrp_dbm(
                best_site.erp_kw,
                stack_loss[best_i, r, c],
                rx_gain,
            )
            interferers = []
            for j, sid in enumerate(site_ids):
                if j == best_i:
                    continue
                if np.isfinite(stack_loss[j, r, c]):
                    interferers.append(
                        erp_kw_to_rsrp_dbm(site_by_id[sid].erp_kw, stack_loss[j, r, c], rx_gain)
                    )
            if interferers:
                i_dbm = 10.0 * np.log10(sum(10 ** (x / 10.0) for x in interferers))
            else:
                i_dbm = -200.0
            sinr_map[r, c] = sinr_db(wanted, i_dbm, noise)

    from scipy.stats import norm

    z = (config.target_sinr_db - sinr_map) / 5.5
    lp_ok = (1.0 - norm.cdf(z)) * 100.0 >= config.location_probability_percent

    xx, yy = grid.mesh_xy()
    sfn_ok = sfn_feasibility_mask(sites, xx, yy, max_delay_us=config.max_sfn_delay_us)

    service = lp_ok & sfn_ok & np.isfinite(sinr_map)
    observed_isd = _observed_isd_km(sites)
    isd_ok = observed_isd <= config.typical_isd_km * 1.25 if len(sites) > 1 else True

    pop_stats = population_coverage_stats(population, service)

    return FiveGCoverageResult(
        rsrp_dbm=rsrp,
        sinr_db_map=sinr_map,
        service_coverage=service,
        sfn_feasible=sfn_ok,
        isd_feasible=isd_ok,
        isd_km_observed=observed_isd,
        population_stats=pop_stats,
        server_id_lookup=id_lookup,
    )
