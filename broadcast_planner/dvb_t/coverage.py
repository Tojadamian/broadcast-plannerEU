"""DVB-T coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from broadcast_planner.core.grids import GridSpec, align_layers_to_grid
from broadcast_planner.core.sites import SiteCollection
from broadcast_planner.dvb_t.models import DvbTConfig
from broadcast_planner.dvb_t.sfn import assign_sfn_delays, sfn_feasibility_mask
from broadcast_planner.planning.interference import compute_cn_plus_i_map
from broadcast_planner.planning.population import population_coverage_stats
from broadcast_planner.propagation.p1546_engine import (
    best_server_map,
    compute_multi_site_fields,
    location_probability_coverage,
)


@dataclass
class DvbTCoverageResult:
    fields_by_site: Dict
    best_field_dbuv_m: np.ndarray
    server_index_map: np.ndarray
    server_id_lookup: Dict[int, str]
    service_coverage: np.ndarray
    sfn_feasible: np.ndarray
    cn_plus_i_db: np.ndarray
    population_stats: Dict


def run_dvb_t_coverage(
    sites: SiteCollection,
    grid: GridSpec,
    dem: np.ndarray,
    clutter: np.ndarray,
    population: np.ndarray,
    config: DvbTConfig,
    interference_margin_db: float = 25.0,
    source_grid: GridSpec | None = None,
) -> DvbTCoverageResult:
    dem, clutter, population = align_layers_to_grid(
        grid, dem, clutter, population, source_grid=source_grid
    )
    sites = assign_sfn_delays(sites)
    fields = compute_multi_site_fields(sites, grid, dem, clutter, fast=True)
    best_field, server_map, id_lookup = best_server_map(fields)

    service = location_probability_coverage(
        best_field,
        config.field_threshold_dbuv_m,
        sigma_db=config.sigma_db,
        target_lp_percent=config.location_probability_percent,
    )

    xx, yy = grid.mesh_xy()
    sfn_ok = sfn_feasibility_mask(sites, xx, yy, config)
    service = service & sfn_ok

    cn_i = compute_cn_plus_i_map(fields, server_map, id_lookup, config.required_cn_db)

    pop_stats = population_coverage_stats(population, service)

    return DvbTCoverageResult(
        fields_by_site=fields,
        best_field_dbuv_m=best_field,
        server_index_map=server_map,
        server_id_lookup=id_lookup,
        service_coverage=service,
        sfn_feasible=sfn_ok,
        cn_plus_i_db=cn_i,
        population_stats=pop_stats,
    )
