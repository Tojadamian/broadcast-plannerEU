"""Greedy site placement optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np

from broadcast_planner.core.grids import GridSpec
from broadcast_planner.core.sites import Site, SiteCollection
from broadcast_planner.dvb_t.models import DvbTConfig
from broadcast_planner.planning.candidates import candidate_to_site, generate_candidates
from broadcast_planner.core.regions import RegionConfig

@dataclass
class OptimizationResult:
    selected_sites: SiteCollection
    incremental_coverage: List[float]
    final_coverage_percent: float


def greedy_add_sites(
    region: RegionConfig,
    grid: GridSpec,
    dem: np.ndarray,
    clutter: np.ndarray,
    population: np.ndarray,
    existing: SiteCollection,
    n_add: int = 3,
    dvb_config: DvbTConfig | None = None,
    topology: str = "MPMT",
    coverage_fn: Callable | None = None,  # DICTATED BY CALLER TO AVOID CIRCULAR IMPORT
) -> OptimizationResult:
    config = dvb_config or DvbTConfig()
    sites = SiteCollection(sites=list(existing.sites))
    candidates = generate_candidates(region, sites)
    incremental: List[float] = []
    
    # Fallback to default DVB-T engine if no specific function provided
    if coverage_fn is None:
        from broadcast_planner.dvb_t.coverage import run_dvb_t_coverage
        eval_fn = run_dvb_t_coverage
    else:
        eval_fn = coverage_fn

    for step in range(n_add):
        best_candidate = None
        best_gain = -1.0
        baseline = eval_fn(sites, grid, dem, clutter, population, config)
        base_served = baseline.population_stats["served_population"]

        for idx, (x, y) in enumerate(candidates):
            trial_site = candidate_to_site(x, y, f"NEW_{step}_{idx}", topology=topology)
            trial = SiteCollection(sites=list(sites.sites) + [trial_site])
            result = eval_fn(trial, grid, dem, clutter, population, config)
            gain = result.population_stats["served_population"] - base_served
            if gain > best_gain:
                best_gain = gain
                best_candidate = (x, y, trial_site)

        if best_candidate is None or best_gain <= 0:
            break

        x, y, new_site = best_candidate
        new_site.id = f"OPT_{step + 1}"
        new_site.name = f"Optimized site {step + 1}"
        sites = SiteCollection(sites=list(sites.sites) + [new_site])
        incremental.append(best_gain)
        candidates = [
            c
            for c in candidates
            if np.hypot(c[0] - x, c[1] - y) > 15_000.0
        ]

    final = eval_fn(sites, grid, dem, clutter, population, config)
    return OptimizationResult(
        selected_sites=sites,
        incremental_coverage=incremental,
        final_coverage_percent=final.population_stats["coverage_percent"],
    )