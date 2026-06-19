import numpy as np

from broadcast_planner.core.grids import make_grid
from broadcast_planner.core.regions import DEFAULT_REGION, EXAMPLES_DIR
from broadcast_planner.core.sites import load_sites
from broadcast_planner.dvb_t.models import DvbTConfig
from broadcast_planner.planning.candidates import generate_candidates
from broadcast_planner.planning.optimizer import greedy_add_sites


def test_candidate_generation():
    region = DEFAULT_REGION
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    candidates = generate_candidates(region, sites)
    assert len(candidates) > 0


def test_greedy_optimizer():
    region = DEFAULT_REGION
    grid = make_grid(region, resolution_m=8000.0)
    dem = np.full(grid.shape, 40.0)
    clutter = np.full(grid.shape, 10, dtype=np.int16)
    population = np.full(grid.shape, 100.0)
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    result = greedy_add_sites(region, grid, dem, clutter, population, sites, n_add=1, dvb_config=DvbTConfig())
    assert len(result.selected_sites) >= len(sites)
