import numpy as np

from broadcast_planner.core.grids import make_grid
from broadcast_planner.core.regions import DEFAULT_REGION, EXAMPLES_DIR
from broadcast_planner.core.sites import load_sites
from broadcast_planner.fiveg_bc.coverage import run_fiveg_broadcast_coverage
from broadcast_planner.fiveg_bc.models import FiveGBroadcastConfig, NUMEROLOGY_PRESETS


def test_numerology_presets():
    assert NUMEROLOGY_PRESETS["1.25kHz"]["cp_us"] == 200.0


def test_fiveg_coverage_runs():
    grid = make_grid(DEFAULT_REGION, resolution_m=5000.0)
    dem = np.full(grid.shape, 40.0)
    clutter = np.full(grid.shape, 10, dtype=np.int16)
    population = np.full(grid.shape, 100.0)
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    result = run_fiveg_broadcast_coverage(
        sites, grid, dem, clutter, population, FiveGBroadcastConfig()
    )
    assert result.sinr_db_map.shape == grid.shape
    assert result.isd_km_observed > 0
