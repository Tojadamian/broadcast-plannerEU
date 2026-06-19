import numpy as np

from broadcast_planner.core.grids import make_grid, read_raster
from broadcast_planner.core.regions import DEFAULT_REGION, EXAMPLES_DIR
from broadcast_planner.core.sites import load_sites
from broadcast_planner.dvb_t.coverage import run_dvb_t_coverage
from broadcast_planner.dvb_t.models import DvbTConfig
from broadcast_planner.dvb_t.sfn import assign_sfn_delays, sfn_delay_table


def test_sfn_delay_table():
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    sites = assign_sfn_delays(sites)
    table = sfn_delay_table(sites)
    pilot_rows = [r for r in table if r["is_pilot"]]
    assert len(pilot_rows) == 1
    assert pilot_rows[0]["launch_delay_us"] == 0.0


def test_dvb_t_coverage_runs():
    region = DEFAULT_REGION
    grid = make_grid(region, resolution_m=5000.0)
    dem = np.full(grid.shape, 40.0)
    clutter = np.full(grid.shape, 10, dtype=np.int16)
    population = np.full(grid.shape, 100.0)
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    result = run_dvb_t_coverage(sites, grid, dem, clutter, population, DvbTConfig())
    assert result.best_field_dbuv_m.shape == grid.shape
    assert result.population_stats["coverage_percent"] >= 0.0


def test_dvb_t_coverage_aligns_native_rasters():
    region = DEFAULT_REGION
    grid = make_grid(region, resolution_m=5000.0)
    dem, source_grid = read_raster(EXAMPLES_DIR / "dem.tif")
    clutter, _ = read_raster(EXAMPLES_DIR / "clutter.tif")
    population, _ = read_raster(EXAMPLES_DIR / "population.tif")
    sites = load_sites(EXAMPLES_DIR / "sites.yaml")
    result = run_dvb_t_coverage(
        sites, grid, dem, clutter, population, DvbTConfig(), source_grid=source_grid
    )
    assert result.best_field_dbuv_m.shape == grid.shape
    assert np.any(np.isfinite(result.best_field_dbuv_m))
