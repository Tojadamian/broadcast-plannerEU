"""Tests for propagation engine."""

import numpy as np
import pytest

from broadcast_planner.core.grids import make_grid
from broadcast_planner.core.regions import DEFAULT_REGION
from broadcast_planner.core.sites import Site
from broadcast_planner.propagation.p1546_engine import (
    HAS_PY1546,
    compute_site_field_strength_fast,
    erp_to_field_strength_dbuv_m,
    _fallback_loss_db,
)


def test_fallback_loss_increases_with_distance():
    near = _fallback_loss_db(474.0, 1.0, 300.0, 10.0, 10.0, "Rural")
    far = _fallback_loss_db(474.0, 50.0, 300.0, 10.0, 10.0, "Rural")
    assert far > near


def test_field_strength_decreases_with_distance():
    region = DEFAULT_REGION
    grid = make_grid(region, resolution_m=5000.0)
    dem = np.full(grid.shape, 50.0)
    clutter = np.full(grid.shape, 10, dtype=np.int16)
    site = Site(id="T1", name="Test", x=4750000.0, y=2290000.0, erp_kw=100.0)
    result = compute_site_field_strength_fast(site, grid, dem, clutter)
    center = result.field_strength_dbuv_m[grid.height // 2, grid.width // 2]
    corner = result.field_strength_dbuv_m[0, 0]
    assert center > corner


@pytest.mark.skipif(not HAS_PY1546, reason="Py1546 not installed")
def test_py1546_point_call():
    from Py1546 import P1546

    lb = P1546.bt_loss(
        f=474.0,
        t=50.0,
        heff=300.0,
        h2=10.0,
        R2=10.0,
        area="Rural",
        d_v=np.array([10.0]),
        path_c=np.array(["Land"]),
        pathinfo=0,
    )
    assert float(lb[0]) > 0
