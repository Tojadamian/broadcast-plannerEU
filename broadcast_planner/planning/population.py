"""Population coverage KPIs."""

from __future__ import annotations

from typing import Dict

import numpy as np


def population_coverage_stats(population: np.ndarray, covered_mask: np.ndarray) -> Dict:
    pop = np.nan_to_num(population, nan=0.0)
    total = float(pop.sum())
    served = float(pop[covered_mask].sum()) if covered_mask.shape == pop.shape else 0.0
    pct = (served / total * 100.0) if total > 0 else 0.0
    return {
        "total_population": total,
        "served_population": served,
        "unserved_population": total - served,
        "coverage_percent": round(pct, 2),
        "pixels_total": int(np.prod(pop.shape)),
        "pixels_covered": int(covered_mask.sum()),
    }
