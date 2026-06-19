"""Simplified co-channel C/N+I analysis."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _power_sum_dbm(values_dbm: np.ndarray) -> float:
    finite = values_dbm[np.isfinite(values_dbm)]
    if finite.size == 0:
        return -200.0
    return 10.0 * np.log10(np.sum(10 ** (finite / 10.0)))


def t_lnm_aggregate(interference_db_list, mu: float = 0.0, sigma: float = 3.0) -> float:
    """Simplified t-LNM: log-normal aggregate of interferers in dBm domain."""
    if not interference_db_list:
        return -200.0
    arr = np.array(interference_db_list, dtype=np.float64)
    return float(_power_sum_dbm(arr) + mu * sigma)


def compute_cn_plus_i_map(
    fields_by_site: Dict,
    server_map: np.ndarray,
    id_lookup: Dict[int, str],
    required_cn_db: float,
) -> np.ndarray:
    site_ids = list(fields_by_site.keys())
    stack = np.stack([fields_by_site[sid].field_strength_dbuv_m for sid in site_ids], axis=0)
    cn_map = np.full(server_map.shape, np.nan, dtype=np.float64)

    for r in range(server_map.shape[0]):
        for c in range(server_map.shape[1]):
            best_i = server_map[r, c]
            if best_i < 0 or not np.isfinite(stack[best_i, r, c]):
                continue
            wanted = stack[best_i, r, c]
            interferers = []
            for j in range(len(site_ids)):
                if j == best_i:
                    continue
                val = stack[j, r, c]
                if np.isfinite(val):
                    interferers.append(val - wanted)
            if interferers:
                i_rel_db = t_lnm_aggregate(interferers)
                cn_i = -i_rel_db
            else:
                cn_i = 60.0
            cn_map[r, c] = cn_i - required_cn_db
    return cn_map


def cochannel_violations(cn_plus_i_map: np.ndarray, min_margin_db: float = 0.0) -> int:
    valid = np.isfinite(cn_plus_i_map)
    return int((cn_plus_i_map[valid] < min_margin_db).sum())
