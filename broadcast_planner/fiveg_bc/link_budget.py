"""5G broadcast link budget helpers."""

from __future__ import annotations

import math

from broadcast_planner.fiveg_bc.models import FiveGBroadcastConfig


def erp_kw_to_rsrp_dbm(erp_kw: float, path_loss_db: float, rx_antenna_gain_dbi: float = 0.0) -> float:
    erp_dbm = 10.0 * math.log10(erp_kw * 1e6)
    return erp_dbm - path_loss_db + rx_antenna_gain_dbi


def thermal_noise_dbm(bandwidth_mhz: float, noise_figure_db: float = 7.0) -> float:
    kt_dbm_hz = -174.0
    return kt_dbm_hz + 10.0 * math.log10(bandwidth_mhz * 1e6) + noise_figure_db


def sinr_db(wanted_dbm: float, interference_dbm: float, noise_dbm: float) -> float:
    def mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    i_plus_n = mw(interference_dbm) + mw(noise_dbm)
    if i_plus_n <= 0:
        return wanted_dbm
    return wanted_dbm - 10.0 * math.log10(i_plus_n)


def rx_antenna_gain(config: FiveGBroadcastConfig) -> float:
    if config.receiver_type.upper() == "ROM":
        return 10.0
    if config.receiver_type.upper() == "MOBILE":
        return 0.0
    return 5.0
