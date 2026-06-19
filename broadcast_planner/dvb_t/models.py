"""DVB-T/T2 COFDM parameters and thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Required C/N (dB) for DVB-T — simplified EBU planning values
MODULATION_CN: Dict[str, float] = {
    "QPSK_1/2": 5.0,
    "QPSK_2/3": 6.5,
    "QPSK_3/4": 7.5,
    "QPSK_5/6": 8.5,
    "QPSK_7/8": 9.5,
    "16QAM_1/2": 10.5,
    "16QAM_2/3": 12.0,
    "16QAM_3/4": 13.5,
    "64QAM_2/3": 17.0,
    "64QAM_3/4": 19.0,
    "64QAM_5/6": 21.0,
}

# Guard interval (us) -> max SFN path delay (us) for COFDM
GUARD_INTERVAL_US: Dict[str, float] = {
    "1/4": 224.0,
    "1/8": 112.0,
    "1/16": 56.0,
    "1/32": 28.0,
}

# Field strength planning thresholds (dBuV/m) at 50% locations
FIELD_THRESHOLDS_DVB_T: Dict[str, float] = {
    "fixed_roof": 48.0,
    "portable_outdoor": 55.0,
    "mobile": 60.0,
}


@dataclass
class DvbTConfig:
    modulation: str = "64QAM_2/3"
    guard_interval: str = "1/4"
    receiver_mode: str = "fixed_roof"
    location_probability_percent: float = 95.0
    planning_time_percent: float = 50.0
    sigma_db: float = 5.5
    margin_db: float = 0.0

    @property
    def required_cn_db(self) -> float:
        return MODULATION_CN.get(self.modulation, 17.0) + self.margin_db

    @property
    def field_threshold_dbuv_m(self) -> float:
        return FIELD_THRESHOLDS_DVB_T.get(self.receiver_mode, 48.0)

    @property
    def max_sfn_delay_us(self) -> float:
        return GUARD_INTERVAL_US.get(self.guard_interval, 224.0)
