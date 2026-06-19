"""5G broadcast numerology and topology presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# ETSI TR 36.976 / ITU TMMB planning presets
TOPOLOGY_PRESETS: Dict[str, Dict[str, float]] = {
    "HPHT": {"erp_dbm": 80.0, "height_m": 300.0, "isd_km": 80.0, "erp_kw": 100.0},
    "MPMT": {"erp_dbm": 67.0, "height_m": 80.0, "isd_km": 23.1, "erp_kw": 5.0},
    "LPLT": {"erp_dbm": 43.0, "height_m": 30.0, "isd_km": 5.0, "erp_kw": 0.02},
}

NUMEROLOGY_PRESETS: Dict[str, Dict[str, float]] = {
    "0.37kHz": {"subcarrier_khz": 0.37, "cp_us": 300.0, "symbol_us": 3000.0},
    "1.25kHz": {"subcarrier_khz": 1.25, "cp_us": 200.0, "symbol_us": 1000.0},
    "2.5kHz": {"subcarrier_khz": 2.5, "cp_us": 100.0, "symbol_us": 500.0},
}


@dataclass
class FiveGBroadcastConfig:
    topology: str = "HPHT"
    numerology: str = "1.25kHz"
    target_sinr_db: float = 12.0
    location_probability_percent: float = 95.0
    receiver_type: str = "ROM"
    reuse_pattern: str = "reuse_1_sfn"
    noise_figure_db: float = 7.0
    bandwidth_mhz: float = 8.0

    @property
    def preset(self) -> Dict[str, float]:
        return TOPOLOGY_PRESETS.get(self.topology, TOPOLOGY_PRESETS["HPHT"])

    @property
    def numerology_params(self) -> Dict[str, float]:
        return NUMEROLOGY_PRESETS.get(self.numerology, NUMEROLOGY_PRESETS["1.25kHz"])

    @property
    def max_sfn_delay_us(self) -> float:
        return self.numerology_params["cp_us"]

    @property
    def typical_isd_km(self) -> float:
        return self.preset["isd_km"]
