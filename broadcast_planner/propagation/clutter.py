"""Land cover to ITU clutter mapping."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Copernicus / simplified class codes -> (area label, R2 metres)
CLUTTER_MAP = {
    10: ("Rural", 10.0),
    20: ("Suburban", 15.0),
    30: ("Urban", 20.0),
    40: ("Dense Urban", 20.0),
    50: ("Dense Urban", 20.0),
    60: ("Rural", 10.0),
    80: ("Sea", 10.0),
    90: ("Rural", 10.0),
}

DEFAULT_R2 = 10.0
DEFAULT_AREA = "Rural"


def r2_from_class(class_array: np.ndarray) -> np.ndarray:
    out = np.full(class_array.shape, DEFAULT_R2, dtype=np.float64)
    for code, (_, r2) in CLUTTER_MAP.items():
        out[class_array == code] = r2
    return out


def area_from_class(class_array: np.ndarray) -> np.ndarray:
    out = np.full(class_array.shape, DEFAULT_AREA, dtype=object)
    for code, (area, _) in CLUTTER_MAP.items():
        out[class_array == code] = area
    return out


def classify_from_elevation(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic clutter when only DEM is available."""
    urban_mask = dem > np.nanpercentile(dem, 85)
    suburban_mask = (dem > np.nanpercentile(dem, 65)) & ~urban_mask
    classes = np.full(dem.shape, 10, dtype=np.int16)
    classes[suburban_mask] = 20
    classes[urban_mask] = 30
    return classes, r2_from_class(classes)
