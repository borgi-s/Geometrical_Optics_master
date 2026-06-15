# src/dfxm_geo/scoring/engine.py
from __future__ import annotations

import numpy as np


def preprocess(img: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Background-subtract (mean + k*std), clip negatives to 0, normalize by std.

    Matches the Borgi 2025 method: the floor uses the raw image statistics, then
    the clipped result is scaled by its own std so different images are comparable.
    """
    arr = np.asarray(img, dtype=np.float64)
    sub = arr - (arr.mean() + k * arr.std())
    sub[sub < 0] = 0.0
    s = sub.std()
    if s == 0.0:
        return sub
    return sub / s
