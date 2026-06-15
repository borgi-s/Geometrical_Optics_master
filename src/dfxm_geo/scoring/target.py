from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .types import GridSpec


def _crop_or_pad(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    h, w = a.shape
    out = np.zeros((H, W), dtype=a.dtype)
    sy0 = max((h - H) // 2, 0)
    sx0 = max((w - W) // 2, 0)
    dy0 = max((H - h) // 2, 0)
    dx0 = max((W - w) // 2, 0)
    th = min(h, H)
    tw = min(w, W)
    out[dy0 : dy0 + th, dx0 : dx0 + tw] = a[sy0 : sy0 + th, sx0 : sx0 + tw]
    return out


def resample_to_grid(
    img: np.ndarray, src_pitch_um: tuple[float, float], grid: GridSpec
) -> np.ndarray:
    """Resample an image from its object-plane pitch onto the library grid.

    Zooms by src_pitch/dst_pitch so a feature of a given micrometre extent keeps
    that extent, then center-crops or zero-pads to the library shape.
    """
    arr = np.asarray(img, dtype=np.float64)
    sy, sx = src_pitch_um
    dy, dx = grid.pitch_um
    if dy <= 0 or dx <= 0 or sy <= 0 or sx <= 0:
        raise ValueError("pixel pitches must be positive")
    zoomed = zoom(arr, (sy / dy, sx / dx), order=1)
    return _crop_or_pad(zoomed, grid.shape)
