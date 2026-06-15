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


def _resolve_backend(backend: str) -> str:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("backend='torch' requested but torch is not installed") from exc
        return "torch"
    if backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "torch"
        except ImportError:
            pass
        return "numpy"
    raise ValueError(f"unknown backend: {backend!r}")


def _peak_from_ffts(fa: np.ndarray, fb: np.ndarray) -> float:
    cc = np.fft.ifft2(fa * np.conj(fb))
    return float(np.abs(cc).max())


def cross_correlation_peak(
    a: np.ndarray, b: np.ndarray, *, normalize: str = "symmetric", k: float = 2.0
) -> float:
    """Translation-invariant similarity of two images (numpy reference path)."""
    fa = np.fft.fft2(preprocess(a, k))
    fb = np.fft.fft2(preprocess(b, k))
    cross = _peak_from_ffts(fa, fb)
    if normalize == "none":
        return cross
    auto_a = _peak_from_ffts(fa, fa)
    auto_b = _peak_from_ffts(fb, fb)
    if normalize == "symmetric":
        denom = np.sqrt(auto_a * auto_b)
        return cross / denom if denom > 0 else 0.0
    if normalize == "diagonal":
        return cross / auto_a if auto_a > 0 else 0.0
    raise ValueError(f"unknown normalize mode: {normalize!r}")
