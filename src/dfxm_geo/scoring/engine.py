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


def _normalize_matrix(C: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return C
    d = np.diag(C).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "symmetric":
            denom = np.sqrt(np.outer(d, d))
            return np.where(denom > 0, C / denom, 0.0)
        if mode == "diagonal":
            return np.where(d[:, None] > 0, C / d[:, None], 0.0)
    raise ValueError(f"unknown normalize mode: {mode!r}")


def score_matrix(
    frames: np.ndarray, *, normalize: str = "symmetric", backend: str = "auto", k: float = 2.0
) -> np.ndarray:
    """All-pairs normalized cross-correlation matrix (N, N)."""
    backend = _resolve_backend(backend)
    pp = np.stack([preprocess(f, k) for f in frames])
    C = _raw_matrix_torch(pp) if backend == "torch" else _raw_matrix_numpy(pp)  # noqa: F821
    return _normalize_matrix(C, normalize)


def _raw_matrix_numpy(pp: np.ndarray) -> np.ndarray:
    n = pp.shape[0]
    F = np.fft.fft2(pp)  # (n, H, W) complex
    C = np.zeros((n, n))
    for i in range(n):
        cc = np.fft.ifft2(F[i] * np.conj(F[i:]))  # broadcast i vs j>=i
        peaks = np.abs(cc).reshape(n - i, -1).max(axis=1)
        C[i, i:] = peaks
        C[i:, i] = peaks
    return C


def score_target(
    target: np.ndarray,
    frames: np.ndarray,
    *,
    normalize: str = "symmetric",
    backend: str = "auto",
    k: float = 2.0,
) -> np.ndarray:
    """Score one target image against every library frame; returns (N,)."""
    backend = _resolve_backend(backend)  # torch path shares the numpy math here
    pp_t = preprocess(target, k)
    pp = np.stack([preprocess(f, k) for f in frames])
    ft = np.fft.fft2(pp_t)
    F = np.fft.fft2(pp)
    cross = np.abs(np.fft.ifft2(ft[None] * np.conj(F))).reshape(F.shape[0], -1).max(axis=1)
    if normalize == "none":
        return cross
    auto_t = _peak_from_ffts(ft, ft)
    auto_f = np.array([_peak_from_ffts(F[j], F[j]) for j in range(F.shape[0])])
    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize == "symmetric":
            denom = np.sqrt(auto_t * auto_f)
            return np.where(denom > 0, cross / denom, 0.0)
        if normalize == "diagonal":
            return np.where(auto_t > 0, cross / auto_t, 0.0)
    raise ValueError(f"unknown normalize mode: {normalize!r}")
