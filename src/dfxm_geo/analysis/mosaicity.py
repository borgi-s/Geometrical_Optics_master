"""Mosaicity-extraction analysis routines for DFXM rocking-grid stacks.

The functions in this module are a port of the per-pixel center-of-mass
extraction in the original ``init_forward.py``. They consume image stacks
already reshaped to ``(chi_steps, phi_steps, H, W)`` and return:

- ``compute_chi_shift``: a scalar χ-axis offset (in radians) measured from the
  corner pixel of a strain-free reference stack. Use it to calibrate the χ
  axis before extracting COMs from a strained stack.
- ``compute_com_maps``: per-pixel mosaicity maps in φ and χ (radians) for a
  strained stack.

All angular ranges are in **radians** — the project-wide convention shared by
the ``[scan.*]`` TOML blocks, ``build_scan_grid``, and ``forward``.

The straight port preserves the original numerical conventions, including the
``abs(shift)`` sign-loss in ``compute_chi_shift``. The per-pixel
``center_of_mass`` loop in ``compute_com_maps`` was vectorized via two einsum
contractions in Phase 8 close-out (May 2026), eliminating ~86k scipy calls
per pipeline run at the production detector size.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass


def compute_chi_shift(
    stack_perfect: np.ndarray,
    chi_steps: int,
    chi_range: float,
    *,
    oversample: int = 100,
) -> float:
    """Measure the systematic χ offset from the corner pixel of a perfect stack.

    The corner pixel ``stack_perfect[:, :, -1, -1]`` of a strain-free crystal
    should peak at χ = 0. Any offset is interpreted as a systematic shift
    introduced by the finite rocking grid; this function returns the magnitude
    of that offset in *radians* (the value is expressed in the same units as
    ``chi_range``, which is radians per the project-wide convention).

    Args:
        stack_perfect: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame
            of the perfect-crystal rocking sweep.
        chi_steps: Number of χ steps in the rocking grid.
        chi_range: Half-range of χ in radians (same units as ``configs/default.toml``).
        oversample: High-resolution refinement factor on the χ axis. The
            original script uses 100.

    Returns:
        Absolute χ-axis shift in radians.
    """
    com = center_of_mass(stack_perfect[:, :, -1, -1])
    chi_high = np.linspace(-chi_range, chi_range, chi_steps * oversample)
    shift_idx = com[0] * oversample - (chi_steps * oversample / 2)
    return float(chi_high[int(abs(shift_idx))] - chi_high[0])


def compute_com_maps(
    stack: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    *,
    chi_shift: float = 0.0,
    oversample: int = 20,
    chi_oversample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel center-of-mass extraction over the (φ, χ) rocking grid.

    For each detector pixel ``(i, j)``, computes the centroid of
    ``stack[:, :, i, j]`` (a ``chi_steps × phi_steps`` rocking-curve image)
    and looks the (φ, χ) values up on a refined grid. The φ axis uses
    ``oversample`` cells per original step; the χ axis uses ``chi_oversample``
    when provided, else falls back to ``oversample`` (matching the original
    ``init_forward.py`` convention of one shared factor).

    Args:
        stack: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame of the
            dislocated rocking sweep.
        phi_range, phi_steps: Half-range (radians) and step count of φ.
        chi_range, chi_steps: Half-range (radians) and step count of χ.
        chi_shift: Additive shift to the χ axis (radians), as returned by
            :func:`compute_chi_shift`.
        oversample: High-resolution refinement factor for the φ grid (and the
            χ grid when ``chi_oversample`` is None).
        chi_oversample: Optional χ-axis refinement factor; defaults to
            ``oversample``.

    Returns:
        ``(phi_list, chi_list)`` mosaicity maps, both shape ``(H, W)`` and in
        radians. Pixels whose rocking-curve image is identically zero produce
        ``np.nan`` in both outputs (dead detector pixels, beamstop shadows).
    """
    chi_over = chi_oversample if chi_oversample is not None else oversample
    # Scan ranges and chi_shift are in radians (project-wide convention, set in
    # the [scan.*] TOML blocks and consumed unconverted by `build_scan_grid` /
    # `forward`). The COM lookup grids are therefore built directly in radians —
    # no deg→rad conversion (the pre-v2.0.2 port carried over init_forward.py's
    # degrees convention and double-counted by deg2rad-ing an already-radian
    # range, shrinking every COM map by 180/π ≈ 57x).
    phi_high = np.linspace(-phi_range, phi_range, phi_steps * oversample)
    chi_high = np.linspace(-chi_range + chi_shift, chi_range + chi_shift, chi_steps * chi_over)

    H, W = stack.shape[2], stack.shape[3]

    chi_indices = np.arange(chi_steps, dtype=np.float64)
    phi_indices = np.arange(phi_steps, dtype=np.float64)
    totals = stack.sum(axis=(0, 1))  # (H, W)
    chi_weighted = np.einsum("cphw,c->hw", stack, chi_indices)
    phi_weighted = np.einsum("cphw,p->hw", stack, phi_indices)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi_idx_arr = chi_weighted / totals
        phi_idx_arr = phi_weighted / totals

    # Dead pixels (totals == 0) produce NaN COMs; preserve them as NaN in the
    # output instead of casting to an arbitrary int (the per-pixel loop hit
    # this via the isnan branch).
    valid = np.isfinite(chi_idx_arr) & np.isfinite(phi_idx_arr)

    phi_list = np.full((H, W), np.nan)
    chi_list = np.full((H, W), np.nan)
    # int(round(x)) on Python floats uses banker's rounding; np.rint matches.
    phi_lookup = np.rint(phi_idx_arr[valid] * oversample).astype(np.int64)
    chi_lookup = np.rint(chi_idx_arr[valid] * chi_over).astype(np.int64)
    phi_list[valid] = phi_high[phi_lookup]
    chi_list[valid] = chi_high[chi_lookup]

    return phi_list, chi_list
