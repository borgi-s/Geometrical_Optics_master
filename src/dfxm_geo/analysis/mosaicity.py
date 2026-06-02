"""Mosaicity-extraction analysis routines for DFXM rocking-grid stacks.

The function in this module is a port of the per-pixel center-of-mass
extraction in the original ``init_forward.py``. It consumes an image stack
already reshaped to ``(chi_steps, phi_steps, H, W)`` and returns:

- ``compute_com_maps``: per-pixel mosaicity maps in φ and χ (radians) for a
  strained stack.

All angular ranges are in **radians** — the project-wide convention shared by
the ``[scan.*]`` TOML blocks, ``build_scan_grid``, and ``forward``.

``compute_com_maps`` was vectorized via two einsum contractions in Phase 8
close-out (May 2026), then in v2.0.2 changed from the original oversampled-grid
index lookup to a direct intensity-weighted mean in radians — the exact center
of mass, with no quantization (see the ``compute_com_maps`` docstring).
"""

from __future__ import annotations

import numpy as np


def compute_com_maps(
    stack: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    *,
    chi_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel center-of-mass extraction over the (φ, χ) rocking grid.

    For each detector pixel ``(i, j)`` this computes the intensity-weighted
    mean (the true first moment) of ``stack[:, :, i, j]`` along the φ and χ
    axes, in **radians**::

        COM_φ = Σ I·φ / Σ I        COM_χ = Σ I·χ / Σ I + chi_shift

    where ``φ`` / ``χ`` run over ``linspace(-range, range, steps)``.

    .. note:: v2.0.2 de-quantization

        Earlier versions (a straight port of ``init_forward.py``) computed the
        centroid *index* and then looked it up on an ``oversample``-refined grid
        via ``np.rint`` — quantizing the COM onto a grid whose step is
        ``2·range/(steps·oversample)``. For a wide axis like χ (±mrad) that step
        (~µrad) is comparable to the χ-COM signal itself, so the map showed
        spurious contour banding while the narrow φ axis looked fine. The direct
        weighted mean here is the exact COM — no quantizer, no ``oversample``
        knob, and free of the half-step bias the refined-grid indexing carried.

    Args:
        stack: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame of the
            dislocated rocking sweep.
        phi_range, phi_steps: Half-range (radians) and step count of φ.
        chi_range, chi_steps: Half-range (radians) and step count of χ.
        chi_shift: Additive calibration offset for the χ axis (radians).
            Defaults to ``0.0`` (the nominal χ grid; no runtime calibration).

    Returns:
        ``(phi_list, chi_list)`` mosaicity maps, both shape ``(H, W)`` and in
        radians. Pixels whose rocking-curve image is identically zero produce
        ``np.nan`` in both outputs (dead detector pixels, beamstop shadows).
    """
    # Scan ranges and chi_shift are radians (project-wide convention, set in the
    # [scan.*] TOML blocks and consumed unconverted by `build_scan_grid` /
    # `forward`). The COM is therefore a weighted mean directly in radians — no
    # deg→rad conversion.
    phi_vals = np.linspace(-phi_range, phi_range, phi_steps)
    chi_vals = np.linspace(-chi_range, chi_range, chi_steps) + chi_shift

    H, W = stack.shape[2], stack.shape[3]
    totals = stack.sum(axis=(0, 1), dtype=np.float64)  # (H, W)
    chi_weighted = np.einsum("cphw,c->hw", stack, chi_vals, dtype=np.float64)
    phi_weighted = np.einsum("cphw,p->hw", stack, phi_vals, dtype=np.float64)

    # Dead pixels (totals == 0) stay NaN; everything else is the weighted mean.
    phi_list = np.full((H, W), np.nan)
    chi_list = np.full((H, W), np.nan)
    valid = totals != 0
    phi_list[valid] = phi_weighted[valid] / totals[valid]
    chi_list[valid] = chi_weighted[valid] / totals[valid]

    return phi_list, chi_list
