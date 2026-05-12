"""Mosaicity-extraction analysis routines for DFXM rocking-grid stacks.

The functions in this module are a port of the per-pixel center-of-mass
extraction in the original ``init_forward.py``. They consume image stacks
already reshaped to ``(chi_steps, phi_steps, H, W)`` and return:

- ``compute_chi_shift``: a scalar χ-axis offset (in degrees) measured from the
  corner pixel of a strain-free reference stack. Use it to calibrate the χ
  axis before extracting COMs from a strained stack.
- ``compute_com_maps``: per-pixel mosaicity maps in φ and χ (radians) for a
  strained stack.

The straight port preserves the original numerical conventions, including the
``abs(shift)`` sign-loss in ``compute_chi_shift``. Performance work (vectorizing
the COM loop) is deferred to Phase 8.
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
    of that offset in *degrees* (matching ``init_forward.py``'s convention,
    which converts to radians downstream via ``np.deg2rad``).

    Args:
        stack_perfect: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame
            of the perfect-crystal rocking sweep.
        chi_steps: Number of χ steps in the rocking grid.
        chi_range: Half-range of χ in degrees (same units as ``configs/default.toml``).
        oversample: High-resolution refinement factor on the χ axis. The
            original script uses 100.

    Returns:
        Absolute χ-axis shift in degrees.
    """
    com = center_of_mass(stack_perfect[:, :, -1, -1])
    chi_high = np.linspace(-chi_range, chi_range, chi_steps * oversample)
    shift_idx = com[0] * oversample - (chi_steps * oversample / 2)
    return float(chi_high[int(abs(shift_idx))] - chi_high[0])
