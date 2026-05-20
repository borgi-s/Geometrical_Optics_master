"""Physical and geometric constants for DFXM forward modelling.

All constants are module-level so callers can override them by direct
assignment for parameter sweeps. Document the physical meaning here
rather than scattering it across function signatures.

These values match the existing hardcoded defaults in `functions.py`
and `direct_space/forward_model.py` as of the pre-cleanup main branch.
"""

from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------

# Burgers vector magnitude in the dislocation coordinate frame (µm).
# 2.862e-4 µm = 0.2862 nm — matches Al lattice parameter * sqrt(2)/2.
BURGERS_VECTOR: Final[float] = 2.862e-4

# Poisson ratio (dimensionless). 0.334 ≈ Al at room temperature.
POISSON_RATIO: Final[float] = 0.334

# ---------------------------------------------------------------------------
# ID06 (ESRF) beamline geometry
# ---------------------------------------------------------------------------

# Detector pixel size in the object plane (m).
ID06_PSIZE: Final[float] = 40e-9

# RMS of the Gaussian beam profile in zl (m).
# Defined as FWHM/2.35 where FWHM = 0.15 µm.
ID06_ZL_RMS: Final[float] = 0.15e-6 / 2.35

# Bragg angle θ0 (rad). 17.953° / 2 — corresponds to ID06's default reflection.
ID06_THETA_0: Final[float] = 17.953 / 2 * np.pi / 180

# Detector field of view.
ID06_NPIXELS: Final[int] = 510
ID06_NSUB: Final[int] = 1  # Borgi 2024 (IUCrJ) used 2 for publication; 1 is the typical real-run.

# Miller indices of the default active reflection.
HKL_DEFAULT: Final[tuple[int, int, int]] = (-1, 1, -1)
