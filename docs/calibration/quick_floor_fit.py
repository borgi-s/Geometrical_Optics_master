"""Quick detector-floor estimate from the near-dark bicrystal frames.

Temporal per-pixel mean/std across multi-frame stacks (FPN-free), plus
whole-frame stats for the single-frame 1 s scans.  Cross-checks the
offset model ``102.5 + 7.5·t`` fitted from the true darks.

Datasets used:

- ``bicrystal_111_find_x_edge/scan0054`` and ``scan0058``: 0.1 s near-dark
  multi-frame stacks → temporal floor and sigma at short exposures.
- ``bicrystal_111_find_x_edge/scan0002``: 1 s single frame → spatial floor
  statistics at longer exposure.
- ``bicrystal_111_find_x_edge/scan0011, 0013, 0014, 0016, 0017, 0019, 0020,
  0022``: repeated 1 s omega frames at the same angle → temporal variance
  per signal-level bin (independent gain cross-check).

Data requirements
-----------------
Reads LIMA HDF5 files from the local experimental-data archive::

    <DATA_ROOT>/bicrystal_111_find_x_edge/scan<NNNN>/pco_ff_0000.h5

The experimental frames are **local only — not in the repository**.
Reading LIMA frames requires ``hdf5plugin`` (bitshuffle codec)::

    pip install hdf5plugin

Provenance disclaimer
---------------------
This is a calibration-provenance script, NOT a CI test.  It records how the
offset vs exposure-time model (``offset_base=102.5``, ``dark_rate=7.5``) was
cross-checked with near-dark bicrystal frames.  Do not add this to pytest.

Run with the project venv::

    C:\\Users\\borgi\\Documents\\GM-reworked\\.venv\\Scripts\\python.exe \\
        docs/calibration/quick_floor_fit.py
"""

import h5py
import hdf5plugin  # noqa: F401  # REQUIRED before h5py.File for LIMA bitshuffle frames
import numpy as np

# ---- paths ---------------------------------------------------------------
# DATA_ROOT: local archive of Sina's ID03 beamtime data (not in the repo).
DATA_ROOT = r"C:\Users\borgi\Documents\GM-reworked\experimental_data"

ROOT = DATA_ROOT + r"\bicrystal_111_find_x_edge"
# --------------------------------------------------------------------------


def load(scan, fn="pco_ff_0000.h5"):
    with h5py.File(rf"{ROOT}\scan{scan:04d}\{fn}", "r") as f:
        g = f["entry_0000/ESRF-ID03/pco_ff"]
        return float(g["acquisition/exposure_time"][()]), g["data"][()].astype(np.float64)


# multi-frame near-dark stacks at 0.1 s
for scan in (54, 58):
    expo, stack = load(scan)
    mu = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1)
    # mask out anything with real signal
    dark = mu < np.percentile(mu, 95)
    print(
        f"scan{scan:04d} expo={expo}s n={stack.shape[0]}: "
        f"floor mean={mu[dark].mean():.2f} ADU, temporal sigma median={np.median(sd[dark]):.2f} ADU, "
        f"FPN sigma (std of per-pixel means)={mu[dark].std():.2f} ADU"
    )

# single-frame 1 s near-dark scan: spatial stats only
expo, fr = load(2)
fr = fr[0]
dark = fr < np.percentile(fr, 95)
print(
    f"scan0002 expo={expo}s single frame: spatial mean={fr[dark].mean():.2f} ADU, "
    f"spatial sigma={fr[dark].std():.2f} ADU (includes FPN)"
)

# repeated identical 1 s omega frames -> temporal stats with signal
stack = []
for scan in (11, 13, 14, 16, 17, 19, 20, 22):
    try:
        e, d = load(scan)
        stack.append(d[0])
    except OSError:
        print(f"scan{scan:04d} unreadable, skipped")
mu = np.mean(stack, axis=0)
sd = np.std(stack, axis=0, ddof=1)
print(f"\nrepeats at omega=-7.1061, n={len(stack)} frames, 1 s:")
for lo, hi in [(0, 120), (120, 200), (200, 400), (400, 700), (700, 1200)]:
    m = (mu >= lo) & (mu < hi)
    if m.sum() < 1000:
        continue
    print(
        f"  mean ADU in [{lo},{hi}): n={m.sum():8d}  <mu>={mu[m].mean():7.1f}  "
        f"<var>={np.mean(sd[m] ** 2):8.1f}  var/(mu-offset)={np.mean(sd[m] ** 2) / max(mu[m].mean() - 103, 1):.3f}"
    )
