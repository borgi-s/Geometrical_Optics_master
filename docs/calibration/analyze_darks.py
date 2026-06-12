"""Characterize the true pco_ff darks: 10 x 0.5 s loopscan frames.

Outputs: offset, readout sigma (temporal), FPN sigma, hot-pixel census,
histogram shape, and a summary figure saved as ``darks_characterization.png``
next to this script.

Data requirements
-----------------
Reads a single LIMA HDF5 file from the local experimental-data archive::

    <DATA_ROOT>/bicrystal_111_layer_rocking_scans_darks/scan0001/pco_ff_0000.h5

The experimental frames are **local only — not in the repository**.
Reading LIMA frames requires ``hdf5plugin`` (bitshuffle codec)::

    pip install hdf5plugin

Provenance disclaimer
---------------------
This is a calibration-provenance script, NOT a CI test.  It records how the
``pco_edge_4.2_id03`` offset, readout-noise, and FPN parameters in
``src/dfxm_geo/detector.py`` were measured.  Do not add it to pytest.

Run with the project venv::

    C:\\Users\\borgi\\Documents\\GM-reworked\\.venv\\Scripts\\python.exe \\
        docs/calibration/analyze_darks.py
"""

import os as _os

import h5py
import hdf5plugin  # noqa: F401  # REQUIRED before h5py.File for LIMA bitshuffle frames
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- paths ---------------------------------------------------------------
# DATA_ROOT: local archive of Sina's ID03 beamtime data (not in the repo).
DATA_ROOT = r"C:\Users\borgi\Documents\GM-reworked\experimental_data"

SRC = DATA_ROOT + r"\bicrystal_111_layer_rocking_scans_darks\scan0001\pco_ff_0000.h5"
# Output figure saved alongside this script.
OUT = _os.path.dirname(_os.path.abspath(__file__))
# --------------------------------------------------------------------------

with h5py.File(SRC, "r") as f:
    g = f["entry_0000/ESRF-ID03/pco_ff"]
    expo = float(g["acquisition/exposure_time"][()])
    stack = g["data"][()].astype(np.float64)

n = stack.shape[0]
mu = stack.mean(axis=0)  # per-pixel mean (offset map incl. FPN)
sd = stack.std(axis=0, ddof=1)  # per-pixel temporal sigma (true noise)

print(f"exposure={expo}s, n={n} frames, shape={stack.shape[1:]}")
print(f"global offset: mean={mu.mean():.3f}, median={np.median(mu):.3f} ADU")
print(f"temporal sigma: median={np.median(sd):.3f}, mean={sd.mean():.3f} ADU")
print(f"FPN sigma (std of per-pixel means): {mu.std():.3f} ADU")
print(f"robust FPN (MAD-based): {1.4826 * np.median(np.abs(mu - np.median(mu))):.3f} ADU")

# hot pixels
for thr in (5, 10, 50):
    nh = int((mu > np.median(mu) + thr).sum())
    print(f"pixels with mean > median+{thr} ADU: {nh} ({100 * nh / mu.size:.4f} %)")

# columnwise / rowwise structure (sCMOS column noise)
col_prof = mu.mean(axis=0)
row_prof = mu.mean(axis=1)
print(f"column-profile p2p: {col_prof.max() - col_prof.min():.2f} ADU, std {col_prof.std():.3f}")
print(f"row-profile    p2p: {row_prof.max() - row_prof.min():.2f} ADU, std {row_prof.std():.3f}")

# does a single dark frame's histogram look Gaussian?
fr0 = stack[0]
med = np.median(fr0)
print(
    f"\nsingle frame: median={med:.1f}, p0.1={np.percentile(fr0, 0.1):.0f}, "
    f"p99.9={np.percentile(fr0, 99.9):.0f}, max={fr0.max():.0f}"
)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ax = axes[0, 0]
im = ax.imshow(mu, vmin=np.percentile(mu, 0.5), vmax=np.percentile(mu, 99.5), cmap="viridis")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title(f"per-pixel mean of {n} darks (offset map)")
ax.axis("off")

ax = axes[0, 1]
im = ax.imshow(sd, vmin=0, vmax=np.percentile(sd, 99.5), cmap="magma")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("per-pixel temporal sigma (true noise)")
ax.axis("off")

ax = axes[1, 0]
ax.hist(fr0.ravel(), bins=np.arange(85, 140), log=True, alpha=0.7, label="single dark frame")
ax.hist(mu.ravel(), bins=np.arange(85, 140), log=True, alpha=0.7, label="per-pixel mean (FPN)")
ax.set_xlabel("ADU")
ax.legend()
ax.set_title("dark histograms (log y)")

ax = axes[1, 1]
ax.plot(col_prof, lw=0.5, label="column means")
ax.plot(row_prof, lw=0.5, label="row means")
ax.set_xlabel("pixel index")
ax.set_ylabel("ADU")
ax.legend()
ax.set_title("row/column offset profiles")

fig.suptitle(f"pco_ff true darks: {n} x {expo} s")
fig.tight_layout()
out_path = _os.path.join(OUT, "darks_characterization.png")
fig.savefig(out_path, dpi=110)
print(f"saved figure -> {out_path}")
