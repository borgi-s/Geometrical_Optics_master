"""Photon-transfer gain fit from the 61-frame focus sweep.

Method: for each adjacent frame pair (scene nearly identical, 1 s apart),
normalize global drift, difference the frames, and accumulate robust
per-signal-bin variance. Scene-change pixels are rejected with a boxcar
mask (scene change is spatially correlated; photon noise is not).

Model: var_single(s) = RN^2 + g * s, with s = signal above offset (ADU).

Fitted values (2026-06-12):
    RN^2 ≈ 6.3 ADU²  (readout noise ≈ 2.5 ADU)
    gain g ≈ 2.14 ADU per incident-photon-equivalent

These are the ``read_noise_var_base`` and ``gain`` constants in
``PCO_EDGE_4P2_ID03`` (``src/dfxm_geo/detector.py``).

Data requirements
-----------------
Reads two LIMA HDF5 files from the local experimental-data archive::

    <DATA_ROOT>/111_individual_dislocations_10x_focusing_2/scan0001/pco_ff_0000.h5
        (61 × 1 s focus sweep — the photon-transfer source)
    <DATA_ROOT>/bicrystal_111_layer_rocking_scans_darks/scan0001/pco_ff_0000.h5
        (true darks — used for the hot-pixel mask)

The experimental frames are **local only — not in the repository**.
Reading LIMA frames requires ``hdf5plugin`` (bitshuffle codec)::

    pip install hdf5plugin

Provenance disclaimer
---------------------
This is a calibration-provenance script, NOT a CI test.  It records how the
photon-transfer gain was measured from the ID03 focus-sweep data.  The fit
figure is archived at ``docs/img/photon_transfer_fit.png``.  Do not add this
script to pytest.

Run with the project venv::

    C:\\Users\\borgi\\Documents\\GM-reworked\\.venv\\Scripts\\python.exe \\
        docs/calibration/fit_gain.py
"""

import os as _os

import h5py
import hdf5plugin  # noqa: F401  # REQUIRED before h5py.File for LIMA bitshuffle frames
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

# ---- paths ---------------------------------------------------------------
# DATA_ROOT: local archive of Sina's ID03 beamtime data (not in the repo).
DATA_ROOT = r"C:\Users\borgi\Documents\GM-reworked\experimental_data"

SRC = DATA_ROOT + r"\111_individual_dislocations_10x_focusing_2\scan0001\pco_ff_0000.h5"
DARK = DATA_ROOT + r"\bicrystal_111_layer_rocking_scans_darks\scan0001\pco_ff_0000.h5"
# write straight into docs/img/ where detector-noise-model.md embeds it
OUT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "img"))
# --------------------------------------------------------------------------

OFFSET = 110.0  # ADU at 1 s (102.5 + 7.5/s, see darks + bicrystal floors)
RN2_GUESS = 14.0  # ADU^2, from darks temporal sigma ~3.7
G_GUESS = 4.0  # ADU/photon rough prior for the mask pre-pass

with h5py.File(DARK, "r") as f:
    dmu = f["entry_0000/ESRF-ID03/pco_ff/data"][()].astype(np.float32).mean(axis=0)
hot = dmu > (np.median(dmu) + 10)
edge = np.zeros_like(hot)
edge[:16, :] = True
edge[-16:, :] = True
print(f"masked: {hot.sum()} hot px, plus 32 edge rows")

bins = np.concatenate([[20], np.geomspace(40, 6000, 25)])
cnt = np.zeros(len(bins) - 1)
sum_s = np.zeros(len(bins) - 1)
sum_d2 = np.zeros(len(bins) - 1)

with h5py.File(SRC, "r") as f:
    data = f["entry_0000/ESRF-ID03/pco_ff/data"]
    n = data.shape[0]
    prev = data[0].astype(np.float32)
    for i in range(1, n):
        cur = data[i].astype(np.float32)
        # global drift normalization on the illuminated mid band
        band = np.s_[600:1500, :]
        r = (cur[band].sum()) / (prev[band].sum())
        d = cur / r - prev
        s = 0.5 * (cur / r + prev) - OFFSET

        # scene-change mask: boxcar-smoothed difference should be ~0 for
        # pure noise (boxcar of k^2 px shrinks noise by k); threshold at 3x
        k = 7
        dsm = uniform_filter(d, k)
        sig_d = np.sqrt(2.0 * (RN2_GUESS + G_GUESS * np.clip(s, 0, None)))
        good = (np.abs(dsm) < 3.0 * sig_d / k) & ~hot & ~edge & (s > 0)

        sv = s[good]
        dv = d[good]
        idx = np.searchsorted(bins, sv) - 1
        ok = (idx >= 0) & (idx < len(cnt))
        np.add.at(cnt, idx[ok], 1)
        np.add.at(sum_s, idx[ok], sv[ok])
        np.add.at(sum_d2, idx[ok], dv[ok] ** 2)
        prev = cur

m = cnt > 5000
s_bin = sum_s[m] / cnt[m]
var_single = sum_d2[m] / cnt[m] / 2.0  # var(d) = 2 var(single)

print("\n  s(ADU)   var(ADU^2)   var/s")
for sb, vb in zip(s_bin, var_single, strict=False):
    print(f"{sb:9.1f} {vb:11.1f} {vb / sb:7.3f}")

# weighted linear fit var = RN2 + g*s over the well-populated range
w = np.sqrt(cnt[m])
A = np.vstack([np.ones_like(s_bin), s_bin]).T
coef, *_ = np.linalg.lstsq(A * w[:, None], var_single * w, rcond=None)
rn2, g = coef
print(f"\nFIT: RN^2 = {rn2:.1f} ADU^2  (RN = {np.sqrt(max(rn2, 0)):.2f} ADU)")
print(f"     gain g = {g:.3f} ADU per incident-photon-equivalent")
print(f"     -> photons at s=1000 ADU: {1000 / g:.0f}, SNR = {1000 / np.sqrt(rn2 + g * 1000):.1f}")

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(s_bin, var_single, "o", label="measured (robust, scene-masked)")
ss = np.geomspace(s_bin.min(), s_bin.max(), 100)
ax.loglog(ss, rn2 + g * ss, "-", label=f"fit: {rn2:.1f} + {g:.2f}·s")
ax.axhline(rn2, color="gray", ls=":", label=f"readout floor RN²={rn2:.1f}")
ax.set_xlabel("signal above offset s (ADU)")
ax.set_ylabel("single-frame variance (ADU²)")
ax.set_title(
    "PCO Edge 4.2 bi photon-transfer curve\n(61-frame focus sweep, adjacent-pair differences)"
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out_path = _os.path.join(OUT, "photon_transfer_fit.png")
fig.savefig(out_path, dpi=110)
print(f"saved figure -> {out_path}")
