"""One-shot: capture a tiny `forward()` stack via the current .npy writer.

Run BEFORE any HDF5 writer code lands. Output is committed as the
Layer 2 bit-equivalence golden in tests/data/golden/.

Usage:
    python tests/_gen_forward_legacy_golden.py

Design notes
------------
phi/chi ranges: The forward() rocking curve is narrow (~0.0006 rad half-width
for phi, ~0.0002 rad for chi). To keep all 4 linspace endpoints *inside* the
rocking curve (so the golden has non-zero signal in every frame), we use very
small half-ranges: 5e-5 rad ≈ 0.003° for both axes.  The default.toml values
(0.034°/0.115°) are the full experimental scan width; the endpoints of a 2-step
linspace over those ranges are at the rocking-curve edge → exactly zero.

Crop: The image is 510x170 px; the bright dislocation region at phi=chi≈0 peaks
near (row=326, col=55).  We crop rows [322:330], cols [51:59] — an 8x8 window
that contains substantial signal from all 4 frames (total per-frame sums ≈1800–
2500).  The top-left :8,:8 corner is background-only (all zeros).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.images import save_images_parallel

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden" / "forward_legacy_writer_4frames_8x8.npy"


def main() -> None:
    # Use the canonical kernel; assert it loaded so we don't silently call
    # forward() with Hg unset.
    if fm.Hg is None:
        raise SystemExit("Kernel didn't auto-load; run dfxm-bootstrap first or set DFXM_PKL_PATH.")

    S = SAMPLE_REMOUNT_OPTIONS["S1"]
    Hg, q_hkl = fm.Find_Hg(
        dis=4.0, ndis=151, psize=fm.psize, zl_rms=fm.zl_rms, S=S, remount_name="S1"
    )
    fm.Hg = Hg
    fm.q_hkl = q_hkl

    # phi/chi half-ranges: 5e-5 rad each.  Small enough that both linspace
    # endpoints (i.e. ±5e-5 rad) fall inside the rocking curve.
    _PHI_RAD = 5e-5
    _CHI_RAD = 5e-5
    PHI_RANGE_DEG = _PHI_RAD * 180 / np.pi  # ≈ 0.002865°
    CHI_RANGE_DEG = _CHI_RAD * 180 / np.pi  # ≈ 0.002865°

    with tempfile.TemporaryDirectory() as tmp:
        save_images_parallel(
            Hg,
            phi_range=PHI_RANGE_DEG,
            phi_steps=2,
            chi_range=CHI_RANGE_DEG,
            chi_steps=2,
            fpath=str(Path(tmp) / "stack"),
            fn_prefix="/golden_",
            ftype=".npy",
        )
        files = sorted((Path(tmp) / "stack").glob("*.npy"))
        if len(files) != 4:
            raise SystemExit(f"Expected 4 frames, got {len(files)}")
        arr = np.stack([np.load(f) for f in files])
        # Crop an 8x8 window centred on the bright dislocation region.
        # rows [322:330], cols [51:59] — all 4 frames have non-zero signal here.
        arr_small = arr[:, 322:330, 51:59]
        assert arr_small.sum() > 0, "Crop is all-zero — bright region has shifted?"
        np.save(GOLDEN, arr_small)
        print(
            f"Wrote {GOLDEN.relative_to(REPO)}, shape={arr_small.shape}, "
            f"dtype={arr_small.dtype}, size={GOLDEN.stat().st_size} bytes"
        )


if __name__ == "__main__":
    main()
