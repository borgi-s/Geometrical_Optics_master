"""One-shot generator for the pickle-era forward() snapshot golden.

Run once before the kernel pickle → npz migration to capture a deterministic
forward() output using the CURRENT pickle-based loader. The resulting .npy
becomes the bit-equivalence reference for test_kernel_format.py:
test_forward_output_matches_pickle_era_snapshot.

Run from the repo root:

    python -m tests._gen_forward_snapshot_pickle_era

Output: tests/data/golden/forward_snapshot_pickle_era.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dfxm_geo.direct_space.forward_model as fm


def main() -> None:
    if fm.Resq_i is None:
        raise RuntimeError(
            "Resq_i not loaded; this generator requires a kernel to be loaded first. "
            "Call pipeline._lookup_and_load_kernel(hkl, keV) before running this script."
        )
    if fm.Hg is None:
        raise RuntimeError(
            "Hg not populated; the auto-load block at forward_model.py:412-418 "
            "must have run with compute_Hg=True. Re-import the module fresh "
            "and re-run."
        )

    # Fixed (phi, chi) chosen to exercise the typical Bragg condition path.
    # Values match the snap_old.npy snapshot taken during Phase 8 Round 25
    # to keep this reference comparable to pre-existing dev snapshots.
    phi = 0.0
    chi = 0.0
    out = fm.forward(fm.Hg, phi=phi, chi=chi)

    if isinstance(out, tuple):
        out = out[0]

    dst = Path("tests/data/golden/forward_snapshot_pickle_era.npy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    print(f"wrote {dst} shape={out.shape} dtype={out.dtype} sum={out.sum():.6e}")


if __name__ == "__main__":
    main()
