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
    # #16 Slice 5: state is threaded as an explicit ForwardContext (the
    # module globals Resq_i/Hg/q_hkl are gone). The pickle loader itself was
    # removed in v1.0.3, so this can no longer reproduce the original
    # pickle-era output bit-for-bit — it captures the CURRENT forward output
    # for the default (-1,1,-1) 17 keV reflection.
    from dfxm_geo.pipeline import (
        ReciprocalConfig,
        SimulationConfig,
        _lookup_and_load_kernel,
        run_theta,
    )

    res = _lookup_and_load_kernel((-1, 1, -1), 17.0)
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    ctx = fm.build_forward_context(run_theta(cfg), res, (-1, 1, -1))
    Hg, _ = fm.Find_Hg(fm.dis, fm.ndis, fm.psize, fm.zl_rms, ctx=ctx)

    # Fixed (phi, chi) chosen to exercise the typical Bragg condition path.
    phi = 0.0
    chi = 0.0
    out = fm.forward(Hg, ctx, phi=phi, chi=chi)

    if isinstance(out, tuple):
        out = out[0]

    dst = Path("tests/data/golden/forward_snapshot_pickle_era.npy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    print(f"wrote {dst} shape={out.shape} dtype={out.dtype} sum={out.sum():.6e}")


if __name__ == "__main__":
    main()
