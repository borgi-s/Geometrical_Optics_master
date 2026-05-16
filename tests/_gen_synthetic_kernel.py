"""One-shot generator for the synthetic cross-platform kernel golden.

A 4x4x4 deterministic synthetic Resq_i plus bundled scalar params, used as
the cross-platform reference in tests/test_kernel_format.py.

Run from the repo root:

    python -m tests._gen_synthetic_kernel

Output: tests/data/golden/synthetic_kernel.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(20260516)
    Resq_i = rng.uniform(0.0, 1.0, size=(4, 4, 4)).astype(np.float64)

    dst = Path("tests/data/golden/synthetic_kernel.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        Resq_i=Resq_i,
        qi1_range=np.float64(5e-4),
        qi2_range=np.float64(0.75e-2),
        qi3_range=np.float64(0.75e-2),
        npoints1=np.int64(4),
        npoints2=np.int64(4),
        npoints3=np.int64(4),
        Nrays=np.int64(1_000),
    )
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
