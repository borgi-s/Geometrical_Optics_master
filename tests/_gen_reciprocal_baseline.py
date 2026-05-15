"""One-shot golden generator for tests/test_reciprocal_resolution.py.

Run once with `python -m tests._gen_reciprocal_baseline` from the repo root.
Output saved to tests/data/golden/reciprocal_baseline.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def main() -> None:
    rng = np.random.default_rng(20260513)
    result = reciprocal_res_func(
        Nrays=10_000,
        npoints1=40,
        npoints2=30,
        npoints3=30,
        qi1_range=5e-4,
        qi2_range=7.5e-3,
        qi3_range=7.5e-3,
        plot_figs=False,
        save_resqi=False,
        zeta_v_fwhm=5.3e-04,
        zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=1.41e-4 / 2.35,
        theta=0.15662,
        phys_aper=(2 * np.sqrt(50e-6 * 1.6e-3)) / 0.274,
        date="golden",
        rng=rng,
        return_qs=True,
    )
    assert result is not None
    qrock, qroll, qpar, qrock_prime, q2th, delta_2theta = result
    out = Path(__file__).parent / "data" / "golden" / "reciprocal_baseline.npz"
    np.savez(
        out,
        qrock=qrock,
        qroll=qroll,
        qpar=qpar,
        qrock_prime=qrock_prime,
        q2th=q2th,
        delta_2theta=delta_2theta,
    )
    print(f"Wrote {out} (sizes: qrock={qrock.shape})")


if __name__ == "__main__":
    main()
