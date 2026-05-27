#!/usr/bin/env python
"""Generate a starter config sweep for the in-node fan-out launcher.

Emits forward-config TOMLs into ``configs/sweep/`` for ``scripts/fanout.py`` /
``lsf/fanout.bsub`` (which default to ``MANIFEST=configs/sweep``). This first
sweep varies ``random_dislocations`` ndis x seed; every config sets
``write_strain_provenance = false`` (drops the ~106 MB/config Hg dump -> slim
HDF5) and a modest rocking phi scan. The ``[reciprocal]`` block mirrors
``configs/profile_rocking.toml`` so the bootstrapped Al 111 @ 17 keV kernel
applies (run ``dfxm-bootstrap --if-missing --config configs/profile_rocking.toml``
first on a fresh machine).

Run:  python scripts/gen_sweep_configs.py

This is the *starter* for the 100k-image ML sweep — extend ``NDIS``/``SEEDS``
(or add structure variation) to scale up. With 8 configs x 21 frames this is
~168 images; the real run grows the grids until ``8 configs x frames`` per wave
fills the node and the total reaches ~100k.
"""

from __future__ import annotations

from pathlib import Path

# Sweep grid — extend these for the real 100k run.
NDIS = [2, 4, 8, 16]
SEEDS = [1, 2]

# Mirrors configs/profile_rocking.toml [reciprocal] so the bootstrapped
# Al 111 @ 17 keV kernel matches. beamstop=true -> MC kernel path.
_RECIPROCAL = """\
[reciprocal]
hkl         = [-1, 1, -1]
keV         = 17.0
Nrays       = 100_000_000
npoints1    = 400
npoints2    = 200
npoints3    = 200
qi1_range   = 5e-4
qi2_range   = 7.5e-3
qi3_range   = 7.5e-3
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.1106382978723403e-4
eps_rms     = 6.0e-5
D           = 0.000565685424949238
d1          = 0.274
beamstop    = true
bs_height   = 25e-3
aperture    = true
knife_edge  = false
dphi_range  = 0.0
"""


def config_text(ndis: int, seed: int) -> str:
    """One forward-config TOML for the given dislocation count + seed."""
    return (
        _RECIPROCAL
        + "\n[scan.phi]\nrange = 6e-4\nsteps = 21\n"
        + '\n[crystal]\nmode = "random_dislocations"\n'
        + "[crystal.random_dislocations]\n"
        + f"ndis = {ndis}\nsigma = 5.0\nmin_distance = 4.0\nseed = {seed}\n"
        # write_strain_provenance=false drops the per-config Hg dump (~106 MB
        # -> ~0.14 MB); this is a batch/ML run, not a reproducibility archive.
        + "\n[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n"
        + "\n[postprocess]\nenabled = false\n"
    )


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "configs" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ndis in NDIS:
        for seed in SEEDS:
            path = out_dir / f"random_ndis{ndis:02d}_seed{seed}.toml"
            # newline="\n": keep LF on Windows too, so regenerating doesn't
            # dirty the tree against the committed (LF) configs.
            path.write_text(config_text(ndis, seed), encoding="utf-8", newline="\n")
            n += 1
    print(f"wrote {n} configs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
