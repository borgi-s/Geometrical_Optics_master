#!/usr/bin/env python
"""Generate a representative config sweep for the in-node fan-out launcher.

Emits forward-config TOMLs into ``configs/sweep/`` for ``scripts/fanout.py`` /
``lsf/fanout.bsub`` (which default to ``MANIFEST=configs/sweep``). The sweep
varies ``random_dislocations`` ndis x seed; every config sets
``write_strain_provenance = false`` (drops the ~106 MB/config Hg dump -> slim
HDF5) and a 2D phi x chi "mosa" scan (PHI_STEPS x CHI_STEPS frames per config) so
per-config wall time is representative of a real DFXM mosaicity stack (and
amortizes the one-time numba JIT/import cost, which dominated the 21-frame
first-light run). The ``[reciprocal]`` block mirrors
``configs/profile_rocking.toml`` so the bootstrapped Al 111 @ 17 keV kernel
applies (run ``dfxm-bootstrap --if-missing --config configs/profile_rocking.toml``
first on a fresh machine).

Run:  python scripts/gen_sweep_configs.py

Sizing run: 8 configs x (21x21=441) frames = ~3528 images. Run it at
``-n 32`` / N_WORKERS=8 x THREADS_PER_WORKER=4 (full hpc node, warm cache) to
measure steady-state per-config throughput, then extend ``NDIS``/``SEEDS`` (or
add structure variation) until the total reaches ~100k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Sweep grid — extend these for the real 100k run.
NDIS = [2, 4, 8, 16]
SEEDS = [1, 2]

# 2D phi x chi "mosa" scan: n_frames = PHI_STEPS * CHI_STEPS per config.
PHI_STEPS = 21
CHI_STEPS = 21


def _parse_hkl_list(spec: str) -> list[tuple[int, int, int]]:
    """'1,1,1;2,0,0' → [(1,1,1), (2,0,0)]."""
    out = []
    for token in spec.split(";"):
        try:
            parts = [int(x) for x in token.split(",")]
        except ValueError:
            raise SystemExit(f"--hkl-list entry must be integers, got {token!r}") from None
        if len(parts) != 3:
            raise SystemExit(f"--hkl-list entry must have 3 indices, got {token!r}")
        h, k, l = parts
        out.append((h, k, l))
    return out


def _hkl_token(hkl: tuple[int, int, int]) -> str:
    """(-1,1,-1) → 'hkl_m1_1_m1' (m = minus; underscore-separated, collision-free)."""
    return "hkl_" + "_".join(f"m{abs(c)}" if c < 0 else str(c) for c in hkl)


# Mirrors configs/profile_rocking.toml [reciprocal] so the bootstrapped
# Al 111 @ 17 keV kernel matches. beamstop=true -> MC kernel path.
_RECIPROCAL_TEMPLATE = """\
[reciprocal]
hkl         = [{h0}, {h1}, {h2}]
keV         = {keV}
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

# Legacy constant kept for backward-compat (no callers, but don't break imports).
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


def config_text(
    ndis: int,
    seed: int,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
) -> str:
    """One forward-config TOML for the given dislocation count + seed."""
    reciprocal_block = _RECIPROCAL_TEMPLATE.format(h0=hkl[0], h1=hkl[1], h2=hkl[2], keV=keV)
    return (
        reciprocal_block
        + f"\n[scan.phi]\nrange = 6e-4\nsteps = {PHI_STEPS}\n"
        + f"\n[scan.chi]\nrange = 6e-4\nsteps = {CHI_STEPS}\n"
        + '\n[crystal]\nmode = "random_dislocations"\n'
        + "[crystal.random_dislocations]\n"
        + f"ndis = {ndis}\nsigma = 5.0\nmin_distance = 4.0\nseed = {seed}\n"
        # write_strain_provenance=false drops the per-config Hg dump (~106 MB
        # -> ~0.14 MB); this is a batch/ML run, not a reproducibility archive.
        + "\n[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n"
        + "\n[postprocess]\nenabled = false\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hkl-list",
        default=None,
        metavar="SPEC",
        help=(
            "Semicolon-separated reflections to sweep, e.g. '1,1,1;2,0,0' — "
            "adds a reflection axis to the sweep (multiplies configs by the "
            "number of reflections given). Without this flag, defaults to Al-111 "
            "[-1,1,-1] with legacy filenames."
        ),
    )
    ap.add_argument(
        "--keV",
        type=float,
        default=17.0,
        help="[reciprocal] X-ray energy in keV (default: 17.0).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for generated TOML files "
            "(default: configs/sweep/ relative to the repo root)."
        ),
    )
    args = ap.parse_args(argv)

    out_dir: Path = (
        args.out_dir
        if args.out_dir is not None
        else Path(__file__).resolve().parents[1] / "configs" / "sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reflections: list[tuple[int, int, int]]
    if args.hkl_list is not None:
        reflections = _parse_hkl_list(args.hkl_list)
        use_hkl_in_filename = True
    else:
        reflections = [(-1, 1, -1)]
        use_hkl_in_filename = False

    n = 0
    for hkl in reflections:
        for ndis in NDIS:
            for seed in SEEDS:
                if use_hkl_in_filename:
                    path = out_dir / f"random_{_hkl_token(hkl)}_ndis{ndis:02d}_seed{seed}.toml"
                else:
                    path = out_dir / f"random_ndis{ndis:02d}_seed{seed}.toml"
                # newline="\n": keep LF on Windows too, so regenerating doesn't
                # dirty the tree against the committed (LF) configs.
                path.write_text(
                    config_text(ndis, seed, hkl=hkl, keV=args.keV),
                    encoding="utf-8",
                    newline="\n",
                )
                n += 1
    print(f"wrote {n} configs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
