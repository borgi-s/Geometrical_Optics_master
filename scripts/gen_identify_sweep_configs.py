#!/usr/bin/env python
"""Generate identify-mode (multi) configs for a cluster-scale sweep.

Each config mirrors the structure of
``examples/identification_ml_tutorial/demo_multi.toml`` and varies only the
``[noise] rng_seed`` (seeds 1..N).  All other parameters are fixed so the
bootstrapped Al 111 @ 17 keV kernel applies to every config in the sweep.

Total images = n_configs x n_samples (scenes/config) x phi_steps (frames/scene).
Example — 110k images as 5000 seeds x 2 scenes x an 11-step phi rocking scan
(0 -> 250 µrad)::

    python scripts/gen_identify_sweep_configs.py --n-configs 5000 --n-samples 2

(the phi rocking defaults are 0 -> 250 µrad in 11 steps; pass --phi-steps 1 for
a single fixed-phi image per scene.)

Then fan out across the cluster with ``lsf/fanout.bsub`` (single node) or
``lsf/identify_sweep_array.bsub`` (array).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _phi_block(value: float, phi_range: float, steps: int) -> str:
    """[scan.phi] TOML block.

    A rocking scan (``value``/``range``/``steps``) when ``steps > 1``, else a
    single fixed phi (``value`` only). Phi runs ``linspace(value - range,
    value + range, steps)`` — i.e. ``value`` is the centre and ``range`` is the
    half-width. To rock 0 -> 250 µrad in 11 steps: value=1.25e-4, range=1.25e-4,
    steps=11.
    """
    if steps > 1:
        return f"[scan.phi]\nvalue = {value}\nrange = {phi_range}\nsteps = {steps}"
    return f"[scan.phi]\nvalue = {value}"


def config_text(
    seed: int,
    *,
    n_samples: int,
    nrays: int,
    render_per_dislocation: bool,
    phi_value: float,
    phi_range: float,
    phi_steps: int,
) -> str:
    """Return a TOML string for one identify-sweep config."""
    rdp = "true" if render_per_dislocation else "false"
    return f"""\
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
Nrays = {nrays}
seed = 0

{_phi_block(phi_value, phi_range, phi_steps)}

[noise]
poisson_noise = true
rng_seed = {seed}
intensity_scale = 7.0

[multi]
n_samples = {n_samples}
pos_std_um = 5.0
render_per_dislocation = {rdp}

[io]
dislocs_dirname = "identify_multi"
perfect_dirname = "ignored"
include_perfect_crystal = false
write_strain_provenance = false
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-configs", type=int, default=5000, help="Number of configs to generate (seeds 1..N)."
    )
    ap.add_argument("--n-samples", type=int, default=20, help="[multi] n_samples per config.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/identify_sweep"),
        help="Output directory for generated TOML files.",
    )
    ap.add_argument(
        "--nrays", type=int, default=100_000_000, help="[reciprocal] Nrays for kernel quality."
    )
    ap.add_argument(
        "--render-per-dislocation",
        action="store_true",
        default=False,
        help="Enable [multi] render_per_dislocation (noiseless per-instance renders).",
    )
    ap.add_argument("--phi-value", type=float, default=1.25e-4, help="[scan.phi] centre (radians).")
    ap.add_argument(
        "--phi-range",
        type=float,
        default=1.25e-4,
        help="[scan.phi] half-width (radians); phi spans [value-range, value+range].",
    )
    ap.add_argument(
        "--phi-steps",
        type=int,
        default=11,
        help="[scan.phi] rocking steps (>1 = rocking scan; 1 = fixed phi).",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in range(1, args.n_configs + 1):
        fname = out_dir / f"multi_seed{s:05d}.toml"
        text = config_text(
            s,
            n_samples=args.n_samples,
            nrays=args.nrays,
            render_per_dislocation=args.render_per_dislocation,
            phi_value=args.phi_value,
            phi_range=args.phi_range,
            phi_steps=args.phi_steps,
        )
        # newline="\n": keep LF on Windows too, so regenerating doesn't dirty the tree.
        fname.write_text(text, encoding="utf-8", newline="\n")

    frames_per_scene = max(args.phi_steps, 1)
    total_images = args.n_configs * args.n_samples * frames_per_scene
    if args.phi_steps > 1:
        lo, hi = args.phi_value - args.phi_range, args.phi_value + args.phi_range
        phi_desc = f"phi rocking {lo * 1e6:.0f}->{hi * 1e6:.0f} urad x {args.phi_steps} steps"
    else:
        phi_desc = f"phi fixed at {args.phi_value * 1e6:.0f} urad"
    print(
        f"wrote {args.n_configs} configs to {out_dir}  "
        f"(n_samples={args.n_samples}, {phi_desc}, nrays={args.nrays}, "
        f"render_per_dislocation={args.render_per_dislocation})\n"
        f"total images = {args.n_configs} configs x {args.n_samples} scenes "
        f"x {frames_per_scene} frames = {total_images:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
