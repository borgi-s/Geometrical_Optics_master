#!/usr/bin/env python
"""Generate identify-mode (multi) configs for a cluster-scale sweep.

Each config mirrors the structure of
``examples/identification_ml_tutorial/demo_multi.toml`` and varies only the
``[noise] rng_seed`` (seeds 1..N).  All other parameters are fixed so the
bootstrapped Al 111 @ 17 keV kernel applies to every config in the sweep.

Run::

    python scripts/gen_identify_sweep_configs.py --n-configs 5000

Then fan out across the cluster with ``lsf/identify_sweep_array.bsub``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def config_text(
    seed: int,
    *,
    n_samples: int,
    nrays: int,
    render_per_dislocation: bool,
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

[scan.phi]
value = 1.25e-4

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
        )
        # newline="\n": keep LF on Windows too, so regenerating doesn't dirty the tree.
        fname.write_text(text, encoding="utf-8", newline="\n")

    print(
        f"wrote {args.n_configs} configs to {out_dir}  "
        f"(n_samples={args.n_samples}, nrays={args.nrays}, "
        f"render_per_dislocation={args.render_per_dislocation})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
