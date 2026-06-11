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
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
) -> str:
    """Return a TOML string for one identify-sweep config."""
    rdp = "true" if render_per_dislocation else "false"
    return f"""\
mode = "multi"

[reciprocal]
hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]
keV = {keV}
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
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reflections: list[tuple[int, int, int]]
    if args.hkl_list is not None:
        reflections = _parse_hkl_list(args.hkl_list)
        use_hkl_in_filename = True
    else:
        reflections = [(-1, 1, -1)]
        use_hkl_in_filename = False

    n_written = 0
    for hkl in reflections:
        for s in range(1, args.n_configs + 1):
            if use_hkl_in_filename:
                fname = out_dir / f"multi_{_hkl_token(hkl)}_seed{s:05d}.toml"
            else:
                fname = out_dir / f"multi_seed{s:05d}.toml"
            text = config_text(
                s,
                n_samples=args.n_samples,
                nrays=args.nrays,
                render_per_dislocation=args.render_per_dislocation,
                phi_value=args.phi_value,
                phi_range=args.phi_range,
                phi_steps=args.phi_steps,
                hkl=hkl,
                keV=args.keV,
            )
            # newline="\n": keep LF on Windows too, so regenerating doesn't dirty the tree.
            fname.write_text(text, encoding="utf-8", newline="\n")
            n_written += 1

    frames_per_scene = max(args.phi_steps, 1)
    total_images = n_written * args.n_samples * frames_per_scene
    if args.phi_steps > 1:
        lo, hi = args.phi_value - args.phi_range, args.phi_value + args.phi_range
        phi_desc = f"phi rocking {lo * 1e6:.0f}->{hi * 1e6:.0f} urad x {args.phi_steps} steps"
    else:
        phi_desc = f"phi fixed at {args.phi_value * 1e6:.0f} urad"
    if use_hkl_in_filename:
        configs_desc = (
            f"{n_written} configs ({args.n_configs} seeds x {len(reflections)} reflections)"
        )
    else:
        configs_desc = f"{n_written} configs"
    print(
        f"wrote {configs_desc} to {out_dir}  "
        f"(n_samples={args.n_samples}, {phi_desc}, nrays={args.nrays}, "
        f"render_per_dislocation={args.render_per_dislocation})\n"
        f"total images = {n_written} configs x {args.n_samples} scenes "
        f"x {frames_per_scene} frames = {total_images:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
