#!/usr/bin/env python
"""Compare the Monte-Carlo and closed-form reciprocal-space resolution backends.

Renders a 2-row x 5-column figure: the top row is the Monte-Carlo central-slice
profile of each q-component, the bottom row is the closed-form analytic density
evaluated along the same axis. Columns are

    qrock | qroll | qpar | qrock' | q2theta

i.e. the three crystal-frame components plus the two imaging-frame components
(qroll is shared by both frames). The analytic backend is overlaid faintly on
the MC panels for a direct visual residual; each column is annotated with its
peak max|MC - analytic| deviation and a summary table prints to stdout.

The analytic backend only exists for the *no-beamstop* regime (it raises if a
beamstop is configured), so the comparison is run with the beamstop OFF. Both
backends share one Bragg angle and one set of instrument parameters, sourced
from ``ReciprocalConfig`` / ``generate_kernel`` so the two sides cannot drift.

Memory stays flat regardless of ``--nrays``: rays are sampled in chunks
(``--chunk``, default 1e8), histogrammed, and discarded. A chunk of 1e8 needs
roughly 10-12 GB of RAM; lower ``--chunk`` if a node is tight.

Run on the cluster, then view the saved PNG over VSCode Remote-SSH. See
``docs/resolution-backend-comparison.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dfxm_geo.crystal.cell import UnitCell  # noqa: E402
from dfxm_geo.pipeline import ReciprocalConfig  # noqa: E402
from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution  # noqa: E402
from dfxm_geo.reciprocal_space.kernel import (  # noqa: E402
    _validate_reflection,
    generate_kernel,
)
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func  # noqa: E402

# Al lattice parameter (m); same constant the pipeline passes to
# _validate_reflection (see pipeline.py ReciprocalConfig.__post_init__).
_A_AL = 4.0495e-10

# Index of each component in the reciprocal_res_func return tuple
# (qrock, qroll, qpar, qrock_prime, q2th, delta_2theta).
_RAY_IDX = {"qrock": 0, "qroll": 1, "qpar": 2, "qrock_prime": 3, "q2th": 4}

# Per-column spec: the component to slice, the two components held near zero to
# define the central slice, a display label, and the analytic eval direction
# expressed in the imaging frame (qrock', qroll, q2th) as a function of theta.
# The crystal-frame axes map into the imaging frame by the theta-rotation:
#   qrock' = cos(th) qrock + sin(th) qpar,  q2th = -sin(th) qrock + cos(th) qpar.
_COLUMNS = [
    {
        "name": "qrock",
        "others": ("qpar", "qroll"),
        "label": r"$q_{rock}$ (crystal)",
        "line": lambda t, th: np.vstack([np.cos(th) * t, 0 * t, -np.sin(th) * t]),
    },
    {
        "name": "qroll",
        "others": ("qrock", "qpar"),
        "label": r"$q_{roll}$",
        "line": lambda t, th: np.vstack([0 * t, t, 0 * t]),
    },
    {
        "name": "qpar",
        "others": ("qrock", "qroll"),
        "label": r"$q_{par}$ (crystal)",
        "line": lambda t, th: np.vstack([np.sin(th) * t, 0 * t, np.cos(th) * t]),
    },
    {
        "name": "qrock_prime",
        "others": ("qroll", "q2th"),
        "label": r"$q'_{rock}$ (imaging)",
        "line": lambda t, th: np.vstack([t, 0 * t, 0 * t]),
    },
    {
        "name": "q2th",
        "others": ("qrock_prime", "qroll"),
        "label": r"$q_{2\theta}$ (imaging)",
        "line": lambda t, th: np.vstack([0 * t, 0 * t, t]),
    },
]


def resolve_params(hkl: tuple[int, int, int], keV: float) -> dict[str, float]:
    """Collect the shared physics, sourced from the production config/kernel.

    theta from the reflection; the four resolution sigmas + the zeta_v clip from
    ``ReciprocalConfig`` defaults; ``phys_aper = D/d1`` from ``generate_kernel``.
    """
    theta = _validate_reflection(hkl, keV, UnitCell.cubic(_A_AL))
    cfg = ReciprocalConfig()  # defaults mirror generate_kernel's instrument args
    gk = inspect.signature(generate_kernel).parameters
    phys_aper = float(gk["D"].default) / float(gk["d1"].default)
    return {
        "theta": theta,
        "zeta_v_fwhm": cfg.zeta_v_fwhm,
        "zeta_h_fwhm": cfg.zeta_h_fwhm,
        "NA_rms": cfg.NA_rms,
        "eps_rms": cfg.eps_rms,
        "zeta_v_clip": cfg.zeta_v_clip,
        "phys_aper": phys_aper,
    }


def sample_rays(n: int, p: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw ``n`` MC rays via the production sampler; return the 5 q-components.

    Calls ``reciprocal_res_func`` (beamstop off) with a trivial internal grid --
    we only want its ray clouds, not its Resq_i histogram. Its chatty prints are
    swallowed so the chunk loop stays readable.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        out = reciprocal_res_func(
            n,
            2,
            2,
            2,  # dummy npoints -- internal Resq_i is discarded
            1.0,
            1.0,
            1.0,  # dummy qi ranges
            plot_figs=False,
            save_resqi=False,
            zeta_v_fwhm=p["zeta_v_fwhm"],
            zeta_h_fwhm=p["zeta_h_fwhm"],
            NA_rms=p["NA_rms"],
            eps_rms=p["eps_rms"],
            theta=p["theta"],
            phys_aper=p["phys_aper"],
            date="compare",
            rng=rng,
            return_qs=True,
            beamstop=False,
        )
    assert out is not None  # return_qs=True always returns the tuple
    return {name: out[idx] for name, idx in _RAY_IDX.items()}


def pilot_ranges(
    p: dict[str, float], rng: np.random.Generator, n_pilot: int, quantile: float
) -> dict[str, float]:
    """Symmetric half-width per component from a small pilot draw.

    Uses a high quantile of |component| so a few extreme rays don't blow the
    axis out. These half-widths set both the plot x-range and (scaled by
    ``slab_frac``) the central-slice thickness on the held-out axes.
    """
    rays = sample_rays(n_pilot, p, rng)
    return {name: float(np.quantile(np.abs(v), quantile)) for name, v in rays.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nrays", type=float, default=1e8, help="total MC rays (default 1e8)")
    ap.add_argument("--chunk", type=float, default=1e8, help="rays per chunk (default 1e8)")
    ap.add_argument("--nbins", type=int, default=151, help="histogram bins per component")
    ap.add_argument(
        "--slab-frac",
        type=float,
        default=0.05,
        help="central-slice half-thickness as a fraction of each held-out axis half-width",
    )
    ap.add_argument("--pilot", type=float, default=2e6, help="pilot rays for auto-ranging")
    ap.add_argument(
        "--quantile", type=float, default=0.999, help="abs-quantile for the axis half-width"
    )
    ap.add_argument("--hkl", type=int, nargs=3, default=(-1, 1, -1), metavar=("H", "K", "L"))
    ap.add_argument("--keV", type=float, default=17.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/img/resolution_backend_comparison.png"),
        help="output PNG path",
    )
    args = ap.parse_args()

    nrays = int(args.nrays)
    chunk = int(args.chunk)
    nbins = int(args.nbins)
    hkl = tuple(args.hkl)
    rng = np.random.default_rng(args.seed)

    p = resolve_params(hkl, args.keV)
    print(
        f"hkl={hkl} keV={args.keV:g}  theta={math.degrees(p['theta']):.3f} deg  "
        f"beamstop=OFF  Nrays={nrays:.2e} in chunks of {chunk:.2e}"
    )

    # --- auto-range from a pilot draw ---
    half = pilot_ranges(p, rng, int(args.pilot), args.quantile)
    edges = {c["name"]: np.linspace(-half[c["name"]], half[c["name"]], nbins + 1) for c in _COLUMNS}
    centers = {name: 0.5 * (e[:-1] + e[1:]) for name, e in edges.items()}
    slab = {name: args.slab_frac * half[name] for name in half}  # central-slice half-thickness
    mc_counts = {c["name"]: np.zeros(nbins, dtype=np.float64) for c in _COLUMNS}

    # --- chunked MC accumulation ---
    nchunks = math.ceil(nrays / chunk)
    done = 0
    for i in range(nchunks):
        n = min(chunk, nrays - done)
        rays = sample_rays(n, p, rng)
        for col in _COLUMNS:
            name = col["name"]
            mask = np.ones(n, dtype=bool)
            for other in col["others"]:
                mask &= np.abs(rays[other]) < slab[other]
            mc_counts[name] += np.histogram(rays[name][mask], bins=edges[name])[0]
        done += n
        print(f"  [chunk {i + 1}/{nchunks}] +{n:.2e} rays -> total {done:.2e}")

    # --- analytic backend, evaluated along each axis ---
    analytic = AnalyticResolution(
        theta=p["theta"],
        zeta_v_fwhm=p["zeta_v_fwhm"],
        zeta_h_fwhm=p["zeta_h_fwhm"],
        NA_rms=p["NA_rms"],
        eps_rms=p["eps_rms"],
        zeta_v_clip=p["zeta_v_clip"],
    )

    # --- normalize, compute residuals ---
    mc_norm: dict[str, np.ndarray] = {}
    ana_norm: dict[str, np.ndarray] = {}
    resid: dict[str, float] = {}
    for col in _COLUMNS:
        name = col["name"]
        t = centers[name]
        mc = mc_counts[name]
        mc = mc / mc.max() if mc.max() > 0 else mc
        qi = col["line"](t, p["theta"])
        ana = analytic(qi)
        ana = ana / ana.max() if ana.max() > 0 else ana
        mc_norm[name] = mc
        ana_norm[name] = ana
        resid[name] = float(np.max(np.abs(mc - ana)))

    # --- summary table ---
    print("\n  component        max|MC - analytic|")
    print("  " + "-" * 38)
    for col in _COLUMNS:
        print(f"  {col['name']:<14}   {resid[col['name']] * 100:6.2f} %")

    # --- figure: 2 rows (MC / analytic) x 5 cols ---
    fig, axes = plt.subplots(2, len(_COLUMNS), figsize=(4 * len(_COLUMNS), 7), sharex="col")
    for j, col in enumerate(_COLUMNS):
        name = col["name"]
        t = centers[name] * 1e4  # display in 1e-4 inverse-Angstrom units
        ax_mc, ax_an = axes[0, j], axes[1, j]

        ax_mc.step(t, mc_norm[name], where="mid", color="#1f77b4", lw=1.4, label="MC")
        ax_mc.plot(t, ana_norm[name], "--", color="#d62728", lw=1.0, alpha=0.8, label="analytic")
        ax_mc.set_title(f"{col['label']}\nmax|$\\Delta$| = {resid[name] * 100:.2f}%", fontsize=10)
        ax_mc.grid(alpha=0.3)

        ax_an.plot(t, ana_norm[name], "-", color="#d62728", lw=1.6, label="analytic")
        ax_an.set_xlabel(r"$q\ /\ 10^{-4}\ \AA^{-1}$", fontsize=9)
        ax_an.grid(alpha=0.3)
        if j == 0:
            ax_mc.set_ylabel("Monte Carlo\n(peak-norm)", fontsize=10)
            ax_an.set_ylabel("Analytic\n(peak-norm)", fontsize=10)
            ax_mc.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Resolution backends: MC (top) vs analytic (bottom)  --  "
        f"hkl={hkl}, {args.keV:g} keV, Nrays={nrays:.1e}, beamstop OFF",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Expected differences: analytic drops the NA square aperture (fuller MC tails in "
        r"$q_{roll}$/$q_{2\theta}$); $\zeta_v$ hard-cut at $\pm$140 $\mu$rad in both.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
