#!/usr/bin/env python
"""Single source of truth for the FCC Al / BCC W / HCP Ti crystal-structure
showcase: deterministic weak-beam DFXM render of one pure-edge dislocation per
structure on the analytic backend.

Used by:
  * tests/test_structure_goldens.py    (locks the raw render per structure)
  * docs figure generation             (--figures -> docs/img/)
  * papers/.../scripts/make_showcase.py (the published paper figure)

All three share the recipe builders below so the golden, the docs figure, and
the paper figure can never drift. Run with the project venv python:

    python scripts/render_structure_showcase.py --figures   # docs/img/*.png|pdf
    python scripts/render_structure_showcase.py --golden    # tests/.../*.npy
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent  # Geometrical_Optics_master/
DOCS_IMG = _REPO_ROOT / "docs" / "img"
GOLDEN_DIR = _REPO_ROOT / "tests" / "data" / "golden" / "structure_showcase"

KEV = 17.0
TAGS = ("fcc", "bcc", "hcp")
PANEL_TITLES = {
    "fcc": ("FCC Al", r"$(\bar{1}1\bar{1})$, edge"),
    "bcc": ("BCC W", r"$(200)$, edge"),
    "hcp": ("HCP Ti", r"$(10\bar{1}1)$, edge"),
}

RESBLOCK = (
    'backend = "analytic"\nbeamstop = false\naperture = false\n'
    "zeta_v_fwhm = 5.3e-4\nzeta_h_fwhm = 0.0\n"
    "NA_rms = 3.1106382978723403e-4\neps_rms = 6.0e-5\n"
)
IO_BLOCK = (
    '[io]\nftype = ".h5"\ndislocs_dirname = "dis"\nperfect_dirname = "ignored"\n'
    "include_perfect_crystal = false\nwrite_strain_provenance = false\n"
)
DET_IDEAL = '[detector]\nmodel = "ideal"\n'


def _vec(t) -> str:
    return f"[{int(t[0])}, {int(t[1])}, {int(t[2])}]"


def sweep_block(plane) -> str:
    # angle 0 deg == pure edge; a single configuration, no slip-plane sweep,
    # no RNG draw -> deterministic raw render.
    return (
        f"slip_plane_normal = {_vec(plane)}\n"
        "sweep_all_slip_planes = false\n"
        "angle_start_deg = 0.0\nangle_stop_deg = 0.0\nangle_step_deg = 30.0\n"
        "exclude_invisibility = false\n"
    )


def eta_for(mount, hkl):
    from dfxm_geo.crystal.oblique import compute_omega_eta

    g = compute_omega_eta(mount, hkl, KEV)
    for eta in (g.eta_1, g.eta_2):
        if not np.isnan(eta):
            return float(eta)
    return None


def fcc_toml() -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((-1, 1, -1))}\nkeV = {KEV}\n{RESBLOCK}\n"
        '[geometry]\nmode = "simplified"\n\n'
        # Classic validated FCC path: no structure_type / material in simplified
        # mode (the bare default is FCC aluminium, nu = 0.334).
        '[crystal]\nlattice = "cubic"\na = 4.05e-10\n'
        f"{sweep_block((1, 1, 1))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def bcc_toml(eta) -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((2, 0, 0))}\nkeV = {KEV}\n{RESBLOCK}\n"
        f'[geometry]\nmode = "oblique"\neta = {eta!r}\n\n'
        '[crystal]\nlattice = "cubic"\na = 3.1652e-10\nstructure_type = "bcc"\n'
        'material = "W"\npoisson_ratio = 0.28\n'
        "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        f"{sweep_block((1, -1, 0))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def hcp_toml(eta) -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((1, 0, -1))}\nkeV = {KEV}\n{RESBLOCK}\n"
        f'[geometry]\nmode = "oblique"\neta = {eta!r}\n\n'
        '[crystal]\nlattice = "hexagonal"\na = 2.9505e-10\nc = 4.6826e-10\n'
        'structure_type = "hcp"\nmaterial = "Ti"\npoisson_ratio = 0.32\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        f"{sweep_block((1, 0, 1))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def build_toml(tag: str) -> str:
    """Return the TOML text for a structure tag, computing eta where needed."""
    from dfxm_geo.crystal.oblique import CrystalMount

    if tag == "fcc":
        return fcc_toml()
    if tag == "bcc":
        mount = CrystalMount(
            lattice="cubic",
            a=3.1652e-10,
            structure_type="bcc",
            mount_x=(1, 0, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )
        eta = eta_for(mount, (2, 0, 0))
        if eta is None:
            raise RuntimeError("BCC W (200) reflection not reachable at 17 keV")
        return bcc_toml(eta)
    if tag == "hcp":
        mount = CrystalMount(
            lattice="hexagonal",
            a=2.9505e-10,
            c=4.6826e-10,
            structure_type="hcp",
            mount_x=(2, -1, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )
        eta = eta_for(mount, (1, 0, -1))
        if eta is None:
            raise RuntimeError("HCP Ti (10-1) reflection not reachable at 17 keV")
        return hcp_toml(eta)
    raise ValueError(f"unknown structure tag {tag!r} (expected one of {TAGS})")


def render_raw(tag: str, workdir: Path) -> np.ndarray:
    """Render one structure in-process and return the raw 2-D detector image.

    Deterministic: single pure-edge dislocation, analytic backend, ideal
    detector. Reads the single frame from the identify master's external link.
    """
    from dfxm_geo.pipeline import load_identification_config, run_identification

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_path = workdir / f"{tag}.toml"
    cfg_path.write_text(build_toml(tag), encoding="utf-8")
    out = workdir / f"out_{tag}"
    cfg = load_identification_config(cfg_path)
    run_identification(cfg, out)
    with h5py.File(out / "dfxm_identify.h5", "r") as f:
        return f["/1.1/instrument/dfxm_sim_detector/data"][0].astype(np.float64)


# --- figure assembly (publication render) ----------------------------------


def crop_to_feature(img, margin=22, min_h=120, min_w=120):
    bg = float(np.median(img))
    dev = np.abs(img.astype(float) - bg)
    thr = np.percentile(dev, 99.0)
    ys, xs = np.where(dev >= max(thr, 1e-12))
    h, w = img.shape
    if len(ys) < 12:
        s = min(h, w)
        cy, cx = h // 2, w // 2
        return img[max(0, cy - s // 2) : cy + s // 2, max(0, cx - s // 2) : cx + s // 2]
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half_h = max((y1 - y0) // 2 + margin, min_h // 2)
    half_w = max((x1 - x0) // 2 + margin, min_w // 2)
    return img[max(0, cy - half_h) : min(h, cy + half_h), max(0, cx - half_w) : min(w, cx + half_w)]


def norm01(crop):
    lo, hi = np.percentile(crop, 1.0), np.percentile(crop, 99.5)
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((crop.astype(float) - lo) / (hi - lo), 0.0, 1.0)


def assemble_figure(images: dict, out_pdf: Path) -> None:
    """Assemble the multi-panel showcase figure + per-panel PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})
    tags = [t for t in TAGS if images.get(t) is not None]
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n, 2.9))
    if n == 1:
        axes = [axes]
    last_im = None
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    for ax, tag in zip(axes, tags, strict=True):
        disp = norm01(crop_to_feature(images[tag]))
        last_im = ax.imshow(
            disp,
            cmap="gray",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )
        name, sub = PANEL_TITLES[tag]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\n{sub}", fontsize=9.5)
        # Per-panel PNG (for notebook/doc reuse).
        pfig, pax = plt.subplots(figsize=(2.6, 2.6))
        pax.imshow(
            disp,
            cmap="gray",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )
        pax.set_xticks([])
        pax.set_yticks([])
        pfig.savefig(out_pdf.parent / f"showcase_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(pfig)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.045, pad=0.03, ticks=[0, 0.5, 1.0])
        cbar.set_label("normalized weak-beam intensity", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Weak-beam DFXM contrast of an edge dislocation across crystal "
        "systems (17 keV, analytic backend)",
        fontsize=9,
        y=1.04,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="render docs/img figures")
    ap.add_argument("--golden", action="store_true", help="(re)generate test goldens")
    args = ap.parse_args()
    if not (args.figures or args.golden):
        ap.error("pass --figures and/or --golden")

    work = Path(tempfile.mkdtemp(prefix="dfxm_showcase_"))
    print("workdir:", work)
    images = {}
    for tag in TAGS:
        print(f"[{tag}] rendering ...")
        try:
            images[tag] = render_raw(tag, work / tag)
            print(f"[{tag}] shape={images[tag].shape} std={images[tag].std():.4g}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            images[tag] = None

    if args.golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for tag in TAGS:
            if images[tag] is not None:
                np.save(GOLDEN_DIR / f"{tag}.npy", images[tag].astype(np.float32))
                print("wrote", GOLDEN_DIR / f"{tag}.npy")
    if args.figures:
        assemble_figure(images, DOCS_IMG / "showcase_fcc_bcc_hcp.pdf")
        print("wrote", DOCS_IMG / "showcase_fcc_bcc_hcp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
