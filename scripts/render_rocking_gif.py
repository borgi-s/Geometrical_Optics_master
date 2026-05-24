"""Render docs/img/rocking_single_dislocation.gif for the README.

An animated rocking curve over a single centered edge dislocation: the forward
DFXM detector image is re-rendered as the goniometer tilt phi sweeps from
-150 to +150 microradians (chi = 0, Delta-theta = 0). As phi rocks through the
Bragg condition the strong-beam field brightens at phi = 0 and the dislocation
contrast localizes toward the core in the weak-beam wings.

The left panel is the detector image; the right panel is the rocking curve
itself -- the per-frame integrated intensity vs phi -- with a dot tracking the
phi of the image currently shown.

This is NOT part of CI (it runs the forward model end-to-end and loads the
reciprocal-space kernel). Regenerate manually after a change to the forward
model or when publishing a new README look:

    python scripts/render_rocking_gif.py

Defaults: single centered dislocation (canonical FCC primary), Al 111 @ 17 keV,
phi in [-150, +150] urad over 31 frames, ping-pong loop, fixed color scale.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from PIL import Image

import dfxm_geo.direct_space.forward_model as fm

# --- knobs -------------------------------------------------------------------
NPIXELS = 360  # detector size for the GIF (paper default is 510)
NSUB = 1
PHI_MAX_URAD = 150.0  # rocking half-range; ~within the rock-curve HWHM so the
#                        shared scale keeps both the intensity pulse and the
#                        dislocation contrast visible across the whole sweep
N_FRAMES = 31  # phi steps from -PHI_MAX to +PHI_MAX (10 urad steps)
NN3_MULT = 10  # multiply the default zl (beam-depth) ray count for finer
#                depth integration through the Gaussian beam profile
FPS = 12
SLOWDOWN = 5  # play each frame SLOWDOWN x longer (slower animation)
GAMMA = 0.45  # display power-law: lifts the weak-beam wings so the
#                        dislocation stays visible across the sweep while phi=0
#                        remains brightest (the rocking-curve pulse is preserved)
HKL = (-1, 1, -1)
KEV = 17.0


def _rescale(npix: int, nsub: int) -> None:
    """Rebuild every Npixels-derived forward_model global (verbatim formulas)."""
    fm.Npixels, fm.Nsub = npix, nsub
    fm.NN1 = int(npix // 3 * nsub)
    fm.NN2 = int(npix * nsub)
    fm.NN3 = int(npix // 30 * nsub) * NN3_MULT
    fm.yl_start = -fm.psize * npix / 2 + fm.psize / (2 * nsub)
    fm.xl_start = fm.yl_start / np.tan(2 * fm.theta_0) / 3
    fm.zl_start = -0.5 * fm.zl_rms * 6
    YI = (np.arange(fm.NN1) // nsub).repeat(fm.NN3 * fm.NN2)
    ZI = np.tile((np.arange(fm.NN2) // nsub).repeat(fm.NN3), fm.NN1)
    fm._flat_indices = ZI.astype(np.int64) * (fm.NN1 // nsub) + YI.astype(np.int64)
    fm.xl_range, fm.xl_steps = -fm.xl_start, fm.NN1
    fm.yl_range, fm.yl_steps = -fm.yl_start, fm.NN2
    fm.zl_range, fm.zl_steps = -fm.zl_start, fm.NN3
    fm.rl = np.vstack(
        np.mgrid[
            -fm.xl_range : fm.xl_range : complex(fm.xl_steps),
            -fm.yl_range : fm.yl_range : complex(fm.yl_steps),
            -fm.zl_range : fm.zl_range : complex(fm.zl_steps),
        ]
    ).reshape(3, -1)
    fm.prob_z = np.exp(-0.5 * (fm.rl[2] / fm.zl_rms) ** 2)


def main() -> None:
    from dfxm_geo.pipeline import (
        CenteredCrystalConfig,
        CrystalConfig,
        _lookup_and_load_kernel,
    )

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(os.environ.get("DFXM_RENDER_OUTPUT_DIR", repo_root / "docs" / "img"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _rescale(NPIXELS, NSUB)
    _lookup_and_load_kernel(HKL, KEV)

    # Single centered dislocation (canonical FCC primary).
    crystal = CrystalConfig(mode="centered", centered=CenteredCrystalConfig())
    fov_um = fm.Npixels * fm.psize * 1e6
    population = fm.build_dislocation_population(crystal, fov_lateral_um=fov_um, rng=None)
    Hg, q_hkl = fm.Find_Hg_from_population(population, h=HKL[0], k=HKL[1], l=HKL[2])
    fm.Hg, fm.q_hkl = Hg, q_hkl
    print(
        f"Npixels={fm.Npixels} NN1={fm.NN1} NN2={fm.NN2} NN3={fm.NN3}  ndis={len(population.positions_um)}"
    )

    phis = np.linspace(-PHI_MAX_URAD * 1e-6, PHI_MAX_URAD * 1e-6, N_FRAMES)
    imgs = [np.asarray(fm.forward(Hg, phi=float(p))).T for p in phis]  # each (NN1, NN2)
    vmax = float(np.percentile(np.stack(imgs), 99.9))  # shared scale; ignore hot pixels
    print(f"rendered {len(imgs)} frames; shared vmax={vmax:.3g}")

    # Rocking curve: integrated intensity per frame vs phi (normalized to peak).
    phis_urad = phis * 1e6
    rock = np.array([float(im_arr.sum()) for im_arr in imgs])
    rock /= rock.max()

    # Two panels: left = detector image, right = the rocking curve with a dot
    # tracking the current phi. Build once; update artists per frame so every
    # frame is the same canvas size.
    ext = [fm.yl_start * 1e6, -fm.yl_start * 1e6, fm.xl_start * 1e6, -fm.xl_start * 1e6]
    norm = PowerNorm(GAMMA, vmin=0.0, vmax=vmax)
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.6, 3.3), dpi=110, gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.18, wspace=0.30)

    im = axL.imshow(
        imgs[0],
        origin="lower",
        extent=ext,
        cmap="viridis",
        norm=norm,
        aspect="equal",
        interpolation="bilinear",
    )
    axL.set_xlabel(r"$y_\ell$ ($\mu$m)", fontsize=9)
    axL.set_ylabel(r"$x_\ell$ ($\mu$m)", fontsize=9)
    title = axL.set_title("", fontsize=11)

    axR.plot(phis_urad, rock, "-", color="0.55", lw=1.6)
    axR.set_xlim(phis_urad[0], phis_urad[-1])
    axR.set_ylim(0, 1.05)
    axR.set_xlabel(r"$\phi$ ($\mu$rad)", fontsize=9)
    axR.set_ylabel("integrated intensity (norm.)", fontsize=9)
    axR.set_title("rocking curve", fontsize=11)
    (marker,) = axR.plot([phis_urad[0]], [rock[0]], "o", color="crimson", ms=8, zorder=5)

    rgb_frames: list[np.ndarray] = []
    for k, (img, phi) in enumerate(zip(imgs, phis, strict=True)):
        im.set_data(img)
        title.set_text(rf"single dislocation   $\phi = {phi * 1e6:+.0f}$ $\mu$rad")
        marker.set_data([phis_urad[k]], [rock[k]])
        fig.canvas.draw()
        rgb_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    # Ping-pong: -150 -> +150 -> -150 so the loop is seamless.
    loop_frames = rgb_frames + rgb_frames[-2:0:-1]
    # Quantize every frame to ONE shared 128-color palette: indexed GIF is far
    # smaller than truecolor and a shared palette avoids inter-frame flicker.
    # Derive the palette from ALL frames stacked together so it spans the full
    # viridis range (dark-purple wings through bright-yellow core) — a palette
    # from a single frame misses colors the other frames need and greys them out.
    palette = Image.fromarray(np.concatenate(rgb_frames, axis=0)).quantize(
        colors=128, method=Image.Quantize.MEDIANCUT
    )
    pil = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in loop_frames
    ]
    gif_path = out_dir / "rocking_single_dislocation.gif"
    pil[0].save(
        gif_path,
        save_all=True,
        append_images=pil[1:],
        duration=int(1000 / FPS * SLOWDOWN),
        loop=0,
        optimize=True,
    )
    size_mb = gif_path.stat().st_size / 1e6
    print(f"saved {gif_path}  ({len(loop_frames)} frames, {size_mb:.2f} MB)")

    # Verification contact sheet: phi = -PHI_MAX, 0, +PHI_MAX.
    fig2, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=120)
    for ax2, k, lbl in (
        (axs[0], 0, f"{-PHI_MAX_URAD:.0f}"),
        (axs[1], N_FRAMES // 2, "0"),
        (axs[2], -1, f"+{PHI_MAX_URAD:.0f}"),
    ):
        ax2.imshow(
            imgs[k],
            origin="lower",
            extent=ext,
            cmap="viridis",
            norm=PowerNorm(GAMMA, vmin=0.0, vmax=vmax),
            aspect="equal",
            interpolation="bilinear",
        )
        ax2.set_title(rf"$\phi={lbl}$ $\mu$rad")
    fig2.tight_layout()
    fig2.savefig(out_dir / "_rocking_contact_sheet.png", bbox_inches="tight")
    plt.close(fig2)
    print(f"saved contact sheet {out_dir / '_rocking_contact_sheet.png'}")


if __name__ == "__main__":
    main()
