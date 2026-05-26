"""Regenerate the example images linked from the top of README.md.

Outputs (default destination is ``docs/img/``; override with the
``DFXM_RENDER_OUTPUT_DIR`` environment variable):

- ``example_dislocs_frame.png`` -- single (phi=0, chi=0) frame from
  the dislocs stack at a scaled-down rocking grid.
- ``example_mosaicity.png`` -- phi-COM mosaicity map from the
  post-processing stage.

This script is *not* part of CI:

- it runs `dfxm-forward` end-to-end (~10-30 s wall-clock at the small
  variant, but loads the 128 MB kernel pickle), and
- the rendered floats are sensitive to hardware/BLAS, so we can't pin
  them in version control without churn.

Run manually after a substantive change to `dfxm_geo.viz` or
`dfxm_geo.pipeline.run_postprocess`:

    python scripts/render_readme_examples.py --small

The committed PNGs in ``docs/img/`` are the canonical version -- only
overwrite them when you intend to publish the new look.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _build_small_config(tmp_dir: Path) -> Path:
    """Write a scaled-down TOML for fast example rendering."""
    cfg = tmp_dir / "render_small.toml"
    cfg.write_text(
        "[crystal]\n"
        'mode = "wall"\n'
        "\n"
        "[crystal.wall]\n"
        "dis = 4\n"
        "ndis = 151\n"
        'sample_remount = "S1"\n'
        "\n"
        # Article ranges (Borgi 2024), radians: phi ±600 µrad, chi ±2 mrad.
        "[scan.phi]\n"
        "range = 6e-4\n"
        "steps = 11\n"
        "\n"
        "[scan.chi]\n"
        "range = 2e-3\n"
        "steps = 11\n"
        "\n"
        "[io]\n"
        'fn_prefix = "/mosa_test_0000_"\n'
        'ftype = ".npy"\n'
        'dislocs_dirname = "images10"\n'
        'perfect_dirname = "images10_perf_crystal"\n'
        "include_perfect_crystal = true\n"
        "\n"
        "[postprocess]\n"
        "enabled = true\n"
        "chi_oversample = 5\n"
        "phi_oversample = 5\n"
        "chi_oversample_for_shift = 20\n"
        'figures_dirname = "figures"\n'
        'data_dirname = "analysis"\n',
        encoding="utf-8",
    )
    return cfg


def _save_dislocs_frame_png(h5_path: Path, out_png: Path) -> None:
    """Save the center frame of the dislocs detector stack as a PNG.

    Reads the BLISS-layout detector dataset from the v1.2+ HDF5 output
    (``/1.1/instrument/dfxm_sim_detector/data``); the legacy per-frame
    ``.npy`` layout was retired in v1.1.0.
    """
    import h5py
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Pick the center frame of the rocking grid.
    with h5py.File(h5_path, "r") as f:
        data = f["/1.1/instrument/dfxm_sim_detector/data"]
        arr = data[data.shape[0] // 2, :, :]
    fig, ax = plt.subplots(figsize=(4, 4), dpi=144)
    im = ax.imshow(arr.T, origin="lower", cmap="viridis")
    ax.set_title("DFXM forward image (center of rocking grid)")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate README example images")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a scaled-down config (11x11 rocking, ~30 s wall clock).",
    )
    parser.add_argument(
        "--backend",
        choices=("mc", "analytic"),
        default="mc",
        help="Resolution backend. 'mc' (default) uses the beamstop Monte-Carlo "
        "kernel and reproduces the published images. 'analytic' uses the v2.1.0 "
        "closed-form backend, which forces the beamstop OFF -- a physically "
        "different (no-beamstop) image.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    default_out_dir = repo_root / "docs" / "img"
    out_dir = Path(os.environ.get("DFXM_RENDER_OUTPUT_DIR", default_out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.small:
        print(
            "warning: running at full resolution -- this will take minutes. "
            "Pass --small for the quick variant.",
            file=sys.stderr,
        )

    # Import here so that --help works on a clean checkout that lacks the kernel.
    from dfxm_geo.data import configs_root
    from dfxm_geo.pipeline import SimulationConfig, run_postprocess, run_simulation

    with tempfile.TemporaryDirectory(prefix="dfxm_render_") as td:
        tmp = Path(td)
        cfg_path = _build_small_config(tmp) if args.small else configs_root() / "default.toml"
        cfg = SimulationConfig.from_toml(cfg_path)
        if args.backend == "analytic":
            # The closed-form backend exists only for the no-beamstop regime, so
            # selecting it forces the beamstop off (run_simulation would raise
            # otherwise). Note this changes the physics vs the published images.
            cfg.reciprocal.backend = "analytic"
            cfg.reciprocal.beamstop = False
        run_dir = tmp / "run"
        run_simulation(cfg, run_dir)
        res = run_postprocess(run_dir, cfg)

        # 1. A dislocs-stack frame (center of the rocking grid, from the HDF5).
        _save_dislocs_frame_png(
            res["h5_path"],
            out_dir / "example_dislocs_frame.png",
        )

        # 2. The README needs PNG, not SVG, for portable embedding. Re-render the
        # mosaicity figure as PNG from the COM maps returned by run_postprocess.
        phi_list = res["phi_list"]
        chi_list = res["chi_list"]
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, (axp, axc) = plt.subplots(1, 2, figsize=(8, 4), dpi=144)
        # Fixed ±1e-4 rad color range — the same scale used for the qi/COM maps
        # in Borgi et al. (2024), where the COM↔qi correspondence reads as an
        # excellent match. Autoscaling per-panel would exaggerate small residuals.
        for ax, data, title in (
            (axp, phi_list, "phi COM (mosaicity)"),
            (axc, chi_list, "chi COM (mosaicity)"),
        ):
            im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-1e-4, vmax=1e-4)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "example_mosaicity.png", bbox_inches="tight")
        plt.close(fig)

        # 3. (Optional) the coordinate-frame diagram if a static asset exists.
        coord_src = repo_root / "docs" / "img" / "_static" / "coordinate_frames.png"
        if coord_src.is_file():
            shutil.copy(coord_src, out_dir / "example_coordinate_frames.png")

    print(f"wrote example images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
