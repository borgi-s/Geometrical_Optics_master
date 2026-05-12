"""High-level orchestration of a DFXM forward-simulation run.

Provides a config-driven entry point (`run_simulation`) and a CLI (`cli_main`)
that wrap `Find_Hg` + `save_images_parallel`. The pipeline produces a directory
of `.npy` images covering a (phi, chi) rocking grid.

Scope note: v1 honors crystal `dis`/`ndis` and scan grid params from config.
`psize`, `zl_rms`, and the (h, k, l) reflection are bound to the module-level
defaults in `dfxm_geo.direct_space.forward_model` (ID06 settings — see
`dfxm_geo.constants`). Making those configurable requires re-deriving the
detector ray grid `rl` at runtime, which is import-time work in forward_model
today. Deferred.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.analysis.mosaicity import compute_chi_shift, compute_com_maps
from dfxm_geo.io.images import load_images, save_images_parallel
from dfxm_geo.viz.mosaicity import plot_mosaicity_maps, plot_qi_cross_section


@dataclass
class CrystalConfig:
    dis: float = 4.0  # inter-dislocation distance (µm)
    ndis: int = 151  # number of dislocations


@dataclass
class ScanConfig:
    phi_range: float  # half-range in degrees
    phi_steps: int
    chi_range: float  # half-range in degrees
    chi_steps: int


@dataclass
class IOConfig:
    fn_prefix: str = "/mosa_test_0000_"
    ftype: str = ".npy"
    dislocs_dirname: str = "images10"
    perfect_dirname: str = "images10_perf_crystal"
    include_perfect_crystal: bool = True


@dataclass
class PostprocessConfig:
    """Knobs for the post-processing stage (Phase 9.2).

    See ``docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md``.
    """

    enabled: bool = True
    chi_oversample: int = 20
    phi_oversample: int = 20
    chi_oversample_for_shift: int = 100
    figures_dirname: str = "figures"
    data_dirname: str = "analysis"


@dataclass
class SimulationConfig:
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    scan: ScanConfig = field(
        default_factory=lambda: ScanConfig(
            phi_range=0.0006 * 180 / np.pi,
            phi_steps=61,
            chi_range=0.002 * 180 / np.pi,
            chi_steps=61,
        )
    )
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    @classmethod
    def from_toml(cls, path: Path) -> SimulationConfig:
        """Load a SimulationConfig from a TOML file."""
        with path.open("rb") as f:
            raw = tomllib.load(f)
        crystal = CrystalConfig(**raw.get("crystal", {}))
        scan = ScanConfig(**raw["scan"])
        io = IOConfig(**raw.get("io", {}))
        postprocess = PostprocessConfig(**raw.get("postprocess", {}))
        return cls(crystal=crystal, scan=scan, io=io, postprocess=postprocess)


def _ensure_kernel_loaded() -> None:
    """Raise a clear error if the reciprocal-space resolution kernel is missing.

    `forward()` reads module-level `Resq_i` populated by `_load_default_kernel`
    at import time iff the default pickle exists. If it didn't, surface that
    here instead of inside the thread pool inside `save_images_parallel`.
    """
    if fm.Resq_i is None:
        raise RuntimeError(
            "Reciprocal-space resolution kernel not loaded. The pipeline "
            "requires the default kernel pickle to be present at "
            f"{fm.pkl_fpath + fm.pkl_fn!r}. Place the pickle there or call "
            "dfxm_geo.direct_space.forward_model._load_default_kernel(...) "
            "with explicit paths before running the pipeline."
        )


def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    """Execute a DFXM forward-simulation run from a config object.

    Computes Hg via `Find_Hg(dis, ndis, psize, zl_rms)`, then sweeps the
    (phi, chi) grid and writes one .npy per (phi_step, chi_step) into
    `output_dir/<io.dislocs_dirname>/`. If `io.include_perfect_crystal` is
    true, also runs a parallel sweep with Hg=0 into
    `output_dir/<io.perfect_dirname>/`.

    Returns a dict with the resolved output paths and the computed Hg/q_hkl.
    """
    _ensure_kernel_loaded()
    output_dir.mkdir(parents=True, exist_ok=True)

    Hg, q_hkl = fm.Find_Hg(config.crystal.dis, config.crystal.ndis, fm.psize, fm.zl_rms)
    # Forward() reads the module-level q_hkl global. Find_Hg() returns a new
    # one based on its h/k/l args but does not update the global itself. Sync
    # them so the rocking sweep uses the same reflection that Hg was built for.
    fm.Hg = Hg
    fm.q_hkl = q_hkl

    dislocs_path = output_dir / config.io.dislocs_dirname
    save_images_parallel(
        Hg,
        config.scan.phi_range,
        config.scan.phi_steps,
        config.scan.chi_range,
        config.scan.chi_steps,
        str(dislocs_path),
        config.io.fn_prefix,
        config.io.ftype,
    )

    perfect_path: Path | None = None
    if config.io.include_perfect_crystal:
        perfect_path = output_dir / config.io.perfect_dirname
        save_images_parallel(
            np.zeros_like(Hg),
            config.scan.phi_range,
            config.scan.phi_steps,
            config.scan.chi_range,
            config.scan.chi_steps,
            str(perfect_path),
            config.io.fn_prefix,
            config.io.ftype,
        )

    return {
        "dislocs_path": dislocs_path,
        "perfect_path": perfect_path,
        "Hg": Hg,
        "q_hkl": q_hkl,
    }


def run_postprocess(output_dir: Path, config: SimulationConfig) -> dict[str, Any]:
    """Stages 2-4 of init_forward.py against an existing output_dir.

    Reads the perfect and dislocated stacks from disk, computes the χ-shift
    correction, computes per-pixel COM maps, calls forward() for the qi field
    at z=0, then writes data products and SVGs under output_dir.

    Raises:
        FileNotFoundError: if either expected stack directory is absent.
        RuntimeError: from :func:`_ensure_kernel_loaded` if the reciprocal-
            space kernel is missing.
    """
    _ensure_kernel_loaded()

    dislocs_path = output_dir / config.io.dislocs_dirname
    perfect_path = output_dir / config.io.perfect_dirname

    if not dislocs_path.is_dir():
        raise FileNotFoundError(
            f"Expected dislocs stack at {dislocs_path}; run dfxm-forward "
            "without --postprocess-only first."
        )
    if not perfect_path.is_dir():
        raise FileNotFoundError(
            f"Expected perfect-crystal stack at {perfect_path}; run "
            "dfxm-forward without --postprocess-only first."
        )

    _, dis_reshape, _, _ = load_images(
        str(dislocs_path),
        config.scan.phi_steps,
        config.scan.chi_steps,
        file_ext=config.io.ftype,
    )
    _, perf_reshape, _, _ = load_images(
        str(perfect_path),
        config.scan.phi_steps,
        config.scan.chi_steps,
        file_ext=config.io.ftype,
    )

    # Stage 2: χ-shift
    chi_shift = compute_chi_shift(
        perf_reshape,
        config.scan.chi_steps,
        config.scan.chi_range,
        oversample=config.postprocess.chi_oversample_for_shift,
    )

    # Stage 3: per-pixel COM maps. The original script uses the same oversample
    # factor for both axes; we keep them independent in the API but they
    # default to the same value.
    phi_list, chi_list = compute_com_maps(
        dis_reshape,
        config.scan.phi_range,
        config.scan.phi_steps,
        config.scan.chi_range,
        config.scan.chi_steps,
        chi_shift=chi_shift,
        oversample=config.postprocess.phi_oversample,
    )

    # Stage 4: qi field at z=0 via a single forward() call. Uses module-level
    # Hg as left by run_simulation (or the default load).
    if fm.Hg is None:
        raise RuntimeError(
            "fm.Hg is not set. Call run_simulation() first or assign fm.Hg "
            "before calling run_postprocess()."
        )
    _, qi_field = fm.forward(fm.Hg, phi=0, qi_return=True)

    # Persist data products.
    data_dir = output_dir / config.postprocess.data_dirname
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "phi_list.npy", phi_list)
    np.save(data_dir / "chi_list.npy", chi_list)
    np.save(data_dir / "qi_field.npy", qi_field)
    (data_dir / "chi_shift_deg.txt").write_text(f"{chi_shift}\n")

    # Render figures.
    fig_dir = output_dir / config.postprocess.figures_dirname
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_mosaicity_maps(
        phi_list,
        chi_list,
        fm.xl_start,
        fm.yl_start,
        fig_dir / "mosaicity_maps.svg",
    )
    plot_qi_cross_section(
        qi_field,
        fm.xl_start,
        fm.yl_start,
        fm.xl_steps,
        fm.yl_steps,
        fm.zl_steps,
        fig_dir / "qi_cross_section.svg",
    )

    return {
        "phi_list": phi_list,
        "chi_list": chi_list,
        "qi_field": qi_field,
        "chi_shift": chi_shift,
        "data_dir": data_dir,
        "figures_dir": fig_dir,
    }


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for ``dfxm-forward`` and ``python scripts/run_forward.py``.

    Default behavior: run simulation, then post-processing.
    """
    parser = argparse.ArgumentParser(description="DFXM forward simulation")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Run simulation only; skip post-processing (Phase 6 behavior).",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip simulation; run post-processing against an existing output dir.",
    )
    args = parser.parse_args(argv)

    config = SimulationConfig.from_toml(args.config)

    if args.postprocess_only:
        run_postprocess(args.output, config)
    else:
        run_simulation(config, args.output)
        if config.postprocess.enabled and not args.no_postprocess:
            run_postprocess(args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
