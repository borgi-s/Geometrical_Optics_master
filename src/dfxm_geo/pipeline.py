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
import csv
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.analysis.mosaicity import compute_chi_shift, compute_com_maps
from dfxm_geo.crystal.burgers import burgers_vectors as _burgers_vectors
from dfxm_geo.crystal.burgers import (
    rotated_t_vectors as _rotated_t_vectors,
)
from dfxm_geo.crystal.burgers import (
    ud_matrices as _ud_matrices,
)
from dfxm_geo.crystal.dislocations import (
    Fd_find_mixed,
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
)
from dfxm_geo.crystal.rotations import fast_inverse2
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
    max_workers: int | None = None


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


@dataclass(frozen=True, kw_only=True)
class IdentificationCrystalConfig:
    """Crystal config for `dfxm-identify`. Slip plane + Burgers vector sweep."""

    slip_plane_normal: tuple[int, int, int]
    angle_start_deg: float = 0.0
    angle_stop_deg: float = 350.0
    angle_step_deg: float = 10.0
    b_vector_indices: list[int] | None = None  # None = all 6
    sweep_all_slip_planes: bool = True
    exclude_invisibility: bool = True
    invisibility_threshold_deg: float = 10.0


@dataclass(frozen=True, kw_only=True)
class IdentificationScanConfig:
    """Forward-model scan parameters for `dfxm-identify`."""

    phi_rad: float = 150e-6
    poisson_noise: bool = True
    rng_seed: int = 0
    intensity_scale: float = 7.0


@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    """Multi-disloc Monte Carlo parameters (mode='multi' only)."""

    n_samples: int = 1000
    pos_std_um: float = 5.0
    n_png_previews: int = 50


@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    """z-scan mode parameters (mode='z-scan' only).

    Each (z_layer, b, α) configuration produces a (phi_steps × chi_steps)
    rocking-curve stack on disk, with a randomly-drawn secondary
    dislocation if `include_secondary` is True. The secondary is drawn
    once per (z, b, α) and shared across the rocking grid.
    """

    z_offsets_um: list[float]
    phi_range_deg: float
    phi_steps: int
    chi_range_deg: float
    chi_steps: int
    include_secondary: bool = True
    secondary_rng_offset: int = 1


@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    """Top-level config for dfxm-identify.

    Validates mode / sub-config / slip-plane consistency in __post_init__.
    """

    mode: Literal["single", "multi", "z-scan"]
    crystal: IdentificationCrystalConfig
    scan: IdentificationScanConfig
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("single", "multi", "z-scan"):
            raise ValueError(f"mode must be 'single', 'multi', or 'z-scan', got {self.mode!r}")
        if self.mode == "multi" and self.multi is None:
            raise ValueError("mode='multi' requires a `multi` config block")
        if self.mode == "z-scan" and self.zscan is None:
            raise ValueError("mode='z-scan' requires a `zscan` config block")
        if self.mode in ("single", "multi") and self.zscan is not None:
            raise ValueError(
                f"mode={self.mode!r}: zscan config block is only valid in mode='z-scan'"
            )
        # Validate the slip plane against the {111} family (also used in 'multi'
        # mode as the starting / fallback plane).
        try:
            _burgers_vectors(self.crystal.slip_plane_normal)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


def load_identification_config(path: Path) -> IdentificationConfig:
    """Load and validate an `dfxm-identify` config from a TOML file.

    Args:
        path: Filesystem path to a TOML config (see configs/identification_*.toml).

    Returns:
        Validated IdentificationConfig.

    Raises:
        ValueError: if the TOML is missing the top-level `mode` field, or if
            the validation in IdentificationConfig.__post_init__ rejects the
            content.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    if "mode" not in data:
        raise ValueError(f"{path}: missing top-level 'mode' field")

    crystal_data = data.get("crystal", {})
    if "slip_plane_normal" in crystal_data:
        crystal_data = {
            **crystal_data,
            "slip_plane_normal": tuple(crystal_data["slip_plane_normal"]),
        }
    crystal = IdentificationCrystalConfig(**crystal_data)
    scan = IdentificationScanConfig(**data.get("scan", {}))
    io = IOConfig(**data.get("io", {}))
    multi = (
        IdentificationMonteCarloConfig(**data["multi"]) if data.get("multi") is not None else None
    )

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        io=io,
        multi=multi,
    )


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
        max_workers=config.io.max_workers,
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
            max_workers=config.io.max_workers,
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

    Warning:
        When invoked via ``--postprocess-only`` against an output dir whose
        stacks were produced with non-default ``dis`` / ``ndis``, the qi field
        is computed against the *module-level* ``fm.Hg``. If no prior
        ``run_simulation`` set ``fm.Hg`` in this process, it will fall back to
        the default kernel auto-load — which may not match the saved stacks.
        For correctness in that workflow, assign ``fm.Hg`` explicitly before
        calling this function.

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
        chi_oversample=config.postprocess.chi_oversample,
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


def _slip_plane_slug(n: tuple[int, int, int]) -> str:
    """Convert (1, -1, 1) -> '1_m1_1' for directory names."""
    return "_".join("m" + str(abs(c)) if c < 0 else str(c) for c in n)


def _save_preview_png(arr: np.ndarray, png_path: Path) -> None:
    """Quick matplotlib heatmap snapshot for spot-checking."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(arr.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(png_path, dpi=72, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _passes_invisibility(
    q_hkl: np.ndarray,
    b_vec: np.ndarray,
    threshold_deg: float,
) -> bool:
    """True if |G·b| / (|G| |b|) >= cos(90° - threshold) — NOT near-orthogonal.

    Paper convention (Borgi 2025): a configuration is excluded if the Burgers
    vector is within `threshold_deg` degrees of perpendicular to G.
    cos(90° - 10°) = cos(80°) ≈ 0.174.
    """
    cos_angle = abs(np.dot(q_hkl, b_vec)) / (np.linalg.norm(q_hkl) * np.linalg.norm(b_vec))
    return bool(cos_angle >= np.cos(np.deg2rad(90.0 - threshold_deg)))


def _run_identification_single(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Deterministic Cartesian sweep: slip planes × Burgers vectors × angles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crystal_cfg = config.crystal
    scan_cfg = config.scan

    all_planes: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, 1, 1),
    ]
    planes = all_planes if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]

    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    rng = np.random.default_rng(scan_cfg.rng_seed) if scan_cfg.poisson_noise else None
    q_hkl = np.asarray(fm.q_hkl, dtype=float)

    manifest_rows: list[dict[str, Any]] = []
    n_written = 0

    for plane in planes:
        plane_slug = _slip_plane_slug(plane)
        im_dir = output_dir / f"n_{plane_slug}" / "im_data"
        png_dir = output_dir / f"n_{plane_slug}" / "images"
        im_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        b_table = _burgers_vectors(plane)
        b_indices = (
            crystal_cfg.b_vector_indices
            if crystal_cfg.b_vector_indices is not None
            else list(range(len(b_table)))
        )
        b_subset = b_table[b_indices]
        n_arr_unnorm = np.asarray(plane, dtype=float)
        n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)

        rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
        Ud_all = _ud_matrices(n_arr, rotated)  # (n_angles, n_b, 3, 3)

        for j, b_idx in enumerate(b_indices):
            if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
            ):
                continue
            for i, alpha in enumerate(angles_deg):
                Ud_mix = Ud_all[i, j]
                Fg = Fd_find_mixed(
                    fm.rl,
                    fm.Us,
                    Ud_mix=Ud_mix,
                    rotation_deg=float(alpha),
                    Theta=fm.Theta,
                )
                Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                fm.Hg = Hg
                fm.q_hkl = q_hkl

                image_arr = fm.forward(Hg, phi=scan_cfg.phi_rad)
                # forward() can return either an ndarray or a (image, qi) tuple
                # depending on qi_return. We don't pass qi_return so it's always
                # an ndarray here; assert for the type-checker.
                assert isinstance(image_arr, np.ndarray)
                image = image_arr * scan_cfg.intensity_scale
                if scan_cfg.poisson_noise:
                    assert rng is not None
                    image = rng.poisson(np.clip(image, a_min=0.0, a_max=None)).astype(float)

                stem = f"b{b_idx}_alpha{int(round(alpha)):03d}"
                npy_path = im_dir / f"{stem}.npy"
                png_path = png_dir / f"{stem}.png"
                np.save(npy_path, image)
                _save_preview_png(image, png_path)

                manifest_rows.append(
                    {
                        "image_path": str(npy_path.relative_to(output_dir)),
                        "n_h": plane[0],
                        "n_k": plane[1],
                        "n_l": plane[2],
                        "b_idx": b_idx,
                        "b_h": int(round(b_table[b_idx, 0] * np.sqrt(2))),
                        "b_k": int(round(b_table[b_idx, 1] * np.sqrt(2))),
                        "b_l": int(round(b_table[b_idx, 2] * np.sqrt(2))),
                        "rotation_deg": float(alpha),
                    }
                )
                n_written += 1

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        if manifest_rows:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        else:
            fh.write("")

    return {"n_images": n_written, "output_dir": output_dir, "manifest_path": manifest_path}


_ALL_111_PLANES: list[tuple[int, int, int]] = [
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, 1),
]


def _draw_dislocation(rng: np.random.Generator, pos_std_um: float) -> dict[str, Any]:
    """Draw a single random dislocation (slip plane, Burgers idx, angle, position)."""
    plane_idx = int(rng.integers(0, len(_ALL_111_PLANES)))
    plane = _ALL_111_PLANES[plane_idx]
    b_table = _burgers_vectors(plane)
    b_idx = int(rng.integers(0, len(b_table)))
    alpha = float(rng.uniform(0.0, 360.0))
    pos = (float(rng.normal(0.0, pos_std_um)), float(rng.normal(0.0, pos_std_um)), 0.0)

    n = np.asarray(plane, dtype=float)
    n_unit = n / np.linalg.norm(n)
    rotated = _rotated_t_vectors(n_unit, b_table[b_idx : b_idx + 1], np.array([alpha]))
    Ud = _ud_matrices(n_unit, rotated)[0, 0]

    return {
        "plane": plane,
        "b_idx": b_idx,
        "b_vec": b_table[b_idx] * np.sqrt(2),
        "alpha_deg": alpha,
        "pos_um": pos,
        "Ud": Ud,
    }


def _run_identification_multi(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Monte Carlo over n_samples; each sample is 2 random mixed dislocations summed."""
    assert config.multi is not None  # validated in __post_init__
    mc = config.multi
    scan_cfg = config.scan

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "im_data").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Split master rng → child streams (param draws, Poisson noise).
    master = np.random.default_rng(scan_cfg.rng_seed)
    param_rng, noise_rng = master.spawn(2)

    manifest_rows: list[dict[str, Any]] = []
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    fm.q_hkl = q_hkl

    pad = max(5, len(str(mc.n_samples - 1)))
    for k in range(mc.n_samples):
        d1 = _draw_dislocation(param_rng, mc.pos_std_um)
        d2 = _draw_dislocation(param_rng, mc.pos_std_um)
        specs = [
            MixedDislocSpec(
                Ud_mix=d1["Ud"], rotation_deg=d1["alpha_deg"], position_lab_um=d1["pos_um"]
            ),
            MixedDislocSpec(
                Ud_mix=d2["Ud"], rotation_deg=d2["alpha_deg"], position_lab_um=d2["pos_um"]
            ),
        ]
        Fg = Fd_find_multi_dislocs_mixed(fm.rl, fm.Us, specs, fm.Theta)
        Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
        fm.Hg = Hg

        image_arr = fm.forward(Hg, phi=scan_cfg.phi_rad)
        assert isinstance(image_arr, np.ndarray)
        image = image_arr * scan_cfg.intensity_scale
        if scan_cfg.poisson_noise:
            image = noise_rng.poisson(np.clip(image, a_min=0.0, a_max=None)).astype(float)

        stem = f"{k:0{pad}d}"
        np.save(output_dir / "im_data" / f"{stem}.npy", image)
        if k < mc.n_png_previews:
            _save_preview_png(image, output_dir / "images" / f"{stem}.png")

        manifest_rows.append(
            {
                "sample_id": stem,
                "n1_h": d1["plane"][0],
                "n1_k": d1["plane"][1],
                "n1_l": d1["plane"][2],
                "b1_idx": d1["b_idx"],
                "b1_h": int(round(d1["b_vec"][0])),
                "b1_k": int(round(d1["b_vec"][1])),
                "b1_l": int(round(d1["b_vec"][2])),
                "alpha1_deg": d1["alpha_deg"],
                "x1_um": d1["pos_um"][0],
                "y1_um": d1["pos_um"][1],
                "n2_h": d2["plane"][0],
                "n2_k": d2["plane"][1],
                "n2_l": d2["plane"][2],
                "b2_idx": d2["b_idx"],
                "b2_h": int(round(d2["b_vec"][0])),
                "b2_k": int(round(d2["b_vec"][1])),
                "b2_l": int(round(d2["b_vec"][2])),
                "alpha2_deg": d2["alpha_deg"],
                "x2_um": d2["pos_um"][0],
                "y2_um": d2["pos_um"][1],
                "image_path": f"im_data/{stem}.npy",
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        if manifest_rows:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    return {"n_samples": mc.n_samples, "output_dir": output_dir, "manifest_path": manifest_path}


def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single or multi runner based on config.mode."""
    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    return _run_identification_multi(config, output_dir)


def cli_main_identify(argv: list[str] | None = None) -> int:
    """Argparse-driven entry point for `dfxm-identify`."""
    parser = argparse.ArgumentParser(description="DFXM dislocation identification simulation")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to identification TOML config"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default=None,
        help="Override the config's mode field",
    )
    args = parser.parse_args(argv)

    cfg = load_identification_config(args.config)
    if args.mode is not None and args.mode != cfg.mode:
        from dataclasses import replace

        cfg = replace(cfg, mode=args.mode)
        cfg.__post_init__()  # re-run validation

    result = run_identification(cfg, args.output)
    if cfg.mode == "single":
        print(f"Wrote {result['n_images']} images to {result['output_dir']}")
    else:
        print(f"Wrote {result['n_samples']} samples to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
