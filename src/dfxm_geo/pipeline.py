"""High-level orchestration of a DFXM forward-simulation run.

Provides a config-driven entry point (`run_simulation`) and a CLI (`cli_main`)
that wrap `Find_Hg` + `write_simulation_h5`. The pipeline produces a master
HDF5 file plus per-scan detector files covering a (phi, chi) rocking grid.

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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
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
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.io.hdf5 import (
    DETECTOR_FILE_FMT,
    DETECTOR_INTERNAL_PATH,
    SCAN_DIR_FMT,
    ScanSpec,
    load_h5_scan,
    write_identification_h5,
    write_simulation_h5,
)
from dfxm_geo.viz.mosaicity import plot_mosaicity_maps, plot_qi_cross_section


@dataclass
class AxisScanConfig:
    """Per-motor-axis scan primitive (sub-project B).

    Each motor axis (phi, chi, two_dtheta, z) is independently fixed
    at `value` or scanned over `[value-range, value+range]` with `steps`
    samples (linspace). Both `range` and `steps` must be present
    together for the axis to be scanned, or both absent for fixed.
    """

    value: float = 0.0
    range: float | None = None
    steps: int | None = None

    def __post_init__(self) -> None:
        if (self.range is None) != (self.steps is None):
            raise ValueError(
                "AxisScanConfig must specify both `range` and `steps`, or neither "
                f"(fixed at `value`). Got range={self.range!r}, steps={self.steps!r}"
            )
        if self.range is not None:
            if self.range <= 0:
                raise ValueError(f"`range` must be > 0; got {self.range!r}")
            assert self.steps is not None  # XOR guard above guarantees this
            if self.steps < 2:
                raise ValueError(f"`steps` must be >= 2 when range is set; got {self.steps!r}")

    @property
    def is_scanned(self) -> bool:
        return self.range is not None and self.steps is not None


_CANONICAL_AXES = ("phi", "chi", "two_dtheta", "z")

_AXIS_TO_LABEL = {
    "phi": "rocking",
    "chi": "rolling",
    "two_dtheta": "strain",
    "z": "layer",
}

_PRE_CANONIZED_MODE_NAMES: dict[frozenset[str], str] = {
    frozenset(): "single",
    frozenset({"phi", "chi"}): "mosa",
    frozenset({"phi", "chi", "two_dtheta"}): "mosa_strain",
    frozenset({"phi", "chi", "z"}): "mosa_layer",
    frozenset({"phi", "chi", "two_dtheta", "z"}): "mosa_strain_layer",
}


@dataclass
class ScanConfig:
    """Per-axis scan primitives (sub-project B).

    Each motor axis is independently fixed or scanned. The "scan mode"
    label is derived from which axes carry range+steps — see
    `derived_mode_name` (Task 3).
    """

    phi: AxisScanConfig = field(default_factory=AxisScanConfig)
    chi: AxisScanConfig = field(default_factory=AxisScanConfig)
    two_dtheta: AxisScanConfig = field(default_factory=AxisScanConfig)
    z: AxisScanConfig = field(default_factory=AxisScanConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> ScanConfig:
        if not data:
            return cls()
        unknown = set(data.keys()) - set(_CANONICAL_AXES)
        if unknown:
            raise ValueError(
                f"unknown scan axis {sorted(unknown)[0]!r}; expected one of {_CANONICAL_AXES}"
            )
        kwargs = {axis: AxisScanConfig(**data[axis]) for axis in _CANONICAL_AXES if axis in data}
        return cls(**kwargs)

    def scanned_axes(self) -> tuple[str, ...]:
        """Names of motor axes that carry a range+steps (in canonical order)."""
        return tuple(a for a in _CANONICAL_AXES if getattr(self, a).is_scanned)

    def is_scanned(self, axis: str) -> bool:
        if axis not in _CANONICAL_AXES:
            raise ValueError(f"unknown axis {axis!r}; expected one of {_CANONICAL_AXES}")
        return getattr(self, axis).is_scanned

    def derived_mode_name(self) -> str:
        """Derive the scan-mode label from which axes are scanned.

        Pre-canonized: single, rocking, rolling, strain, layer, mosa,
        mosa_strain, mosa_layer, mosa_strain_layer. All other combos
        are the 1D labels concatenated in canonical axis order.
        """
        scanned = self.scanned_axes()
        key = frozenset(scanned)
        if key in _PRE_CANONIZED_MODE_NAMES:
            return _PRE_CANONIZED_MODE_NAMES[key]
        if len(scanned) == 1:
            return _AXIS_TO_LABEL[scanned[0]]
        return "_".join(_AXIS_TO_LABEL[a] for a in scanned)


@dataclass
class CenteredCrystalConfig:
    """Single dislocation at the origin (sub-project C, mode='centered').

    The Ud rotation matrix is built from (b, n, t):
      - b = Burgers vector indices
      - n = slip-plane normal indices
      - t = dislocation line direction indices

    Geometric constraints (validated):
      - b · n = 0   (Burgers vector lies in slip plane)
      - t parallel to (n × b)  (line direction consistent with slip system)
    """

    b: tuple[int, int, int]
    n: tuple[int, int, int]
    t: tuple[int, int, int]

    def __post_init__(self) -> None:
        b = self.b
        n = self.n
        t = self.t
        # b · n == 0 (exact, since these are integer crystallographic indices)
        if b[0] * n[0] + b[1] * n[1] + b[2] * n[2] != 0:
            raise ValueError(
                f"Burgers vector b={b} must be perpendicular to slip plane normal n={n} "
                "(integer dot product must be 0)"
            )
        # t parallel to (n × b) — both vectors in integer indices; parallel ⇔ cross == 0
        nxb = (
            n[1] * b[2] - n[2] * b[1],
            n[2] * b[0] - n[0] * b[2],
            n[0] * b[1] - n[1] * b[0],
        )
        # Cross product of t and nxb should be zero if they are parallel.
        cross = (
            t[1] * nxb[2] - t[2] * nxb[1],
            t[2] * nxb[0] - t[0] * nxb[2],
            t[0] * nxb[1] - t[1] * nxb[0],
        )
        if cross != (0, 0, 0):
            raise ValueError(
                f"line direction t={t} must be parallel to (n x b)={nxb} for the "
                "slip system to be self-consistent (cross product must be zero)"
            )


@dataclass
class WallCrystalConfig:
    """Dis-spaced grid of dislocations (sub-project C, mode='wall').

    The current Borgi/Purdue IUCrJ 2024 layout. Preserved unchanged from
    the legacy flat `CrystalConfig`.
    """

    dis: float = 4.0
    ndis: int = 151
    sample_remount: str = "S1"

    def __post_init__(self) -> None:
        if self.sample_remount not in SAMPLE_REMOUNT_OPTIONS:
            valid = ", ".join(SAMPLE_REMOUNT_OPTIONS.keys())
            raise ValueError(
                f"sample_remount must be one of: {valid} (got {self.sample_remount!r})"
            )


@dataclass
class RandomDislocationsConfig:
    """N random dislocations placed by 2D Gaussian (sub-project C).

    `sigma=None` → resolved at draw time from the FOV
    (sigma = FOV_lateral_half / 2).
    `min_distance=None` → no inter-dislocation distance constraint.
    `seed=None` → fresh entropy-pool seed drawn at run time; the realized
    seed value is logged into the sidecar for reproducibility.
    """

    ndis: int
    sigma: float | None = None
    min_distance: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.ndis < 1:
            raise ValueError(f"`ndis` must be >= 1 for random_dislocations; got {self.ndis}")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError(f"`sigma` must be > 0 when set; got {self.sigma}")
        if self.min_distance is not None and self.min_distance < 0:
            raise ValueError(f"`min_distance` must be >= 0 when set; got {self.min_distance}")


_CRYSTAL_MODE_NAMES = ("centered", "wall", "random_dislocations")


@dataclass
class CrystalConfig:
    """Discriminated union over the three crystal-layout modes (sub-project C).

    Exactly one of `centered`/`wall`/`random_dislocations` is non-None and
    matches `mode`. Constructed via `CrystalConfig.from_dict` from a TOML
    `[crystal]` table.
    """

    mode: Literal["centered", "wall", "random_dislocations"]
    centered: CenteredCrystalConfig | None = None
    wall: WallCrystalConfig | None = None
    random_dislocations: RandomDislocationsConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in _CRYSTAL_MODE_NAMES:
            raise ValueError(
                f"unknown crystal mode {self.mode!r}; expected one of {_CRYSTAL_MODE_NAMES}"
            )
        # Single pass: collect extras, then check required sub-block.
        # (from_dict catches both for TOML callers; this is defense-in-depth
        # for programmatic CrystalConfig(...) construction.)
        extras = sorted(
            m for m in _CRYSTAL_MODE_NAMES if m != self.mode and getattr(self, m) is not None
        )
        if extras:
            raise ValueError(
                f"crystal mode={self.mode!r}: extra sub-block {extras} "
                f"present; only [crystal.{self.mode}] is valid"
            )
        if getattr(self, self.mode) is None:
            raise ValueError(
                f"crystal mode={self.mode!r}: [crystal.{self.mode}] sub-block is required"
            )

    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig:
        if data is None:
            raise ValueError(
                "missing [crystal] block — forward/identify require explicit "
                "crystal layout; see configs/default.toml."
            )
        if "mode" not in data:
            raise ValueError("missing `mode` in [crystal] — required to pick a layout.")
        mode = data["mode"]
        if mode not in _CRYSTAL_MODE_NAMES:
            raise ValueError(
                f"unknown crystal mode {mode!r}; expected one of {_CRYSTAL_MODE_NAMES}"
            )

        # Reject sibling sub-blocks early for a precise error.
        siblings = sorted(m for m in _CRYSTAL_MODE_NAMES if m != mode and m in data)
        if siblings:
            raise ValueError(
                f"crystal mode={mode!r}: extra sub-block {siblings} present; "
                f"only [crystal.{mode}] is valid"
            )
        if mode not in data:
            raise ValueError(f"crystal mode={mode!r}: [crystal.{mode}] sub-block is required")

        sub_data = data[mode]
        kwargs: dict = {"mode": mode}
        if mode == "centered":
            kwargs["centered"] = CenteredCrystalConfig(
                b=tuple(sub_data["b"]),
                n=tuple(sub_data["n"]),
                t=tuple(sub_data["t"]),
            )
        elif mode == "wall":
            kwargs["wall"] = WallCrystalConfig(**sub_data)
        elif mode == "random_dislocations":
            kwargs["random_dislocations"] = RandomDislocationsConfig(**sub_data)
        return cls(**kwargs)


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
class ReciprocalConfig:
    """Sub-project D: reflection identity for kernel lookup.

    The TOML ``[reciprocal]`` block carries both this (small, consumed by
    forward + identify) and bootstrap's MC params (large, consumed only by
    `dfxm-bootstrap`). This dataclass holds only the lookup-relevant keys.
    """

    hkl: tuple[int, int, int]
    keV: float

    @classmethod
    def from_dict(cls, data: dict | None) -> ReciprocalConfig:
        if data is None:
            raise ValueError(
                "missing [reciprocal] block — forward/identify require explicit "
                "hkl + keV; see configs/default.toml."
            )
        if "hkl" not in data:
            raise ValueError("missing `hkl` in [reciprocal] — required for kernel lookup.")
        if "keV" not in data:
            raise ValueError("missing `keV` in [reciprocal] — required for kernel lookup.")
        hkl = tuple(data["hkl"])
        keV = float(data["keV"])
        # Early validation per spec — catches typos / Bragg-unsatisfiable
        # before the kernel lookup. Propagates A's ValueErrors verbatim.
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # TODO(non-Al materials): hardcoded Al lattice parameter; revisit if/when
        # the codebase supports other crystals. Tracked as deferred work in the
        # sub-project A spec ("materials other than Al") and in the sub-project D
        # spec ("out of scope").
        _validate_reflection(hkl, keV, 4.0495e-10)
        return cls(hkl=hkl, keV=keV)


@dataclass
class SimulationConfig:
    crystal: CrystalConfig  # NO DEFAULT — required; construct via CrystalConfig.from_dict
    scan: ScanConfig = field(default_factory=ScanConfig)
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Sub-project D: optional in Python construction (defaults to None for
    # back-compat with test fixtures). `from_toml` requires it; `run_simulation`
    # raises if None at runtime.
    reciprocal: ReciprocalConfig | None = None

    @classmethod
    def from_toml(cls, path: Path) -> SimulationConfig:
        """Load a SimulationConfig from a TOML file."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        crystal = CrystalConfig.from_dict(raw.get("crystal"))
        scan = ScanConfig.from_dict(raw.get("scan"))
        io = IOConfig(**raw.get("io", {}))
        postprocess = PostprocessConfig(**raw.get("postprocess", {}))
        reciprocal = ReciprocalConfig.from_dict(raw.get("reciprocal"))
        return cls(
            crystal=crystal, scan=scan, io=io, postprocess=postprocess, reciprocal=reciprocal
        )


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
class IdentificationNoiseConfig:
    """Noise + intensity parameters for dfxm-identify forward calls.

    Sub-project B carry-out: these moved out of the old
    IdentificationScanConfig (now deleted) into their own block since
    they describe noise/detector, not the scan trajectory.
    """

    poisson_noise: bool = True
    rng_seed: int = 0
    intensity_scale: float = 7.0


@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    """Multi-disloc Monte Carlo parameters (mode='multi' only).

    v1.2.0: `n_png_previews` removed (PNG sidecars dropped). New opt-in
    `render_per_dislocation`: when True, each scan dir also writes
    per-dislocation detector files for unambiguous instance labels.
    """

    n_samples: int = 1000
    pos_std_um: float = 5.0
    render_per_dislocation: bool = False


@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    """z-scan mode parameters (mode='z-scan' only).

    Each (z_layer, b, α) configuration produces a (phi_steps × chi_steps)
    rocking-curve stack on disk (driven by `config.scan.phi` / `config.scan.chi`
    from the shared B+C ScanConfig), with a randomly-drawn secondary
    dislocation if `include_secondary` is True. The secondary is drawn
    once per (z, b, α) and shared across the rocking grid.

    v1.2.0: the duplicate `phi_range_deg / phi_steps / chi_range_deg /
    chi_steps` fields have been removed; the scan grid is now read from
    `[scan.phi]` / `[scan.chi]` via the shared ScanConfig.
    """

    z_offsets_um: list[float]
    include_secondary: bool = True
    # 0 = independent of the Poisson-noise stream (which uses
    # `default_rng(seed).spawn(2)[1]` in `_maybe_apply_poisson_noise`).
    # Bump to a different value only if a future RNG split needs slot 0.
    secondary_rng_offset: int = 0


@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    """Top-level config for dfxm-identify.

    Validates mode / sub-config / slip-plane consistency in __post_init__.
    """

    mode: Literal["single", "multi", "z-scan"]
    crystal: IdentificationCrystalConfig
    scan: ScanConfig  # shared with forward (was IdentificationScanConfig)
    noise: IdentificationNoiseConfig  # noise/intensity block (was flat in IdentificationScanConfig)
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None
    # Sub-project D: optional in Python construction; load_identification_config
    # requires it.
    reciprocal: ReciprocalConfig | None = None

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
        # z-scan mode owns the z dimension via z_offsets_um; [scan.z] would conflict.
        if self.mode == "z-scan" and self.scan.is_scanned("z"):
            raise ValueError(
                "mode='z-scan' uses [zscan].z_offsets_um for the z dimension; [scan.z] is forbidden"
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
    scan = ScanConfig.from_dict(data.get("scan"))  # shared ScanConfig
    noise = IdentificationNoiseConfig(**data.get("noise", {}))  # noise/intensity block
    io = IOConfig(**data.get("io", {}))
    multi = (
        IdentificationMonteCarloConfig(**data["multi"]) if data.get("multi") is not None else None
    )
    zscan = IdentificationZScanConfig(**data["zscan"]) if data.get("zscan") is not None else None
    reciprocal = ReciprocalConfig.from_dict(data.get("reciprocal"))

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        noise=noise,
        io=io,
        multi=multi,
        zscan=zscan,
        reciprocal=reciprocal,
    )


def _lookup_and_load_kernel(
    hkl: tuple[int, int, int],
    keV: float,
) -> None:
    """Pre-flight: look up the kernel npz matching (hkl, keV) and load it.

    Sub-project D replacement for `_ensure_kernel_loaded()`. Composes:
    1. `fm._lookup_kernel_path(hkl, keV, fm.pkl_fpath)` — glob + newest pick.
    2. `fm._load_default_kernel(path, expected_hkl=hkl, expected_keV=keV)` —
       load + bundled-metadata verification.

    Idempotent for the same (hkl, keV): if `fm._loaded_kernel_path` already
    matches what we'd look up, skip the reload. (Helpful for test loops and
    interactive REPL.)

    Raises FileNotFoundError on lookup miss, ValueError on metadata mismatch,
    KeyError on pre-sub-project-D legacy npz lacking metadata.
    """
    target = fm._lookup_kernel_path(hkl, keV, fm.pkl_fpath)
    if fm._loaded_kernel_path == target:
        return
    fm._load_default_kernel(
        str(target),
        expected_hkl=hkl,
        expected_keV=keV,
    )


def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    """Execute a DFXM forward-simulation run from a config object.

    Writes one `<output_dir>/dfxm_geo.h5` containing BLISS scan `/1.1`
    (dislocations) and, if `io.include_perfect_crystal=True`, `/2.1`
    (Hg=0 reference). For `crystal.mode='random_dislocations'`, also writes
    a `<output_dir>/dfxm_geo_random_dislocations.json` sidecar.
    """
    if config.reciprocal is None:
        raise ValueError(
            "SimulationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_simulation."
        )
    # v1.2.0 scope: the forward kernel only consumes the phi + chi axes from
    # ScanConfig. ScanGrid/build_scan_grid is implemented and tested but not
    # yet wired into write_simulation_h5. Raise eagerly so users don't get
    # silently-wrong output from scanning two_dtheta or z. Lifting this guard
    # is tracked as a v1.3.0 follow-up.
    unwired = [axis for axis in ("two_dtheta", "z") if config.scan.is_scanned(axis)]
    if unwired:
        raise ValueError(
            f"scan axes {unwired} are configured but not yet wired into the "
            f"forward kernel (v1.2.0 scope). For now, set range+steps only on "
            f"[scan.phi] and/or [scan.chi]."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dislocation population (dispatches on crystal.mode).
    fov_lateral_um = fm.Npixels * fm.psize * 1e6  # m -> um
    population = fm.build_dislocation_population(
        config.crystal, fov_lateral_um=fov_lateral_um, rng=None
    )

    # Write sidecar BEFORE forward kernel so a forward crash still leaves
    # the realized draw recoverable.
    if population.sidecar is not None:
        from dfxm_geo.io.sidecar import write_random_dislocations_sidecar

        sidecar_path = write_random_dislocations_sidecar(
            output_dir / "dfxm_geo", population.sidecar
        )
        print(f"[dfxm-forward] sidecar: {sidecar_path}", flush=True)

    # Effective-config print.
    print(
        f"[dfxm-forward] effective config:\n"
        f"  Nsub={fm.Nsub}  Npixels={fm.Npixels}  NN1={fm.NN1}  NN2={fm.NN2}\n"
        f"  kernel={fm._loaded_kernel_path}\n"
        f"  crystal.mode={config.crystal.mode}  ndis={len(population.positions_um)}\n"
        f"  scan.mode={config.scan.derived_mode_name()}  "
        f"axes_scanned={config.scan.scanned_axes()}",
        flush=True,
    )

    # Wall mode preserves legacy Find_Hg path (Fg cache + sidecar _vars.txt).
    # Centered + random_dislocations use Find_Hg_from_population.
    if config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        S = SAMPLE_REMOUNT_OPTIONS[w.sample_remount]
        Hg, q_hkl = fm.Find_Hg(
            w.dis,
            w.ndis,
            fm.psize,
            fm.zl_rms,
            h=config.reciprocal.hkl[0],
            k=config.reciprocal.hkl[1],
            l=config.reciprocal.hkl[2],
            S=S,
            remount_name=w.sample_remount,
        )
        sample_dis = w.dis
        sample_ndis = w.ndis
        sample_remount = w.sample_remount
    else:
        Hg, q_hkl = fm.Find_Hg_from_population(
            population,
            fm.psize,
            fm.zl_rms,
            h=config.reciprocal.hkl[0],
            k=config.reciprocal.hkl[1],
            l=config.reciprocal.hkl[2],
        )
        sample_dis = -1.0  # sentinel: not applicable for centered/random
        sample_ndis = len(population.positions_um)
        sample_remount = "N/A"

    fm.Hg = Hg
    fm.q_hkl = q_hkl

    config_toml = _dataclass_to_toml_str(config)

    h5_path = output_dir / "dfxm_geo.h5"
    write_simulation_h5(
        h5_path,
        Hg=Hg,
        q_hkl=q_hkl,
        phi_range=config.scan.phi.range or 0.0,
        phi_steps=config.scan.phi.steps or 1,
        chi_range=config.scan.chi.range or 0.0,
        chi_steps=config.scan.chi.steps or 1,
        include_perfect_crystal=config.io.include_perfect_crystal,
        sample_dis=sample_dis,
        sample_ndis=sample_ndis,
        sample_remount=sample_remount,
        config_toml=config_toml,
        cli=" ".join(sys.argv),
        max_workers=config.io.max_workers,
        crystal_mode=config.crystal.mode,
        scan_mode=config.scan.derived_mode_name(),
        scanned_axes=list(config.scan.scanned_axes()),
    )
    return {
        "h5_path": h5_path,
        "Hg": Hg,
        "q_hkl": q_hkl,
        "include_perfect_crystal": config.io.include_perfect_crystal,
    }


def _dataclass_to_toml_str(config: SimulationConfig) -> str:
    """Serialize a SimulationConfig back to TOML-formatted text.

    Renders:
      [reciprocal] (hkl, keV)
      [scan.<axis>] for each axis whose value/range/steps differ from default
      [crystal] mode + matching [crystal.<mode>] sub-block
      [io], [postprocess]

    The output is round-trippable through SimulationConfig.from_toml.
    """
    from dataclasses import asdict as _asdict

    lines: list[str] = []

    # [reciprocal]
    if config.reciprocal is not None:
        lines.append("[reciprocal]")
        h, k, l = config.reciprocal.hkl
        lines.append(f"hkl = [{h}, {k}, {l}]")
        lines.append(f"keV = {config.reciprocal.keV}")
        lines.append("")

    # [scan.<axis>] - only render axes that differ from default (skip if value==0
    # and not scanned, to keep TOML tidy).
    for axis_name in _CANONICAL_AXES:
        axis = getattr(config.scan, axis_name)
        if axis.value == 0.0 and not axis.is_scanned:
            continue
        lines.append(f"[scan.{axis_name}]")
        if axis.value != 0.0:
            lines.append(f"value = {axis.value}")
        if axis.is_scanned:
            lines.append(f"range = {axis.range}")
            lines.append(f"steps = {axis.steps}")
        lines.append("")

    # [crystal] + matching sub-block
    lines.append("[crystal]")
    lines.append(f'mode = "{config.crystal.mode}"')
    lines.append(f"[crystal.{config.crystal.mode}]")
    if config.crystal.mode == "centered":
        c = config.crystal.centered
        assert c is not None
        lines.append(f"b = [{c.b[0]}, {c.b[1]}, {c.b[2]}]")
        lines.append(f"n = [{c.n[0]}, {c.n[1]}, {c.n[2]}]")
        lines.append(f"t = [{c.t[0]}, {c.t[1]}, {c.t[2]}]")
    elif config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        lines.append(f"dis = {w.dis}")
        lines.append(f"ndis = {w.ndis}")
        lines.append(f'sample_remount = "{w.sample_remount}"')
    elif config.crystal.mode == "random_dislocations":
        rd = config.crystal.random_dislocations
        assert rd is not None
        lines.append(f"ndis = {rd.ndis}")
        if rd.sigma is not None:
            lines.append(f"sigma = {rd.sigma}")
        if rd.min_distance is not None:
            lines.append(f"min_distance = {rd.min_distance}")
        if rd.seed is not None:
            lines.append(f"seed = {rd.seed}")
    lines.append("")

    # [io] - render every field (default-skipping is brittle here since users
    # may explicitly want to set fields to defaults). Use asdict for simplicity.
    lines.append("[io]")
    for k_io, v_io in _asdict(config.io).items():
        if v_io is None:
            continue
        if isinstance(v_io, str):
            lines.append(f'{k_io} = "{v_io}"')
        elif isinstance(v_io, bool):
            lines.append(f"{k_io} = {str(v_io).lower()}")
        else:
            lines.append(f"{k_io} = {v_io}")
    lines.append("")

    # [postprocess]
    lines.append("[postprocess]")
    for k_pp, v_pp in _asdict(config.postprocess).items():
        if v_pp is None:
            continue
        if isinstance(v_pp, str):
            lines.append(f'{k_pp} = "{v_pp}"')
        elif isinstance(v_pp, bool):
            lines.append(f"{k_pp} = {str(v_pp).lower()}")
        else:
            lines.append(f"{k_pp} = {v_pp}")

    return "\n".join(lines) + "\n"


def run_postprocess(output_dir: Path, config: SimulationConfig) -> dict[str, Any]:
    """Read /1.1 and /2.1 from dfxm_geo.h5; compute χ-shift, COM maps, qi field.

    Analysis outputs are written into /1.1/dfxm_geo/analysis/ inside the same
    HDF5 file. SVG figures land on disk under <output_dir>/figures/ (F1).

    Warning:
        When invoked via ``--postprocess-only`` against an output dir whose
        stacks were produced with non-default ``dis`` / ``ndis``, the qi field
        is computed against the *module-level* ``fm.Hg``. If no prior
        ``run_simulation`` set ``fm.Hg`` in this process, ``fm.Hg`` will be
        None and this function will raise RuntimeError. For correctness in that
        workflow, call ``pipeline._lookup_and_load_kernel(hkl, keV)`` and
        assign ``fm.Hg`` explicitly before calling this function.

    Raises:
        FileNotFoundError: if the expected dfxm_geo.h5 file is absent.
        FileNotFoundError: if /2.1 (perfect crystal scan) is missing from the .h5.
    """
    h5_path = output_dir / "dfxm_geo.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Expected {h5_path}; run dfxm-forward without --postprocess-only first."
        )

    # Sanity-check that /2.1 exists (chi-shift needs perfect crystal).
    with h5py.File(h5_path, "r") as f:
        if "/2.1" not in f:
            raise FileNotFoundError(
                f"{h5_path} has no /2.1 scan (perfect crystal). Re-run with "
                "include_perfect_crystal=True, or skip postprocess."
            )

    phi_steps = config.scan.phi.steps or 1
    chi_steps = config.scan.chi.steps or 1
    phi_range = config.scan.phi.range or 0.0
    chi_range = config.scan.chi.range or 0.0

    _, dis_reshape, _, _ = load_h5_scan(
        h5_path,
        scan_id="1.1",
        phi_steps=phi_steps,
        chi_steps=chi_steps,
    )
    _, perf_reshape, _, _ = load_h5_scan(
        h5_path,
        scan_id="2.1",
        phi_steps=phi_steps,
        chi_steps=chi_steps,
    )

    chi_shift = compute_chi_shift(
        perf_reshape,
        chi_steps,
        chi_range,
        oversample=config.postprocess.chi_oversample_for_shift,
    )
    phi_list, chi_list = compute_com_maps(
        dis_reshape,
        phi_range,
        phi_steps,
        chi_range,
        chi_steps,
        chi_shift=chi_shift,
        oversample=config.postprocess.phi_oversample,
        chi_oversample=config.postprocess.chi_oversample,
    )
    if fm.Hg is None:
        raise RuntimeError("fm.Hg is not set. Call run_simulation() first.")
    _, qi_field = fm.forward(fm.Hg, phi=0, qi_return=True)

    # Append analysis to /1.1/dfxm_geo/analysis/ inside the existing .h5
    with h5py.File(h5_path, "a") as f:
        analysis = f.require_group("/1.1/dfxm_geo/analysis")
        for name, val in [
            ("phi_list", phi_list),
            ("chi_list", chi_list),
            ("qi_field", qi_field),
            ("chi_shift_deg", float(chi_shift)),
        ]:
            if name in analysis:
                del analysis[name]
            analysis.create_dataset(name, data=val)

    # F1: render SVG figures on disk alongside the .h5
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
        "h5_path": h5_path,
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


def _frame_grid_from_scan(
    scan: ScanConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (Phi_rad, Chi_rad, n_frames) for a single-scan rocking grid.

    Both arrays are 1-D of length n_frames, in phi-inner / chi-outer order.
    Fixed axes contribute a single repeated value (the axis ``value`` in
    radians).
    """
    if scan.phi.is_scanned:
        assert scan.phi.range is not None and scan.phi.steps is not None
        Phi = np.linspace(-np.deg2rad(scan.phi.range), np.deg2rad(scan.phi.range), scan.phi.steps)
    else:
        Phi = np.asarray([scan.phi.value], dtype=float)
    if scan.chi.is_scanned:
        assert scan.chi.range is not None and scan.chi.steps is not None
        Chi = np.linspace(-np.deg2rad(scan.chi.range), np.deg2rad(scan.chi.range), scan.chi.steps)
    else:
        Chi = np.asarray([scan.chi.value], dtype=float)
    return Phi, Chi, Phi.size * Chi.size


def _scan_frames_args(
    Hg: np.ndarray, Phi: np.ndarray, Chi: np.ndarray
) -> tuple[list[tuple[int, np.ndarray, float, float]], np.ndarray, np.ndarray]:
    """Build (args_list, phi_per_frame, chi_per_frame) for one ScanSpec.

    Frame order: phi-inner, chi-outer (matches forward fscan2d convention).
    Each args tuple is (frame_idx, Hg, phi_rad, chi_rad).
    """
    n = Phi.size * Chi.size
    args_list: list[tuple[int, np.ndarray, float, float]] = []
    phi_pf = np.empty(n, dtype=np.float64)
    chi_pf = np.empty(n, dtype=np.float64)
    for chi_idx in range(Chi.size):
        for phi_idx in range(Phi.size):
            k = chi_idx * Phi.size + phi_idx
            phi_pf[k] = float(Phi[phi_idx])
            chi_pf[k] = float(Chi[chi_idx])
            args_list.append((k, Hg, float(Phi[phi_idx]), float(Chi[chi_idx])))
    return args_list, phi_pf, chi_pf


def _positioners_for_scan(
    phi_pf: np.ndarray, chi_pf: np.ndarray, scan: ScanConfig
) -> dict[str, np.ndarray | float]:
    """Return phi/chi entries for ScanSpec.positioners.

    Fixed axes collapse to a scalar in radians; scanned axes are the
    full (N_frames,) array (also in radians, per ScanSpec convention).
    """
    out: dict[str, np.ndarray | float] = {}
    out["phi"] = phi_pf if scan.phi.is_scanned else float(scan.phi.value)
    out["chi"] = chi_pf if scan.chi.is_scanned else float(scan.chi.value)
    return out


def _identify_title(scan_mode: str, n_frames: int, scan: ScanConfig) -> str:
    """Compact human title for /N.1/title in identification masters."""
    return f"identify-{scan_mode} N_frames={n_frames}"


def _identification_config_to_toml_str(cfg: IdentificationConfig) -> str:
    """Best-effort TOML render of an IdentificationConfig (for /dfxm_geo/config_toml).

    Not round-trip-perfect; the goal is provenance, not reconstruction.
    Captures mode, reciprocal, scan, crystal (identification), noise, multi, zscan.
    """
    lines: list[str] = [f'mode = "{cfg.mode}"']
    if cfg.reciprocal is not None:
        h, k, l = cfg.reciprocal.hkl
        lines += [
            "",
            "[reciprocal]",
            f"hkl = [{h}, {k}, {l}]",
            f"keV = {cfg.reciprocal.keV}",
        ]
    # [crystal] (identification flavor; not the SimulationConfig crystal)
    c = cfg.crystal
    lines += [
        "",
        "[crystal]",
        f"slip_plane_normal = [{c.slip_plane_normal[0]}, "
        f"{c.slip_plane_normal[1]}, {c.slip_plane_normal[2]}]",
        f"angle_start_deg = {c.angle_start_deg}",
        f"angle_stop_deg = {c.angle_stop_deg}",
        f"angle_step_deg = {c.angle_step_deg}",
        f"sweep_all_slip_planes = {str(c.sweep_all_slip_planes).lower()}",
        f"exclude_invisibility = {str(c.exclude_invisibility).lower()}",
        f"invisibility_threshold_deg = {c.invisibility_threshold_deg}",
    ]
    if c.b_vector_indices is not None:
        lines.append(f"b_vector_indices = {list(c.b_vector_indices)}")
    # [scan.<axis>]
    for axis_name in _CANONICAL_AXES:
        axis = getattr(cfg.scan, axis_name)
        if axis.value == 0.0 and not axis.is_scanned:
            continue
        lines += ["", f"[scan.{axis_name}]"]
        if axis.value != 0.0:
            lines.append(f"value = {axis.value}")
        if axis.is_scanned:
            lines.append(f"range = {axis.range}")
            lines.append(f"steps = {axis.steps}")
    # [noise]
    lines += [
        "",
        "[noise]",
        f"poisson_noise = {str(cfg.noise.poisson_noise).lower()}",
        f"rng_seed = {cfg.noise.rng_seed}",
        f"intensity_scale = {cfg.noise.intensity_scale}",
    ]
    # [multi]
    if cfg.multi is not None:
        lines += [
            "",
            "[multi]",
            f"n_samples = {cfg.multi.n_samples}",
            f"pos_std_um = {cfg.multi.pos_std_um}",
            f"render_per_dislocation = {str(cfg.multi.render_per_dislocation).lower()}",
        ]
    # [zscan]
    if cfg.zscan is not None:
        lines += [
            "",
            "[zscan]",
            f"z_offsets_um = {list(cfg.zscan.z_offsets_um)}",
            f"include_secondary = {str(cfg.zscan.include_secondary).lower()}",
            f"secondary_rng_offset = {cfg.zscan.secondary_rng_offset}",
        ]
    return "\n".join(lines) + "\n"


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


def _iter_identification_single(
    config: IdentificationConfig,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per (plane, b_idx, alpha) configuration.

    Supports `[scan.phi]` / `[scan.chi]` from the shared ScanConfig: when
    either axis is scanned, each scan dir contains a (N_frames, H, W)
    stack with frame ordering phi-inner, chi-outer.
    """
    crystal_cfg = config.crystal
    # Noiseless frames are emitted here; intensity scaling and optional
    # Poisson noise are applied to the combined detector file post-write
    # by `_maybe_apply_poisson_noise` (called from the dispatcher).

    planes = (
        _ALL_111_PLANES if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )

    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for plane in planes:
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
        Ud_all = _ud_matrices(n_arr, rotated)

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

                args_list, phi_pf, chi_pf = _scan_frames_args(Hg, Phi, Chi)

                burgers_int = (
                    int(round(b_table[b_idx, 0] * np.sqrt(2))),
                    int(round(b_table[b_idx, 1] * np.sqrt(2))),
                    int(round(b_table[b_idx, 2] * np.sqrt(2))),
                )
                yield ScanSpec(
                    title=_identify_title(scan_mode, n_frames, config.scan),
                    sample={
                        "name": "simulated, dislocation identification (single)",
                        "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                        "burgers": np.asarray(burgers_int, dtype=np.int32),
                        "rotation_deg": float(alpha),
                    },
                    positioners=_positioners_for_scan(phi_pf, chi_pf, config.scan),
                    dfxm_geo={
                        "Hg": Hg,
                        "q_hkl": q_hkl,
                        "theta": float(fm.theta),
                        "psize": float(fm.psize),
                        "zl_rms": float(fm.zl_rms),
                    },
                    detectors={"dfxm_sim_detector": args_list},
                    attrs={
                        "scan_mode": scan_mode,
                        "scanned_axes": scanned_axes,
                        "identify_mode": "single",
                    },
                )


def _run_identification_single(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatcher: feed `_iter_identification_single` into write_identification_h5.

    Empty sweep (e.g. exclude_invisibility filters out everything) is
    allowed: the orchestrator writes an empty master with `n_images=0`.
    Mirrors the old behavior which also emitted an empty manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_toml = _identification_config_to_toml_str(config)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_single(config),
        cli=" ".join(sys.argv),
        config_toml=config_toml,
        max_workers=config.io.max_workers,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
    return {
        "n_images": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


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


def _build_dislocation_sample_entry(d: dict[str, Any]) -> dict[str, Any]:
    """Convert a `_draw_dislocation` output to an NXsample-shaped dict.

    Used to populate `/N.1/sample/dislocations/<idx>` inside the master
    identification HDF5. The Burgers vector is rounded to integer
    components (matches the `[h, k, l]` convention used for single mode).
    """
    return {
        "slip_plane_normal": np.asarray(d["plane"], dtype=np.int32),
        "burgers": np.asarray([int(round(c)) for c in d["b_vec"]], dtype=np.int32),
        "rotation_deg": float(d["alpha_deg"]),
        "position_um": np.asarray(d["pos_um"], dtype=float),
    }


def _iter_identification_multi(
    config: IdentificationConfig,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per Monte Carlo sample (2 random mixed dislocations).

    `render_per_dislocation=False` (default): a single detector
    (`dfxm_sim_detector`) holds the combined-scene frames. `=True`: two
    additional detectors (`dfxm_sim_detector_dis0`, `_dis1`) hold each
    dislocation rendered in isolation; these per-dislocation files are
    NOISELESS by construction (they bypass the post-write Poisson pass).

    All frames yielded here are NOISELESS; intensity scaling and
    optional Poisson noise are applied to the combined detector file
    post-write by `_maybe_apply_poisson_noise`.
    """
    assert config.multi is not None  # validated in __post_init__
    mc = config.multi
    noise_cfg = config.noise
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    fm.q_hkl = q_hkl

    # Split master rng → child streams. [0] = param draws (consumed here);
    # [1] = Poisson noise (consumed by _maybe_apply_poisson_noise, which
    # re-spawns with the same seed to get the same noise stream).
    param_rng, _noise_rng = np.random.default_rng(noise_cfg.rng_seed).spawn(2)

    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for _ in range(mc.n_samples):
        d1 = _draw_dislocation(param_rng, mc.pos_std_um)
        d2 = _draw_dislocation(param_rng, mc.pos_std_um)

        # Combined-scene Hg (sum of both dislocations)
        specs = [
            MixedDislocSpec(
                Ud_mix=d1["Ud"],
                rotation_deg=d1["alpha_deg"],
                position_lab_um=d1["pos_um"],
            ),
            MixedDislocSpec(
                Ud_mix=d2["Ud"],
                rotation_deg=d2["alpha_deg"],
                position_lab_um=d2["pos_um"],
            ),
        ]
        Fg_combined = Fd_find_multi_dislocs_mixed(fm.rl, fm.Us, specs, fm.Theta)
        Hg_combined = np.transpose(fast_inverse2(Fg_combined), [0, 2, 1]) - np.identity(3)

        combined_args, phi_pf, chi_pf = _scan_frames_args(Hg_combined, Phi, Chi)
        detectors: dict[str, list[tuple[int, np.ndarray, float, float]]] = {
            "dfxm_sim_detector": combined_args,
        }

        if mc.render_per_dislocation:
            # Per-dislocation Hg: each rendered alone (other one absent).
            # Noiseless by design — these are ground-truth instance labels.
            Fg_dis0 = Fd_find_mixed(
                fm.rl,
                fm.Us,
                Ud_mix=d1["Ud"],
                rotation_deg=d1["alpha_deg"],
                Theta=fm.Theta,
            )
            Hg_dis0 = np.transpose(fast_inverse2(Fg_dis0), [0, 2, 1]) - np.identity(3)
            Fg_dis1 = Fd_find_mixed(
                fm.rl,
                fm.Us,
                Ud_mix=d2["Ud"],
                rotation_deg=d2["alpha_deg"],
                Theta=fm.Theta,
            )
            Hg_dis1 = np.transpose(fast_inverse2(Fg_dis1), [0, 2, 1]) - np.identity(3)
            dis0_args, _, _ = _scan_frames_args(Hg_dis0, Phi, Chi)
            dis1_args, _, _ = _scan_frames_args(Hg_dis1, Phi, Chi)
            detectors["dfxm_sim_detector_dis0"] = dis0_args
            detectors["dfxm_sim_detector_dis1"] = dis1_args

        sample: dict[str, Any] = {
            "name": "simulated, dislocation identification (multi)",
            "dislocations": {
                "0": _build_dislocation_sample_entry(d1),
                "1": _build_dislocation_sample_entry(d2),
            },
        }

        yield ScanSpec(
            title=_identify_title(scan_mode, n_frames, config.scan),
            sample=sample,
            positioners=_positioners_for_scan(phi_pf, chi_pf, config.scan),
            dfxm_geo={
                "Hg": Hg_combined,
                "q_hkl": q_hkl,
                "theta": float(fm.theta),
                "psize": float(fm.psize),
                "zl_rms": float(fm.zl_rms),
            },
            detectors=detectors,
            attrs={
                "scan_mode": scan_mode,
                "scanned_axes": scanned_axes,
                "identify_mode": "multi",
            },
        )


def _run_identification_multi(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatcher: feed `_iter_identification_multi` into write_identification_h5.

    After the master + per-scan detector files are written, applies a
    post-write Poisson noise pass (and intensity scaling) to combined
    detector files only — per-dislocation files stay noiseless.
    """
    assert config.multi is not None  # validated in __post_init__
    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_multi(config),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
    return {
        "n_samples": config.multi.n_samples,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


def _maybe_apply_poisson_noise(
    config: IdentificationConfig, output_dir: Path, n_scans: int
) -> None:
    """Apply intensity scaling (always) and Poisson noise (if enabled) to
    combined-detector files post-write.

    Only modifies `dfxm_sim_detector_0000.h5` files; per-dislocation
    detector files (`*_dis0_0000.h5`, `*_dis1_0000.h5`) are intentionally
    left untouched so they remain deterministic ground-truth labels.

    Determinism contract: two runs at the same `noise.rng_seed` produce
    identical combined detector files. The noise stream is the second
    spawn of `np.random.default_rng(rng_seed).spawn(2)` — the same split
    used by `_iter_identification_*` for parameter draws (first stream).
    """
    noise_cfg = config.noise
    scale = noise_cfg.intensity_scale
    if scale == 1.0 and not noise_cfg.poisson_noise:
        return  # nothing to do; skip the per-scan file open
    rng = np.random.default_rng(noise_cfg.rng_seed).spawn(2)[1] if noise_cfg.poisson_noise else None
    for k in range(1, n_scans + 1):
        det_file = (
            output_dir / SCAN_DIR_FMT.format(k) / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
        )
        if not det_file.is_file():
            continue
        with h5py.File(det_file, "a") as f:
            img = f[DETECTOR_INTERNAL_PATH]
            arr = img[...] * scale
            if rng is not None:
                arr = rng.poisson(np.clip(arr, a_min=0.0, a_max=None)).astype(float)
            img[...] = arr


def _iter_identification_zscan(
    config: IdentificationConfig,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per (z_offset, plane, b_idx, alpha) configuration.

    Each ScanSpec carries the primary (deterministic, on-axis) dislocation
    in ``sample["primary"]`` and, when ``zscan.include_secondary`` is True,
    a randomly-drawn ``sample["secondary"]`` (one draw per configuration,
    shared across the rocking grid). The (phi, chi) rocking grid comes
    from ``config.scan.phi`` / ``config.scan.chi`` (shared B+C schema).

    All frames are noiseless; intensity scaling / Poisson noise are
    applied post-write by ``_maybe_apply_poisson_noise``.
    """
    assert config.zscan is not None  # validated in __post_init__
    zscan = config.zscan
    crystal_cfg = config.crystal
    noise_cfg = config.noise

    planes = (
        _ALL_111_PLANES if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )
    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    # Secondary stream uses SeedSequence child [secondary_rng_offset]. Default
    # is 0; _maybe_apply_poisson_noise uses child [1] from a spawn(2), so the
    # two streams are independent SeedSequence siblings.
    spawned = np.random.default_rng(noise_cfg.rng_seed).spawn(zscan.secondary_rng_offset + 1)
    secondary_rng = spawned[zscan.secondary_rng_offset]

    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for z_off in zscan.z_offsets_um:
        rl_shifted = fm.Z_shift(z_off)
        for plane in planes:
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
            Ud_all = _ud_matrices(n_arr, rotated)

            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                    q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_primary = Ud_all[i, j]
                    primary_spec = MixedDislocSpec(
                        Ud_mix=Ud_primary,
                        rotation_deg=float(alpha),
                        position_lab_um=(0.0, 0.0, 0.0),
                    )

                    burgers_int = np.asarray(
                        [
                            int(round(b_table[b_idx, 0] * np.sqrt(2))),
                            int(round(b_table[b_idx, 1] * np.sqrt(2))),
                            int(round(b_table[b_idx, 2] * np.sqrt(2))),
                        ],
                        dtype=np.int32,
                    )
                    sample: dict[str, Any] = {
                        "name": "simulated, dislocation identification (z-scan)",
                        "z_offset_um": float(z_off),
                        "primary": {
                            "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                            "burgers": burgers_int,
                            "rotation_deg": float(alpha),
                            "position_um": np.asarray([0.0, 0.0, 0.0]),
                        },
                    }

                    if zscan.include_secondary:
                        sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)
                        secondary_spec = MixedDislocSpec(
                            Ud_mix=sec["Ud"],
                            rotation_deg=sec["alpha_deg"],
                            position_lab_um=sec["pos_um"],
                        )
                        Fg = Fd_find_multi_dislocs_mixed(
                            rl_shifted,
                            fm.Us,
                            [primary_spec, secondary_spec],
                            fm.Theta,
                        )
                        sample["secondary"] = _build_dislocation_sample_entry(sec)
                    else:
                        Fg = Fd_find_mixed(
                            rl_shifted,
                            fm.Us,
                            Ud_mix=Ud_primary,
                            rotation_deg=float(alpha),
                            Theta=fm.Theta,
                        )

                    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                    args_list, phi_pf, chi_pf = _scan_frames_args(Hg, Phi, Chi)

                    yield ScanSpec(
                        title=_identify_title(scan_mode, n_frames, config.scan),
                        sample=sample,
                        positioners=_positioners_for_scan(phi_pf, chi_pf, config.scan),
                        dfxm_geo={
                            "Hg": Hg,
                            "q_hkl": q_hkl,
                            "theta": float(fm.theta),
                            "psize": float(fm.psize),
                            "zl_rms": float(fm.zl_rms),
                        },
                        detectors={"dfxm_sim_detector": args_list},
                        attrs={
                            "scan_mode": scan_mode,
                            "scanned_axes": scanned_axes,
                            "identify_mode": "z-scan",
                        },
                    )


def _run_identification_zscan(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatcher: feed ``_iter_identification_zscan`` into write_identification_h5.

    After the master + per-scan detector files are written, applies the
    post-write Poisson / intensity-scale pass (combined detector only;
    no per-dislocation files are produced in z-scan mode).
    """
    assert config.zscan is not None  # validated in __post_init__
    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_zscan(config),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
    return {
        "n_configurations": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single / multi / z-scan runner based on config.mode."""
    if config.reciprocal is None:
        raise ValueError(
            "IdentificationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_identification."
        )
    # v1.2.0 scope: identify kernels only consume phi + chi. ScanGrid for
    # two_dtheta / z is implemented but not wired into the identify forward
    # path. Raise eagerly so users don't get silently-wrong output. Lifting
    # this guard is tracked as a v1.3.0 follow-up.
    unwired = [axis for axis in ("two_dtheta", "z") if config.scan.is_scanned(axis)]
    if unwired:
        raise ValueError(
            f"scan axes {unwired} are configured but not yet wired into "
            f"identification (v1.2.0 scope). For now, set range+steps only on "
            f"[scan.phi] and/or [scan.chi]."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)

    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    if config.mode == "multi":
        return _run_identification_multi(config, output_dir)
    return _run_identification_zscan(config, output_dir)


def cli_main_identify(argv: list[str] | None = None) -> int:
    """Argparse-driven entry point for `dfxm-identify`."""
    parser = argparse.ArgumentParser(description="DFXM dislocation identification simulation")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to identification TOML config"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "z-scan"],
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
    elif cfg.mode == "multi":
        print(f"Wrote {result['n_samples']} samples to {result['output_dir']}")
    else:  # z-scan
        print(f"Wrote {result['n_configurations']} configurations to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
