"""Config dataclasses, TOML loaders and serializers for dfxm_geo — extracted from pipeline.py
(refactor gate, 2026-06-11). Import via ``dfxm_geo.pipeline`` (the stable facade) or directly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from dfxm_geo.crystal.burgers import (
    burgers_vectors as _burgers_vectors,
)
from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.crystal.reflections import (
    ReflectionRun as _ReflectionRun,
)
from dfxm_geo.crystal.reflections import (
    resolve_reflections as _resolve_reflections,
)
from dfxm_geo.crystal.reflections import (
    resolve_reflections_auto as _resolve_reflections_auto,
)
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.detector import resolve_model


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

# Keys in a [crystal] block that belong to the mount geometry
# (_crystal_mount_from_toml), NOT to the dislocation-layout schema
# (CrystalConfig) or the identify sweep schema (IdentificationCrystalConfig).
# Both CrystalConfig.from_dict and load_identification_config strip these
# before parsing their own schemas; keep them sharing this one source so the
# two strip sites can't drift (the identify list silently fell behind once).
_CRYSTAL_MOUNT_KEYS: frozenset[str] = frozenset(
    {
        "lattice",
        "a",
        "b",
        "c",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "cif",
        "space_group",
        "mount_x",
        "mount_y",
        "mount_z",
        # M4 Stage 4.3a: structure/material metadata + user slip-system hatch.
        "structure_type",
        "material",
        "poisson_ratio",
        "slip_families",
        "slip_system",  # [[crystal.slip_system]] array-of-tables (user escape hatch)
    }
)

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

    b: tuple[int, int, int] = (1, 0, -1)
    n: tuple[int, int, int] = (1, 1, 1)
    t: tuple[int, int, int] = (1, -2, 1)

    def __post_init__(self) -> None:
        # Accept 4-index Miller–Bravais notation; convert to 3-index before validation.
        # n is a PLANE (hkil → hkl); b and t are DIRECTIONS (uvtw → uvw).
        # Mirror the pattern used in IdentificationCrystalConfig.__post_init__.
        for field_name, field_val in (("b", self.b), ("n", self.n), ("t", self.t)):
            raw: tuple[int, ...] = tuple(int(x) for x in field_val)
            length = len(raw)
            if length == 4:
                from dfxm_geo.crystal.slip_systems import hkil_to_hkl, uvtw_to_uvw

                idx4 = (raw[0], raw[1], raw[2], raw[3])
                converted: tuple[int, int, int] = (
                    hkil_to_hkl(idx4) if field_name == "n" else uvtw_to_uvw(idx4)
                )
                setattr(self, field_name, converted)
            elif length != 3:
                raise ValueError(
                    f"[crystal.centered] {field_name} must be a 3- or 4-index tuple; "
                    f"got length {length}: {field_val!r}"
                )
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


@dataclass(kw_only=True)
class WallCrystalConfig:
    """Dis-spaced grid of dislocations (sub-project C, mode='wall').

    The current Borgi/Purdue IUCrJ 2024 layout. Sub-project F strips the
    publication-grade defaults: dis/ndis/sample_remount must be specified
    explicitly. This is the v2.0.0 breaking change. `kw_only=True` ensures
    the strip surfaces as a clear "missing N required keyword-only argument"
    TypeError rather than positional-arg confusion.
    """

    dis: float
    ndis: int
    sample_remount: str

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
    def default(cls) -> CrystalConfig:
        """Sub-project F: canonical empty-TOML default.

        Returns mode='centered' with CenteredCrystalConfig() (canonical FCC
        primary). Used as the SimulationConfig.crystal default factory and
        as the empty-TOML fallback in from_dict.
        """
        return cls(mode="centered", centered=CenteredCrystalConfig())

    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig:
        # Sub-project F: empty/missing [crystal] → canonical centered default.
        # Explicit `[crystal] mode = "<m>"` without `[crystal.<m>]` still raises
        # below; "default" is reached only by omission, not declaration.
        if not data:
            return cls.default()
        # Strip oblique [geometry] mount/cell keys (cif, space_group, lattice
        # cell params, mount axes) — these feed _crystal_mount_from_toml, NOT
        # the dislocation-layout schema here. A [crystal] block carrying only
        # those keys (no dislocation `mode`) falls through to the default.
        data = {k: v for k, v in data.items() if k not in _CRYSTAL_MOUNT_KEYS}
        if not data:
            return cls.default()
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
    # When False, omit the large per-ray Hg displacement-gradient array from
    # the master HDF5 /N.1/dfxm_geo group (keeps the tiny q_hkl/theta/psize/
    # zl_rms scalars). The Hg dump dominates per-config size (~106 MB); drop
    # it for batch/ML runs that don't need per-ray strain provenance.
    write_strain_provenance: bool = True


@dataclass
class PostprocessConfig:
    """Knobs for the post-processing stage (Phase 9.2).

    See ``docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md``.
    """

    enabled: bool = True
    # Deprecated: COM maps are now an exact weighted mean (v2.0.2) and the χ axis
    # is read off the nominal grid with no runtime calibration, so none of these
    # three knobs affect output any more. Retained so existing [postprocess] TOML
    # blocks still parse. (chi_oversample_for_shift used to drive compute_chi_shift,
    # which was dropped from postprocessing — see run_postprocess.)
    chi_oversample: int = 20
    phi_oversample: int = 20
    chi_oversample_for_shift: int = 100
    figures_dirname: str = "figures"
    data_dirname: str = "analysis"


@dataclass
class ReciprocalConfig:
    """Reflection identity + resolution-backend selection.

    `backend`: "auto" (default; analytic when beamstop off, else MC),
    "analytic" (force closed-form; errors if beamstop on), or "mc".
    The instrument params mirror reciprocal_space.kernel.generate_kernel and
    feed the analytic backend (the MC backend reads them from the kernel .npz).
    """

    hkl: tuple[int, int, int] = (-1, 1, -1)
    keV: float = 17.0
    backend: str = "auto"
    beamstop: bool = True  # matches generate_kernel default
    zeta_v_fwhm: float = 5.3e-4
    zeta_h_fwhm: float = 0.0
    NA_rms: float = 7.31e-4 / 2.35
    eps_rms: float = 1.41e-4 / 2.35
    zeta_v_clip: float = 1.4e-4
    eta: float = 0.0  # Azimuthal tilt (rad); 0.0 = simplified geometry (v2.2.0 default)
    # Cubic lattice parameter (m); Al default. Drives the Bragg-angle (theta)
    # derivation in __post_init__ and the analytic backend
    # (forward_model._load_analytic_resolution reads config.lattice_a).
    lattice_a: float = 4.0495e-10

    _VALID_BACKENDS = ("auto", "analytic", "mc")

    def __post_init__(self) -> None:
        if not isinstance(self.hkl, tuple):
            self.hkl = tuple(self.hkl)
        if self.backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self._VALID_BACKENDS}, got {self.backend!r}."
            )
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # lattice_a defaults to Al (4.0495e-10) but is now config-driven, so a
        # non-Al cubic lattice can be supplied via [reciprocal] lattice_a and
        # flows through to both the Bragg-validity check and the analytic backend.
        _validate_reflection(self.hkl, self.keV, UnitCell.cubic(self.lattice_a))

    @classmethod
    def from_dict(cls, data: dict | None) -> ReciprocalConfig:
        if not data:
            return cls()
        kwargs: dict = {}
        if "hkl" in data:
            kwargs["hkl"] = tuple(data["hkl"])
        if "keV" in data:
            kwargs["keV"] = float(data["keV"])
        for key in ("backend",):
            if key in data:
                kwargs[key] = str(data[key])
        if "beamstop" in data:
            kwargs["beamstop"] = bool(data["beamstop"])
        for key in (
            "zeta_v_fwhm",
            "zeta_h_fwhm",
            "NA_rms",
            "eps_rms",
            "zeta_v_clip",
            "eta",
            "lattice_a",
        ):
            if key in data:
                kwargs[key] = float(data[key])
        return cls(**kwargs)


@dataclass
class GeometryConfig:
    """Diffraction-geometry mode + oblique-angle parameters (v2.3.0).

    Populated from the TOML ``[geometry]`` block (``mode``, ``eta``) plus the
    crystal-mount fields in ``[crystal]`` (``lattice``/``a``/``mount_x/y/z``).
    For ``mode='oblique'`` the config ``eta`` is cross-checked against
    ``compute_omega_eta(mount, hkl, keV)`` — the *same* solver the bootstrap
    uses — and the matching ``(theta, omega)`` are stored. ``theta_validated``
    therefore equals the value baked into the oblique LUT filename, so the
    pipeline-side lookup resolves the bootstrapped kernel exactly.

    ``mode='simplified'`` (the default, and any config without a ``[geometry]``
    block) reproduces v2.2.0 behaviour: ``eta=0``, ``theta_validated=None``.
    """

    mode: str = "simplified"
    eta: float = 0.0
    theta_validated: float | None = None
    omega: float = 0.0
    mount: CrystalMount | None = None


def _maybe_inherit_cif_lattice_a(raw: dict, base_dir: Path | None) -> None:
    """[crystal] cif + cubic cell + no explicit [reciprocal] lattice_a → inherit a.

    Mutates ``raw`` in place BEFORE ReciprocalConfig.from_dict so the Bragg
    validity check and the analytic backend see the CIF lattice parameter.
    Explicit [reciprocal] lattice_a wins (per-key override rule). Non-cubic
    cells don't inherit — lattice_a is the CUBIC simplified-geometry knob.
    """
    crystal_raw = raw.get("crystal") or {}
    if "cif" not in crystal_raw:
        return
    if "lattice_a" in (raw.get("reciprocal") or {}):
        return
    from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

    mount = _crystal_mount_from_toml(crystal_raw, base_dir=base_dir)
    if mount.cell.is_cubic:
        raw.setdefault("reciprocal", {})["lattice_a"] = mount.a


def _build_geometry_config(
    raw: dict,
    reciprocal: ReciprocalConfig,
    multi_reflection: bool = False,
    base_dir: Path | None = None,
) -> GeometryConfig:
    """Build a GeometryConfig from raw TOML tables (``[geometry]`` + ``[crystal]``).

    Mirrors ``reciprocal_space.kernel.cli_main``'s bootstrap-side parsing so the
    consumer (forward/identify) and the producer (dfxm-bootstrap) agree on the
    geometry. For oblique mode, validates ``eta`` against the reflection geometry
    and resolves the Bragg ``theta`` / azimuthal ``omega`` from the solver.

    Args:
        raw: parsed TOML dict.
        reciprocal: already-built ReciprocalConfig (provides hkl + keV for
            the single-reflection eta cross-check).
        multi_reflection: when True (``[[reflections]]`` / ``[reflections_auto]``
            present), parse mode + mount but skip single-eta validation — each
            reflection carries its own eta/theta/omega resolved by
            ``_parse_reflections_tables``.

    Raises:
        ValueError: on a malformed ``[geometry]`` block, a missing mount key, or
            an ``eta`` that does not match any computed reflection geometry.
    """
    from dfxm_geo.crystal.cif import reject_extinct
    from dfxm_geo.reciprocal_space.kernel import (
        _crystal_mount_from_toml,
        _parse_geometry_block,
        _validate_eta_against_compute_omega_eta,
    )

    mode, eta = _parse_geometry_block(raw.get("geometry"), allow_missing_eta=multi_reflection)
    if mode == "simplified":
        crystal_raw = raw.get("crystal") or {}
        # M4 Stage 4.3a: structure-aware [crystal] keys are carried ONLY on the
        # oblique mount (simplified mode discards it -> mount=None -> FCC). Silently
        # computing FCC for a "structure_type=bcc" request is the bug this guards:
        # raise loudly instead of dropping the keys. (Plain cell/cif/space_group
        # keys are still accepted in simplified mode — see the branch below.)
        _structure_keys = (
            "structure_type",
            "material",
            "poisson_ratio",
            "slip_families",
            "slip_system",
        )
        _present = [k for k in _structure_keys if k in crystal_raw]
        if _present:
            raise ValueError(
                f'[crystal] {_present} require [geometry] mode = "oblique": the '
                "structure family / material / Poisson ratio / slip systems are "
                "carried on the oblique crystal mount, which simplified geometry "
                "discards (the run would silently resolve to FCC). Set "
                '[geometry] mode = "oblique" (with an eta) to use these keys.'
            )
        # A "lattice" or "cif" key signals an explicit bootstrap-style mount;
        # parse it (which may surface mount errors simplified mode previously
        # ignored) so non-cubic cells cannot slip through the cubic-only path.
        if "lattice" in crystal_raw or "cif" in crystal_raw:
            mount = _crystal_mount_from_toml(crystal_raw, base_dir=base_dir)
            if not mount.cell.is_cubic:
                raise ValueError(
                    "non-cubic [crystal] cells require [geometry] mode='oblique' "
                    "(simplified mode hardwires the cubic symmetric geometry)."
                )
            if not multi_reflection:
                reject_extinct(mount.space_group, reciprocal.hkl, "[reciprocal] hkl")
        elif "space_group" in crystal_raw and not multi_reflection:
            # Bare symmetry knowledge on an otherwise-default mount (no CIF,
            # no mount keys) still gates the configured reflection.
            reject_extinct(str(crystal_raw["space_group"]), reciprocal.hkl, "[reciprocal] hkl")
        return GeometryConfig(mode="simplified")

    mount = _crystal_mount_from_toml(raw.get("crystal"), base_dir=base_dir)

    # Diagnose a forbidden reflection before the (coarser) non-cubic-unsupported
    # gate: "systematically absent" is the more specific, actionable error.
    if not multi_reflection:
        reject_extinct(mount.space_group, reciprocal.hkl, "[reciprocal] hkl")

    if not mount.cell.is_cubic and mount.resolved_structure_type != "hcp":
        # M4 Stage 4.3b delivered HCP (hexagonal) forward/identify; other
        # non-cubic systems (orthorhombic/monoclinic/triclinic) still have no
        # slip-system registry, so they remain unsupported in the pipeline.
        # Common case: a hexagonal cell with no structure_type/space_group
        # resolves to the back-compat default 'fcc' and lands here — point the
        # user straight at the HCP knob rather than the generic message.
        hcp_hint = ""
        if mount.lattice in ("hexagonal", "trigonal"):
            hcp_hint = (
                f" This is a {mount.lattice} cell with no structure_type or "
                "space_group, so it resolved to the default 'fcc'; set "
                '[crystal] structure_type = "hcp" (or a P-hexagonal space_group / '
                "CIF) to use HCP slip systems."
            )
        raise ValueError(
            f"non-cubic cells with resolved structure "
            f"{mount.resolved_structure_type!r} are not yet supported in the "
            f"forward/identify pipeline (only cubic FCC/BCC and hexagonal HCP "
            f"are wired; M4 Stage 4.3b added HCP).{hcp_hint} Stage 4.1 supports "
            "arbitrary non-cubic lattices in dfxm-bootstrap and dfxm-find-reflections."
        )

    if multi_reflection:
        # Per-entry angles resolved by _parse_reflections_tables; return a
        # placeholder GeometryConfig with eta=0.0 — per-reflection angles live
        # on the ReflectionRun list, not on this shared GeometryConfig.
        return GeometryConfig(mode="oblique", eta=0.0, theta_validated=None, omega=0.0, mount=mount)

    theta_validated, omega = _validate_eta_against_compute_omega_eta(
        mount, reciprocal.hkl, reciprocal.keV, eta
    )
    return GeometryConfig(
        mode="oblique",
        eta=eta,
        theta_validated=theta_validated,
        omega=omega,
        mount=mount,
    )


def _parse_reflections_tables(
    raw: dict,
    geometry: GeometryConfig,
    reciprocal: ReciprocalConfig,
) -> list[_ReflectionRun]:
    """Resolve ``[[reflections]]`` / ``[reflections_auto]`` from raw TOML.

    Returns an empty list when neither block is present (single-reflection config).

    Validation rules per spec §5: oblique-only, mutually exclusive with
    ``[reciprocal] hkl`` and with each other.

    Raises:
        ValueError: on any of the three exclusivity or mode violations.
    """
    has_list = "reflections" in raw
    has_auto = "reflections_auto" in raw
    if not has_list and not has_auto:
        return []
    if has_list and has_auto:
        raise ValueError("[[reflections]] and [reflections_auto] are mutually exclusive.")
    if geometry.mode != "oblique":
        raise ValueError(
            "[[reflections]] / [reflections_auto] require [geometry] mode='oblique' "
            "(sweep simplified-mode reflections by emitting one config per hkl via "
            "the gen-sweep scripts instead)."
        )
    if "hkl" in raw.get("reciprocal", {}):
        raise ValueError(
            "[reciprocal] hkl and [[reflections]]/[reflections_auto] are mutually "
            "exclusive: list every reflection in the reflections table."
        )
    mount = geometry.mount
    if mount is None:
        raise ValueError(
            "[[reflections]] requires a [crystal] mount block (lattice/a/mount_x/y/z)."
        )
    keV = reciprocal.keV
    if has_auto:
        return _resolve_reflections_auto(raw["reflections_auto"], mount, keV)
    default_eta = raw.get("geometry", {}).get("eta")
    return _resolve_reflections(
        raw["reflections"],
        mount,
        keV,
        default_eta=float(default_eta) if default_eta is not None else None,
    )


@dataclass(frozen=True, kw_only=True)
class DetectorConfig:
    """[detector] block: realistic detector model + absolute calibration.

    Replaces the pre-v3 [noise] block. ``rng_seed`` is the run's stochastic
    seed: identification parameter draws use spawn child [0], detector noise
    child [1], the synthetic sensor map child [2] (SeedSequence children are
    stable under spawn-count growth, so [0]/[1] are bit-identical to the old
    layout when seeds match). ``counts_scale`` is the data anchor in ADU/s
    per normalized intensity unit (docs/detector-noise-model.md).
    """

    model: str = "pco_edge_4.2_id03"
    exposure_time: float = 1.0
    counts_scale: float = 1.0e4  # provisional anchor; data-anchored derivation NOT yet pinned (fails per-pixel sanity due to optic/pixel-pitch mismatch) — see docs/detector-noise-model.md and docs/calibration/derive_counts_scale.py
    rng_seed: int = 0

    def __post_init__(self) -> None:
        resolve_model(self.model)  # raises ValueError on unknown names
        if self.exposure_time <= 0:
            raise ValueError(f"exposure_time must be > 0, got {self.exposure_time}")
        if self.counts_scale <= 0:
            raise ValueError(f"counts_scale must be > 0, got {self.counts_scale}")


@dataclass
class SimulationConfig:
    # Sub-project F: crystal cascades to canonical-centered default (was: required).
    crystal: CrystalConfig = field(default_factory=CrystalConfig.default)
    scan: ScanConfig = field(default_factory=ScanConfig)
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Sub-project F: reciprocal cascades to Al 111 @ 17 keV (was: Optional[None]).
    reciprocal: ReciprocalConfig = field(default_factory=ReciprocalConfig)
    # v2.3.0: diffraction geometry (simplified vs oblique-angle).
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    # v2.6.0 (M3): resolved multi-reflection runs; empty = single-reflection config.
    reflections: list[_ReflectionRun] = field(default_factory=list)
    # v3 (detector-noise-model): realistic detector model + calibration.
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @classmethod
    def from_toml(cls, path: Path) -> SimulationConfig:
        """Load a SimulationConfig from a TOML file."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        base_dir = Path(path).parent
        _maybe_inherit_cif_lattice_a(raw, base_dir)
        crystal = CrystalConfig.from_dict(raw.get("crystal"))
        scan = ScanConfig.from_dict(raw.get("scan"))
        if "noise" in raw:
            raise ValueError(
                "[noise] was removed; use [detector] instead "
                "(poisson_noise -> model, intensity_scale -> counts_scale, "
                "rng_seed -> [detector] rng_seed). See docs/detector-noise-model.md."
            )
        detector = DetectorConfig(**raw.get("detector", {}))
        io = IOConfig(**raw.get("io", {}))
        postprocess = PostprocessConfig(**raw.get("postprocess", {}))
        reciprocal = ReciprocalConfig.from_dict(raw.get("reciprocal"))
        multi = ("reflections" in raw) or ("reflections_auto" in raw)
        geometry = _build_geometry_config(
            raw, reciprocal, multi_reflection=multi, base_dir=base_dir
        )
        reflections = _parse_reflections_tables(raw, geometry, reciprocal)
        # The analytic backend reads eta off ReciprocalConfig (see
        # forward_model._load_analytic_resolution). The user supplies it in
        # [geometry], so surface it there for single oblique runs only.
        if geometry.mode == "oblique" and not reflections:
            reciprocal.eta = geometry.eta
        return cls(
            crystal=crystal,
            scan=scan,
            io=io,
            postprocess=postprocess,
            reciprocal=reciprocal,
            geometry=geometry,
            reflections=reflections,
            detector=detector,
        )


@dataclass(frozen=True, kw_only=True)
class IdentificationCrystalConfig:
    """Crystal config for `dfxm-identify`. Slip plane + Burgers vector sweep."""

    slip_plane_normal: tuple[int, int, int] = (1, 1, 1)
    angle_start_deg: float = 0.0
    angle_stop_deg: float = 350.0
    angle_step_deg: float = 10.0
    b_vector_indices: list[int] | None = None  # None = all 6
    sweep_all_slip_planes: bool = True
    exclude_invisibility: bool = True
    invisibility_threshold_deg: float = 10.0

    def __post_init__(self) -> None:
        spn = self.slip_plane_normal
        if len(spn) == 4:
            from dfxm_geo.crystal.slip_systems import hkil_to_hkl

            object.__setattr__(self, "slip_plane_normal", hkil_to_hkl(tuple(int(x) for x in spn)))
        elif len(spn) != 3:
            raise ValueError(f"slip_plane_normal must be 3- or 4-index; got {spn!r}.")


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
    # 0 = independent of the detector-noise stream (which uses
    # `default_rng(seed).spawn(2)[1]` in `_apply_detector_model`; child[2]
    # is reserved for the sensor fixed-pattern map).
    # Bump to a different value only if a future RNG split needs slot 0.
    secondary_rng_offset: int = 0
    # When True (requires include_secondary), each scan dir also writes
    # `dfxm_sim_detector_primary` / `_secondary` files: the primary and the
    # secondary dislocation rendered in isolation. Noiseless by design (they
    # bypass the post-write Poisson pass) — ground-truth instance labels, the
    # z-scan analogue of multi mode's `render_per_dislocation`.
    render_per_dislocation: bool = False


@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    """Top-level config for dfxm-identify.

    Validates mode / sub-config / slip-plane consistency in __post_init__.
    """

    # Sub-project F: every field cascades to the empty-TOML default.
    mode: Literal["single", "multi", "z-scan"] = "single"
    crystal: IdentificationCrystalConfig = field(default_factory=IdentificationCrystalConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    io: IOConfig = field(default_factory=IOConfig)
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None
    # Sub-project F: reciprocal tightens to non-Optional + default.
    reciprocal: ReciprocalConfig = field(default_factory=ReciprocalConfig)
    # v2.3.0: diffraction geometry (simplified vs oblique-angle), mirroring
    # SimulationConfig. Defaults to simplified so existing identify configs
    # are unchanged.
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    # v2.6.0 (M3): resolved multi-reflection runs; empty = single-reflection config.
    reflections: list[_ReflectionRun] = field(default_factory=list)

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
        # Per-dislocation rendering needs a secondary to render in isolation;
        # with a single primary it would just duplicate the combined detector.
        if (
            self.zscan is not None
            and self.zscan.render_per_dislocation
            and not self.zscan.include_secondary
        ):
            raise ValueError(
                "zscan.render_per_dislocation=True requires include_secondary=True "
                "(nothing to separate from a single primary dislocation)"
            )
        # Validate the slip plane against the crystal's slip families.
        # With an oblique mount (geometry.mount is set), use the registry-driven
        # plane_normals for the resolved structure (fcc/bcc/custom) so BCC and
        # user-defined structures are accepted. Without a mount (simplified mode,
        # no [crystal] mount block), fall back to the legacy FCC-only check so
        # existing identify configs remain backward-compatible.
        mount = self.geometry.mount
        if mount is not None:
            from dfxm_geo.crystal.slip_systems import plane_normals

            structure = mount.resolved_structure_type
            valid_planes = plane_normals(
                structure, families=None
            )  # families=None: accept any plane valid for the structure (accept-more). Tasks 8/9 may tighten to families=mount.slip_families once Burgers lookup honors it.
            from dfxm_geo.crystal.slip_systems import _canon as _ss_canon

            if _ss_canon(self.crystal.slip_plane_normal) not in valid_planes:
                raise ValueError(
                    f"slip_plane_normal {self.crystal.slip_plane_normal!r} is not a valid "
                    f"slip plane for structure {structure!r}; valid: {valid_planes}"
                )
        else:
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
        ValueError: if `mode` is present but not one of {"single", "multi",
            "z-scan"}, or if the validation in
            IdentificationConfig.__post_init__ rejects the content. A missing
            top-level `mode` field defaults to `"single"` (sub-project F).
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    base_dir = Path(path).parent
    _maybe_inherit_cif_lattice_a(data, base_dir)

    # Sub-project F: 'mode' is now optional in TOML; defaults to 'single'.
    mode = data.get("mode", "single")

    crystal_data = dict(data.get("crystal", {}))
    # Mount keys belong to the oblique [geometry] machinery (read via
    # _build_geometry_config -> _crystal_mount_from_toml), NOT to the
    # IdentificationCrystalConfig dislocation-sweep schema. Strip them so an
    # oblique identify config's [crystal] block parses (forward's
    # CrystalConfig.from_dict filters the same keys by picking known fields).
    for _mount_key in _CRYSTAL_MOUNT_KEYS:
        crystal_data.pop(_mount_key, None)
    if "slip_plane_normal" in crystal_data:
        crystal_data = {
            **crystal_data,
            "slip_plane_normal": tuple(crystal_data["slip_plane_normal"]),
        }
    crystal = IdentificationCrystalConfig(**crystal_data)
    scan = ScanConfig.from_dict(data.get("scan"))  # shared ScanConfig
    if "noise" in data:
        raise ValueError(
            "[noise] was removed; use [detector] instead "
            "(poisson_noise -> model, intensity_scale -> counts_scale, "
            "rng_seed -> [detector] rng_seed). See docs/detector-noise-model.md."
        )
    detector = DetectorConfig(**data.get("detector", {}))
    io = IOConfig(**data.get("io", {}))
    multi = (
        IdentificationMonteCarloConfig(**data["multi"]) if data.get("multi") is not None else None
    )
    zscan = IdentificationZScanConfig(**data["zscan"]) if data.get("zscan") is not None else None
    reciprocal = ReciprocalConfig.from_dict(data.get("reciprocal"))
    multi_refl = ("reflections" in data) or ("reflections_auto" in data)
    geometry = _build_geometry_config(
        data, reciprocal, multi_reflection=multi_refl, base_dir=base_dir
    )
    reflections = _parse_reflections_tables(data, geometry, reciprocal)
    # Mirror SimulationConfig.from_toml: the analytic backend reads eta off
    # ReciprocalConfig, so surface the validated oblique eta there (single
    # reflection only — multi-reflection runs carry per-entry angles).
    if geometry.mode == "oblique" and not reflections:
        reciprocal.eta = geometry.eta

    return IdentificationConfig(
        mode=mode,
        crystal=crystal,
        scan=scan,
        detector=detector,
        io=io,
        multi=multi,
        zscan=zscan,
        reciprocal=reciprocal,
        geometry=geometry,
        reflections=reflections,
    )


def run_theta(config: SimulationConfig | IdentificationConfig) -> float:
    """The run's Bragg angle (radians).

    Oblique mode uses the solver result ``geometry.theta_validated``; simplified
    mode computes the reflection's true Bragg angle from ``(hkl, keV, lattice_a)``
    via ``_validate_reflection``. This makes the forward geometry consistent with
    the kernel (bootstrapped at the same angle) and fixes the prior simplified-
    reflection staleness (a non-default reflection used the import-default theta).

    Accepts both ``SimulationConfig`` and ``IdentificationConfig`` (both carry
    ``geometry`` + ``reciprocal`` sub-configs with the same interface).
    """
    geom = config.geometry
    if geom.mode == "oblique" and geom.theta_validated is not None:
        return float(geom.theta_validated)
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    r = config.reciprocal
    return float(_validate_reflection(r.hkl, r.keV, UnitCell.cubic(r.lattice_a)))


@dataclass(frozen=True, kw_only=True)
class ScanFrames:
    """Per-frame trajectory for one scan, all parallel arrays of length n_frames.

    Frame ordering: phi-innermost, chi, two_dtheta, z-outermost.
    Units: phi/chi/two_dtheta in radians; z in micrometers.
    """

    phi_pf: np.ndarray
    chi_pf: np.ndarray
    two_dtheta_pf: np.ndarray
    z_pf: np.ndarray
    n_frames: int


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
        # Record the backend selection so the embedded TOML round-trips the run
        # and the HDF5 provenance is self-describing (analytic vs MC).
        lines.append(f'backend = "{config.reciprocal.backend}"')
        lines.append(f"beamstop = {str(config.reciprocal.beamstop).lower()}")
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

    # [crystal] + matching sub-block. The crystal-mount fields (oblique
    # geometry, v2.3.0) live in the top-level [crystal] table and MUST be
    # emitted before the [crystal.<mode>] sub-table (TOML key-ordering rule)
    # — otherwise an oblique config round-trips to 'simplified' and the lossy
    # TOML misattributes oblique runs in the embedded HDF5 provenance.
    lines.append("[crystal]")
    mount = config.geometry.mount
    if mount is not None:
        lines.append(f'lattice = "{mount.lattice}"')
        lines.append(f"a = {mount.a}")
        if not mount.cell.is_cubic:
            lines.append(f"b = {mount.cell.b}")
            lines.append(f"c = {mount.cell.c}")
            lines.append(f"alpha_deg = {mount.cell.alpha_deg}")
            lines.append(f"beta_deg = {mount.cell.beta_deg}")
            lines.append(f"gamma_deg = {mount.cell.gamma_deg}")
        if mount.space_group is not None:
            lines.append(f'space_group = "{mount.space_group}"')
        lines.append(f"mount_x = [{mount.mount_x[0]}, {mount.mount_x[1]}, {mount.mount_x[2]}]")
        lines.append(f"mount_y = [{mount.mount_y[0]}, {mount.mount_y[1]}, {mount.mount_y[2]}]")
        lines.append(f"mount_z = [{mount.mount_z[0]}, {mount.mount_z[1]}, {mount.mount_z[2]}]")
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

    # [geometry] - only for oblique runs. Simplified mode is the default and
    # carries no [geometry] block, so omitting it keeps simplified configs
    # byte-for-byte unchanged. The mount fields are emitted in [crystal] above;
    # together they let _build_geometry_config re-derive (theta_validated,
    # omega) identically on round-trip.
    if config.geometry.mode == "oblique":
        lines.append("[geometry]")
        lines.append('mode = "oblique"')
        if not config.reflections:
            # Single-reflection: emit the validated eta so the config round-trips.
            lines.append(f"eta = {config.geometry.eta}")
        else:
            # Multi-reflection: eta=0.0 on GeometryConfig is a placeholder;
            # per-reflection angles live on the ReflectionRun list.
            # TODO(M3 plan 2): emit [[reflections]] here
            pass
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


def _identification_config_to_toml_str(cfg: IdentificationConfig) -> str:
    """Best-effort TOML render of an IdentificationConfig (for /dfxm_geo/config_toml).

    Not round-trip-perfect; the goal is provenance, not reconstruction.
    Captures mode, reciprocal, scan, crystal (identification), detector, multi, zscan.
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
    # [crystal] (identification flavor; not the SimulationConfig crystal).
    # Oblique mount fields (v2.3.0) are emitted FIRST in the table; together
    # with the [geometry] block below they let load_identification_config
    # re-derive the validated theta, so oblique identify provenance round-trips
    # instead of misattributing to 'simplified'. Mirrors _dataclass_to_toml_str.
    c = cfg.crystal
    lines += ["", "[crystal]"]
    mount = cfg.geometry.mount
    if mount is not None:
        lines += [
            f'lattice = "{mount.lattice}"',
            f"a = {mount.a}",
        ]
        if not mount.cell.is_cubic:
            lines += [
                f"b = {mount.cell.b}",
                f"c = {mount.cell.c}",
                f"alpha_deg = {mount.cell.alpha_deg}",
                f"beta_deg = {mount.cell.beta_deg}",
                f"gamma_deg = {mount.cell.gamma_deg}",
            ]
        if mount.space_group is not None:
            lines += [f'space_group = "{mount.space_group}"']
        lines += [
            f"mount_x = [{mount.mount_x[0]}, {mount.mount_x[1]}, {mount.mount_x[2]}]",
            f"mount_y = [{mount.mount_y[0]}, {mount.mount_y[1]}, {mount.mount_y[2]}]",
            f"mount_z = [{mount.mount_z[0]}, {mount.mount_z[1]}, {mount.mount_z[2]}]",
        ]
    lines += [
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
    # [detector]
    lines += [
        "",
        "[detector]",
        f'model = "{cfg.detector.model}"',
        f"exposure_time = {cfg.detector.exposure_time}",
        f"counts_scale = {cfg.detector.counts_scale}",
        f"rng_seed = {cfg.detector.rng_seed}",
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
            f"render_per_dislocation = {str(cfg.zscan.render_per_dislocation).lower()}",
        ]
    # [geometry] — oblique runs only (mirrors _dataclass_to_toml_str). With the
    # mount emitted in [crystal] above, load_identification_config re-derives the
    # validated theta so oblique identify provenance round-trips.
    if cfg.geometry.mode == "oblique":
        lines += [
            "",
            "[geometry]",
            'mode = "oblique"',
        ]
        if not cfg.reflections:
            # Single-reflection: emit the validated eta so the config round-trips.
            lines.append(f"eta = {cfg.geometry.eta}")
        else:
            # Multi-reflection: eta=0.0 on GeometryConfig is a placeholder;
            # per-reflection angles live on the ReflectionRun list.
            # TODO(M3 plan 2): emit [[reflections]] here
            pass
    return "\n".join(lines) + "\n"
