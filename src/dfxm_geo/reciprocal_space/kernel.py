"""Driver: generates a reciprocal-space resolution kernel npz.

Configures the Monte Carlo integration parameters and calls
:func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`. Side
effects: writes ``pkl_files/Resq_i_<timestamp>.npz`` (with bundled scalar
params) to the current working directory.

Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
17 keV, beamstop ON via square aperture of side 25 mm at the BFP).

Run as a script::

    python -m dfxm_geo.reciprocal_space.kernel
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

_DEFAULT_AL_CRYSTAL = CrystalMount(
    lattice="cubic",
    a=4.0495e-10,  # legacy default (Al); paper §6.1 uses 4.0493
    mount_x=(1, 0, 0),
    mount_y=(0, 1, 0),
    mount_z=(0, 0, 1),
)


def _crystal_mount_from_toml(data: dict | None) -> CrystalMount:
    """Build a CrystalMount from a `[crystal]` TOML block (or None → default Al).

    Optional cell parameters ``b``/``c`` (metres) and ``alpha_deg``/``beta_deg``/
    ``gamma_deg`` (degrees) extend the mount beyond cubic; constrained values
    are filled per crystal system (see ``UnitCell.from_lattice``).
    """
    if data is None:
        return _DEFAULT_AL_CRYSTAL

    def _opt(key: str) -> float | None:
        return float(data[key]) if key in data else None

    try:
        return CrystalMount(
            lattice=data["lattice"],
            a=float(data["a"]),
            b=_opt("b"),
            c=_opt("c"),
            alpha_deg=_opt("alpha_deg"),
            beta_deg=_opt("beta_deg"),
            gamma_deg=_opt("gamma_deg"),
            mount_x=tuple(int(x) for x in data["mount_x"]),  # type: ignore[arg-type]
            mount_y=tuple(int(x) for x in data["mount_y"]),  # type: ignore[arg-type]
            mount_z=tuple(int(x) for x in data["mount_z"]),  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise ValueError(f"[crystal] block missing key: {exc.args[0]}") from None


def _parse_geometry_block(
    data: dict | None,
    *,
    allow_missing_eta: bool = False,
) -> tuple[str, float]:
    """Parse [geometry] block. Returns (mode, eta_rad).

    Args:
        data: raw ``[geometry]`` TOML table, or None.
        allow_missing_eta: when True and mode='oblique' lacks ``eta``, return
            ``("oblique", float("nan"))`` instead of raising.  Used by the
            multi-reflection path where per-entry η values are resolved
            separately from ``[[reflections]]`` entries.
    """
    if data is None:
        return "simplified", 0.0
    mode = data.get("mode", "simplified")
    if mode not in ("simplified", "oblique"):
        raise ValueError(f"[geometry] mode must be 'simplified' or 'oblique'; got {mode!r}.")
    if mode == "simplified":
        if "eta" in data and float(data["eta"]) != 0.0:
            print(
                f"warning: simplified mode forces eta=0; ignoring [geometry] eta={data['eta']}.",
                file=sys.stderr,
            )
        return "simplified", 0.0
    # oblique
    if "eta" not in data:
        if allow_missing_eta:
            return "oblique", float("nan")
        raise ValueError("[geometry] mode='oblique' requires [geometry] eta (radians).")
    eta = float(data["eta"])
    if not math.isfinite(eta):
        raise ValueError(f"[geometry] eta must be finite, got {eta!r}.")
    return "oblique", eta


def _validate_eta_against_compute_omega_eta(
    mount: CrystalMount,
    hkl: tuple[int, int, int],
    keV: float,
    config_eta: float,
    *,
    tol: float = 1e-3,
) -> tuple[float, float]:
    """Cross-check the config's eta against compute_omega_eta(mount, hkl, keV).

    Returns (theta_rad, omega_rad) of the matching ω-solution. Raises
    ValueError with a diff if neither (η₁, η₂) matches.

    Default tol = 1e-3 rad (~0.06°): strict enough to catch user typos
    (a wrong-η-by-degree fails), loose enough to accept paper-quoting
    precision (e.g. "20.233°" rounds to 0.353142 rad while the solver
    returns 0.353125 rad — a 1.5e-5 gap that's well below physical relevance).
    """
    geom = compute_omega_eta(mount, hkl, keV)
    if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
        raise ValueError(
            f"Laue condition unsatisfiable for hkl={hkl}, mount={mount}, keV={keV}. "
            "Try a higher keV or a different mount; use 'dfxm-find-reflections' to "
            "enumerate reachable reflections."
        )
    candidates = [
        (geom.eta_1, geom.theta_1, geom.omega_1),
        (geom.eta_2, geom.theta_2, geom.omega_2),
    ]
    for eta_i, theta_i, omega_i in candidates:
        if not np.isnan(eta_i) and abs(eta_i - config_eta) <= tol:
            return float(theta_i), float(omega_i)
    raise ValueError(
        f"Config [geometry] eta={config_eta:.6f} rad does not match the computed "
        f"reflection geometry: (η₁={geom.eta_1:.6f}, η₂={geom.eta_2:.6f}) at "
        f"hkl={hkl}, keV={keV}. Use 'dfxm-find-reflections' to find valid groups."
    )


def _default_theta_al_111(keV: float = 17) -> float:
    """Bragg angle for Al 111 at the given beam energy (default 17 keV)."""
    a = 4.0495e-10  # Al lattice parameter, m
    d_111 = a / np.sqrt(3)
    wavelength = 1.239841984e-9 / keV
    return float(np.arcsin(wavelength / (2 * d_111)))


def _validate_reflection(
    hkl: tuple[int, int, int],
    keV: float,
    cell: UnitCell,
) -> float:
    """Compute and validate the Bragg angle θ for an arbitrary reflection.

    Args:
        hkl: Miller indices (must be ints, length 3, not all zero).
        keV: beam energy in keV (must be > 0).
        cell: unit cell; d-spacing is the metric-tensor form d = 2π/|B·G|
            (bit-identical to the legacy a/√(h²+k²+l²) for cubic cells).

    Returns:
        Bragg angle θ in radians.

    Raises:
        ValueError: on structural input errors or unsatisfiable Bragg geometry.

    Emits warnings to stderr when θ ∉ [5°, 85°] (unusual but valid reflections).
    """
    import sys

    if len(hkl) != 3:
        raise ValueError(f"hkl must have 3 components, got {len(hkl)}.")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in hkl):
        raise ValueError(f"hkl components must be int, got {hkl}.")
    if all(c == 0 for c in hkl):
        raise ValueError("hkl=(0,0,0) is not a valid reflection (no diffraction).")
    if keV <= 0:
        raise ValueError(f"keV must be > 0, got {keV}.")

    d_hkl = cell.d_spacing(hkl)
    wavelength = 1.239841984e-9 / keV  # hc/E, metres
    sin_theta = wavelength / (2 * d_hkl)
    if sin_theta > 1:
        lam_A = wavelength * 1e10
        two_d_A = 2 * d_hkl * 1e10
        raise ValueError(
            f"Bragg condition unsatisfiable: lam={lam_A:.4f} A, "
            f"2*d_hkl={two_d_A:.4f} A, sin(theta) = {sin_theta:.4f} > 1 for "
            f"hkl={hkl} at {keV} keV. Pick a lower-order reflection or "
            f"higher beam energy."
        )

    theta = float(np.arcsin(sin_theta))
    theta_deg = float(np.degrees(theta))
    if theta_deg < 5.0:
        print(
            f"warning: theta = {theta_deg:.2f} deg is very low (< 5 deg); reflection unusual but valid.",
            file=sys.stderr,
        )
    elif theta_deg > 85.0:
        print(
            f"warning: theta = {theta_deg:.2f} deg near back-reflection (> 85 deg); "
            f"reflection unusual but valid.",
            file=sys.stderr,
        )
    return theta


def _build_kernel_filename(
    mode: str,
    hkl: tuple[int, int, int],
    keV: float,
    *,
    theta: float = 0.0,
    eta: float = 0.0,
    date: str,
) -> str:
    """Per-mode kernel npz basename.

    simplified: ``Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz`` (legacy, v2.2.0 pattern).
    oblique:    ``Resq_i_theta{θ:.4f}rad_eta{η:.4f}rad_{keV:g}keV_{date}.npz``.

    `:g` formatting drops trailing zeros on keV (17.0 → "17", 17.5 → "17.5").
    Negative hkl components render naturally (-1 → "h-1").
    """
    if mode == "simplified":
        h, k, l = hkl
        return f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz"
    if mode == "oblique":
        return f"Resq_i_theta{theta:.4f}rad_eta{eta:.4f}rad_{keV:g}keV_{date}.npz"
    raise ValueError(f"unknown geometry mode: {mode!r}")


def _kernel_glob_pattern(theta: float, eta: float, keV: float) -> str:
    """Glob pattern matching any existing oblique kernel for a (theta, eta, keV) group.

    Uses 4-decimal-place formatting consistent with ``_build_kernel_filename``.
    Two geometries within 5e-5 rad of each other share the same pattern; in
    practice no realistic multi-reflection set collides at this granularity.
    """
    return f"Resq_i_theta{theta:.4f}rad_eta{eta:.4f}rad_{keV:g}keV_*.npz"


def _bootstrap_multi_reflection(
    *,
    args: argparse.Namespace,
    data: dict,
    mount: CrystalMount,
    mode: str,
    config_eta: float,
    raw_keV: float | None,
    reciprocal_kwargs: dict,
    pkl_fpath: str,
) -> int:
    """Implement the multi-reflection bootstrap loop extracted from ``cli_main``.

    Called when the TOML carries ``[[reflections]]`` or ``[reflections_auto]``.
    Returns an integer exit code (0 = success, 1 = error).
    """
    from dfxm_geo.crystal.reflections import (
        resolve_reflections,
        resolve_reflections_auto,
    )

    # Guard: [reciprocal] hkl is mutually exclusive with multi-reflection mode.
    # The reflections list IS the hkl source; a bare hkl key would be ambiguous.
    if "hkl" in data.get("reciprocal", {}):
        print(
            "error: [reciprocal] hkl is mutually exclusive with "
            "[[reflections]] / [reflections_auto]; "
            "remove hkl from [reciprocal] and specify it per-reflection instead.",
            file=sys.stderr,
        )
        return 1

    if mode != "oblique":
        print(
            "error: [[reflections]] / [reflections_auto] require [geometry] mode='oblique'.",
            file=sys.stderr,
        )
        return 1

    if raw_keV is None:
        print(
            "warning: no [reciprocal] keV; defaulting to 17.0 keV.",
            file=sys.stderr,
        )
    keV_multi = float(raw_keV) if raw_keV is not None else 17.0

    # [geometry] eta (if present) is forwarded to the resolver as the
    # default branch-selector; nan (missing) means per-entry or solution-1.
    default_eta_multi: float | None = None if math.isnan(config_eta) else config_eta

    try:
        if "reflections_auto" in data:
            runs = resolve_reflections_auto(data["reflections_auto"], mount, keV_multi)
        else:
            runs = resolve_reflections(
                data["reflections"],
                mount,
                keV_multi,
                default_eta=default_eta_multi,
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Resolve output directory.
    if args.output is not None:
        if args.output.exists() and args.output.is_file():
            print(
                f"error: --output {args.output} is a file; "
                "multi-reflection mode needs a directory.",
                file=sys.stderr,
            )
            return 1
        out_dir = args.output
    else:
        out_dir = Path(pkl_fpath)

    out_dir.mkdir(parents=True, exist_ok=True)

    # One kernel per unique group; pick the first member as the group
    # representative for filename + theta/eta/omega.
    date_multi = datetime.now().strftime("%Y%m%d_%H%M")
    groups_seen: dict[int, dict] = {}  # group id → manifest entry
    for run in runs:
        if run.group in groups_seen:
            # Already scheduled / built; just add this reflection to the entry.
            groups_seen[run.group]["reflections"].append(run.hkl)
            groups_seen[run.group]["omegas"].append(run.omega)
            continue

        # First member of this group → representative.
        # --if-missing: scan for any existing kernel matching the (theta, eta, keV)
        # pattern rather than an exact filename.  A fresh run stamps a new datetime
        # in the name, so an exact-path check would never match a previous run's file.
        pattern = _kernel_glob_pattern(run.theta, run.eta, run.keV)
        existing = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

        if existing:
            if args.if_missing:
                # Reuse the newest matching file; record its name in the manifest.
                newest = existing[-1]
                print(f"kernel already present at {newest}; skipping.")
                groups_seen[run.group] = {
                    "group": run.group,
                    "theta": run.theta,
                    "eta": run.eta,
                    "keV": run.keV,
                    "filename": newest.name,
                    "reflections": [run.hkl],
                    "omegas": [run.omega],
                }
                continue
            if not args.force:
                print(
                    f"refusing to overwrite existing kernel npz matching {pattern}; "
                    "pass --force to regenerate or --if-missing to reuse.",
                    file=sys.stderr,
                )
                return 1
            # --force: fall through and generate under a new timestamped name.

        filename = _build_kernel_filename(
            "oblique",
            run.hkl,
            run.keV,
            theta=run.theta,
            eta=run.eta,
            date=date_multi,
        )
        kernel_path = out_dir / filename

        # Build generate_kernel kwargs: instrument params from [reciprocal]
        # plus the per-group geometry.
        gk_kwargs = dict(reciprocal_kwargs)
        gk_kwargs["theta"] = run.theta
        gk_kwargs["hkl"] = run.hkl
        gk_kwargs["keV"] = run.keV
        if args.seed is not None:
            gk_kwargs["seed"] = args.seed

        theta_deg = float(np.degrees(run.theta))
        print(
            f"reflection: hkl={run.hkl}, keV={run.keV:g} -> theta = {theta_deg:.4f} deg "
            f"(group {run.group}, eta={run.eta:.4f} rad)"
        )
        written = generate_kernel(
            output_path=kernel_path,
            mode="oblique",
            eta=run.eta,
            mount=mount,
            omega=run.omega,
            **gk_kwargs,
        )
        print(f"wrote {written}")

        groups_seen[run.group] = {
            "group": run.group,
            "theta": run.theta,
            "eta": run.eta,
            "keV": run.keV,
            "filename": filename,
            "reflections": [run.hkl],
            "omegas": [run.omega],
        }

    # Write manifest alongside the kernels.
    manifest_entries = list(groups_seen.values())
    _write_kernel_manifest(out_dir / "kernel_manifest.toml", manifest_entries)
    return 0


def _write_kernel_manifest(path: Path, entries: list[dict]) -> None:
    """Render kernel_manifest.toml (no tomli-w dependency; hand-rendered).

    Each entry must have keys: group, theta, eta, keV, filename,
    reflections (list of hkl tuples), omegas (list of floats).
    """
    lines = [
        "# generated by dfxm-bootstrap — one row per unique (theta, eta, keV) kernel group",
        "",
    ]
    for e in entries:
        lines += [
            "[[kernels]]",
            f"group = {e['group']}",
            f"theta = {e['theta']!r}",
            f"eta = {e['eta']!r}",
            f"keV = {e['keV']!r}",
            f'filename = "{e["filename"]}"',
            "reflections = [" + ", ".join(str(list(h)) for h in e["reflections"]) + "]",
            "omegas = [" + ", ".join(repr(o) for o in e["omegas"]) + "]",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    npoints1: int = 400,
    npoints2: int = 200,
    npoints3: int = 200,
    qi1_range: float = 5e-4,
    qi2_range: float = 0.75e-2,
    qi3_range: float = 0.75e-2,
    zeta_v_fwhm: float = 5.3e-04,
    zeta_h_fwhm: float = 0,
    NA_rms: float = 7.31e-4 / 2.35,
    eps_rms: float = 1.41e-4 / 2.35,
    theta: float = _default_theta_al_111(17),
    D: float = float(2 * np.sqrt(50e-6 * 1.6e-3)),
    d1: float = 0.274,
    beamstop: bool = True,
    bs_height: float = 25e-3,
    aperture: bool = True,
    knife_edge: bool = False,
    dphi_range: float = 0.0,
    output_path: Path | None = None,
    hkl: tuple[int, int, int] | None = None,
    keV: float | None = None,
    seed: int | None = None,
    mode: str = "simplified",
    eta: float = 0.0,
    mount: CrystalMount | None = None,
    omega: float = 0.0,
    batch_size: int | None = None,
) -> Path:
    """Run the kernel-generation Monte Carlo and write the npz to ``pkl_files/``.

    Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
    17 keV; square-aperture beamstop with 25 mm side at the BFP).

    Args:
        date: Timestamp tag for the output filenames. Defaults to
            ``YYYYmmdd_HHMM`` from the current local time.
        Nrays: number of Monte Carlo rays.
        npoints1/2/3: voxel counts for the qi grid.
        qi1_range/qi2_range/qi3_range: half-widths of the qi grid.
        zeta_v_fwhm/zeta_h_fwhm: incoming-beam divergence FWHM (rad).
        NA_rms/eps_rms: objective NA / energy-bandwidth rms.
        theta: Bragg angle (rad).
        D: physical objective aperture (m).
        d1: sample-objective distance (m).
        beamstop/bs_height/aperture/knife_edge: beamstop config; see
            :func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`.
        dphi_range: rocking-curve sweep half-width (rad).
        hkl: Miller indices of the reflection (optional). Bundled into the
            npz scalar metadata for downstream load verification
            (sub-project D — multi-reflection lookup).
        keV: beam energy in keV (optional). Bundled into the npz scalar
            metadata for downstream load verification (sub-project D).
        seed: optional RNG seed for the Monte Carlo ray sampling. When given,
            two runs with identical parameters produce a bit-identical kernel
            (useful for regression fixtures and reproducible cluster reruns).
            When None (default), an entropy-seeded `default_rng()` is used and
            the bundled `seed` metadata is recorded as the -1 sentinel.
        mode: geometry mode — ``"simplified"`` (default, v2.2.0 behaviour) or
            ``"oblique"`` (oblique-angle DFXM, v2.3.0+).  Preserved in npz
            metadata as ``geometry_mode``.
        eta: sample-tilt angle η (radians).  0.0 in simplified mode; set from
            ``[geometry] eta`` in the TOML for oblique mode.
        mount: :class:`CrystalMount` describing the crystal orientation.
            ``None`` (default) falls back to :data:`_DEFAULT_AL_CRYSTAL`.
        omega: azimuthal motor angle ω (radians) derived from the geometry
            validation; 0.0 in simplified mode.
        batch_size: when set, process the Monte Carlo rays in batches of this
            size, accumulating into the histogram, so peak memory is
            ~O(batch_size) instead of ~O(Nrays). Lets the full Nrays=1e8
            bootstrap run on a memory-constrained machine at the cost of a few
            extra seconds. ``None`` (default) keeps the single-shot path. A
            single batch (batch_size >= Nrays) is bit-identical to None.

    Returns:
        The path the npz was written to.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    rng = np.random.default_rng(seed)
    phys_aper = D / d1

    if output_path is not None:
        output_path = Path(output_path)

    _mount = mount if mount is not None else _DEFAULT_AL_CRYSTAL

    kernel_meta = {
        "Nrays": np.int64(Nrays),
        "npoints1": np.int64(npoints1),
        "npoints2": np.int64(npoints2),
        "npoints3": np.int64(npoints3),
        "qi1_range": np.float64(qi1_range),
        "qi2_range": np.float64(qi2_range),
        "qi3_range": np.float64(qi3_range),
        "zeta_v_fwhm": np.float64(zeta_v_fwhm),
        "zeta_h_fwhm": np.float64(zeta_h_fwhm),
        "NA_rms": np.float64(NA_rms),
        "eps_rms": np.float64(eps_rms),
        "theta": np.float64(theta),
        "D": np.float64(D),
        "d1": np.float64(d1),
        "phys_aper": np.float64(phys_aper),
        "beamstop": np.bool_(beamstop),
        "bs_height": np.float64(bs_height),
        "aperture": np.bool_(aperture),
        "knife_edge": np.bool_(knife_edge),
        "dphi_range": np.float64(dphi_range),
        # Sub-project D: reflection identity. Verified by
        # _load_default_kernel when a lookup expects a specific (hkl, keV).
        "hkl": np.array(hkl if hkl is not None else (0, 0, 0), dtype=np.int64),
        "keV": np.float64(keV if keV is not None else 0.0),
        # MC seed provenance. -1 sentinel = unseeded (entropy) run.
        "seed": np.int64(seed if seed is not None else -1),
        # Oblique-angle metadata (defaults preserve v2.2.0 LUT consumability).
        "eta": np.float64(eta),
        "geometry_mode": np.str_(mode),
        "lattice": np.str_(_mount.lattice),
        "a": np.float64(_mount.a),
        "mount_x": np.array(_mount.mount_x, dtype=np.int64),
        "mount_y": np.array(_mount.mount_y, dtype=np.int64),
        "mount_z": np.array(_mount.mount_z, dtype=np.int64),
        "omega": np.float64(omega),
    }

    reciprocal_res_func(
        Nrays,
        npoints1,
        npoints2,
        npoints3,
        qi1_range,
        qi2_range,
        qi3_range,
        plot_figs=False,
        save_resqi=True,
        zeta_v_fwhm=zeta_v_fwhm,
        zeta_h_fwhm=zeta_h_fwhm,
        NA_rms=NA_rms,
        eps_rms=eps_rms,
        theta=theta,
        phys_aper=phys_aper,
        date=date,
        beamstop=beamstop,
        bs_height=bs_height,
        aperture=aperture,
        knife_edge=knife_edge,
        dphi_range=dphi_range,
        eta=eta,
        output_path=output_path,
        kernel_meta=kernel_meta,
        rng=rng,
        batch_size=batch_size,
    )

    return output_path if output_path is not None else Path("pkl_files") / f"Resq_i_{date}.npz"


# Stable reference to the original generate_kernel for CLI introspection.
# `cli_main` uses this (not the module-level `generate_kernel` name) when
# validating TOML keys, so that test monkeypatches replacing the public
# symbol don't shrink the inspected signature to the fake's `**kwargs`.
_generate_kernel_original = generate_kernel


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-bootstrap`.

    Reads a TOML config (e.g. `configs/default.toml`), parses the
    `[reciprocal]` block, and writes the resulting reciprocal-space kernel
    npz to `<fm.pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV}keV_<date>.npz`,
    or to `--output <path>` if given.
    """
    import tomllib

    import dfxm_geo.direct_space.forward_model as fm

    parser = argparse.ArgumentParser(
        prog="dfxm-bootstrap",
        description=(
            "Generate the reciprocal-space resolution kernel npz for "
            "dfxm-forward / dfxm-identify. Takes ~50 s wall-clock at the "
            "default Nrays=1e8."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config containing a [reciprocal] block.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination kernel npz path. Defaults to "
            "<pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV}keV_<date>.npz "
            "(discovered at runtime by _lookup_kernel_path)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing kernel npz at the destination.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help=(
            "Skip silently (exit 0) when the destination kernel npz already exists, "
            "instead of returning the usual 'refusing to overwrite' error. "
            "Useful as an idempotent guard in cluster batch templates that don't "
            "want to hardcode the kernel filename."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed for the Monte Carlo ray sampling. Makes the kernel "
            "bit-reproducible across runs. Overrides any `seed` set in the "
            "[reciprocal] TOML block. Omit for an entropy-seeded (random) run."
        ),
    )
    args = parser.parse_args(argv)

    if args.force and args.if_missing:
        print(
            "error: --force and --if-missing are mutually exclusive.",
            file=sys.stderr,
        )
        return 1

    if not args.config.is_file():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 1

    with args.config.open("rb") as f:
        data = tomllib.load(f)
    if "reciprocal" not in data:
        print(
            f"error: {args.config} has no [reciprocal] block; "
            "see configs/default.toml for the expected schema.",
            file=sys.stderr,
        )
        return 1

    import inspect

    # Bind to the original `generate_kernel` (not the module-level name) so
    # that test monkeypatches replacing kmod.generate_kernel don't shrink the
    # valid-key set to just the fake's **kwargs.
    sig = inspect.signature(_generate_kernel_original)
    valid_params = set(sig.parameters)
    # `date` and `output_path` are CLI-managed kwargs, not TOML-driven.
    # `hkl` and `keV` are cli_main-scope reflection inputs, not
    # `generate_kernel` kwargs — added explicitly to the allow-list.
    # `mode`, `eta`, `mount`, `omega` come from [geometry]/[crystal] blocks,
    # not from [reciprocal] — exclude them from the [reciprocal] key check.
    cli_managed = {"date", "output_path", "mode", "eta", "mount", "omega"}
    valid_recip_keys = (valid_params - cli_managed) | {"hkl", "keV"}
    unknown = set(data["reciprocal"]) - valid_recip_keys
    if unknown:
        print(
            f"error: unknown [reciprocal] keys: {sorted(unknown)}; "
            f"valid keys are {sorted(valid_recip_keys)}.",
            file=sys.stderr,
        )
        return 1

    # Pop hkl/keV — they are cli_main-scope, not generate_kernel kwargs.
    reciprocal_kwargs = dict(data["reciprocal"])
    raw_hkl = reciprocal_kwargs.pop("hkl", None)
    raw_keV = reciprocal_kwargs.pop("keV", None)

    # Detect multi-reflection mode: [[reflections]] or [reflections_auto] present.
    # In this case [reciprocal] carries only instrument params (no hkl); hkl comes
    # from each [[reflections]] entry instead. Skip the single-reflection hkl/keV
    # parity check — the multi branch handles geometry resolution itself.
    multi = ("reflections" in data) or ("reflections_auto" in data)

    if not multi:
        if (raw_hkl is None) != (raw_keV is None):
            print(
                "error: must provide both `hkl` and `keV`, or neither.",
                file=sys.stderr,
            )
            return 1

        if (raw_hkl is not None or raw_keV is not None) and "theta" in reciprocal_kwargs:
            print(
                "error: cannot specify both `theta` and `hkl`+`keV`; pick one.",
                file=sys.stderr,
            )
            return 1

    # Parse [crystal] and [geometry] blocks.
    mount_block = data.get("crystal")
    geometry_block = data.get("geometry")

    # The crystal MOUNT (lattice/a/mount_x/y/z) is only used geometrically by
    # oblique mode (theta/omega via the solver); simplified mode needs only the
    # cubic lattice parameter `a`. A mount-style [crystal] carries the mount keys;
    # a forward-style [crystal] (dislocation layout: mode + sub-block) does NOT
    # and is irrelevant to the kernel. Distinguish them so
    # `dfxm-bootstrap --config configs/default.toml` (whose [crystal] is a
    # forward layout) works, while still requiring an explicit [geometry] mode
    # when a genuine mount is supplied (the spec-§6 explicitness guard, now
    # correctly scoped so it no longer over-fires on a forward layout).
    _MOUNT_KEYS = (
        "lattice",
        "a",
        "b",
        "c",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "mount_x",
        "mount_y",
        "mount_z",
    )
    is_mount_block = mount_block is not None and any(k in mount_block for k in _MOUNT_KEYS)
    if is_mount_block and geometry_block is None:
        print(
            "error: a [crystal] mount block requires [geometry] mode to be set "
            "explicitly (mode='simplified' or mode='oblique').",
            file=sys.stderr,
        )
        return 1

    try:
        # Multi-reflection path: eta may be absent (resolver picks it per-entry)
        # or present as a default for the resolver. Pass allow_missing_eta=True
        # so a missing eta returns ("oblique", nan) rather than raising.
        mode, config_eta = _parse_geometry_block(geometry_block, allow_missing_eta=multi)
        # Ignore a forward-layout [crystal] (use the default Al mount); only a
        # genuine mount block feeds the lattice parameter / orientation.
        mount = _crystal_mount_from_toml(mount_block if is_mount_block else None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # -------------------------------------------------------------------------
    # Multi-reflection branch: [[reflections]] / [reflections_auto]
    # -------------------------------------------------------------------------
    if multi:
        return _bootstrap_multi_reflection(
            args=args,
            data=data,
            mount=mount,
            mode=mode,
            config_eta=config_eta,
            raw_keV=raw_keV,
            reciprocal_kwargs=reciprocal_kwargs,
            pkl_fpath=fm.pkl_fpath,
        )

    cell = mount.cell
    if raw_hkl is not None and raw_keV is not None:
        try:
            hkl_tuple: tuple[int, int, int] = tuple(raw_hkl)
            theta = _validate_reflection(hkl_tuple, float(raw_keV), cell)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        keV_for_filename: float = float(raw_keV)
    else:
        print(
            "warning: [reciprocal] has no `hkl`/`keV`; defaulting to Al (-1, 1, -1) @ 17 keV.",
            file=sys.stderr,
        )
        hkl_tuple = (-1, 1, -1)
        keV_for_filename = 17.0
        theta = _default_theta_al_111(17)

    # Inject the computed theta so generate_kernel uses our value, not its
    # module-load default. (Skip if the TOML already set theta — that path
    # was rejected above when hkl/keV also present, so it only fires when
    # neither hkl/keV were given AND theta was.)
    if "theta" not in reciprocal_kwargs:
        reciprocal_kwargs["theta"] = theta

    # Oblique mode: cross-check eta, override theta, and record omega.
    omega_for_meta = 0.0
    if mode == "oblique":
        try:
            theta_validated, omega_validated = _validate_eta_against_compute_omega_eta(
                mount, hkl_tuple, keV_for_filename, config_eta
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        reciprocal_kwargs["theta"] = theta_validated
        omega_for_meta = omega_validated
        theta = theta_validated

    # Echo computed θ for sanity (Q4).
    theta_deg = float(np.degrees(theta))
    print(f"reflection: hkl={hkl_tuple}, keV={keV_for_filename:g} -> theta = {theta_deg:.4f} deg")

    # Build output path with the per-mode filename selector.
    if args.output is not None:
        output_path = args.output
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(fm.pkl_fpath) / _build_kernel_filename(
            mode=mode,
            hkl=hkl_tuple,
            keV=keV_for_filename,
            theta=reciprocal_kwargs.get("theta", 0.0),
            eta=config_eta,
            date=date,
        )

    if output_path.exists():
        if args.if_missing:
            print(f"kernel already present at {output_path}; skipping.")
            return 0
        if not args.force:
            print(
                f"refusing to overwrite existing kernel npz at {output_path}; "
                f"pass --force to regenerate.",
                file=sys.stderr,
            )
            return 1

    reciprocal_kwargs["hkl"] = hkl_tuple
    reciprocal_kwargs["keV"] = keV_for_filename
    # CLI --seed overrides any `seed` set in the TOML [reciprocal] block.
    if args.seed is not None:
        reciprocal_kwargs["seed"] = args.seed
    written = generate_kernel(
        output_path=output_path,
        mode=mode,
        eta=config_eta,
        mount=mount,
        omega=omega_for_meta,
        **reciprocal_kwargs,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(cli_main())
