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

from datetime import datetime
from pathlib import Path

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def _default_theta_al_111(keV: float = 17) -> float:
    """Bragg angle for Al 111 at the given beam energy (default 17 keV)."""
    a = 4.0495e-10  # Al lattice parameter, m
    d_111 = a / np.sqrt(3)
    wavelength = 1.239841984e-9 / keV
    return float(np.arcsin(wavelength / (2 * d_111)))


def _validate_reflection(
    hkl: tuple[int, int, int],
    keV: float,
    a: float,
) -> float:
    """Compute and validate the Bragg angle θ for an arbitrary cubic reflection.

    Args:
        hkl: Miller indices (must be ints, length 3, not all zero).
        keV: beam energy in keV (must be > 0).
        a: cubic lattice parameter in metres.

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
    if a <= 0:
        raise ValueError(f"lattice parameter `a` must be > 0, got {a}.")

    h, k, l = hkl
    d_hkl = a / np.sqrt(h * h + k * k + l * l)
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
    hkl: tuple[int, int, int],
    keV: float,
    date: str,
) -> str:
    """Per-reflection kernel npz basename: `Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz`.

    `:g` formatting drops trailing zeros on keV (17.0 → "17", 17.5 → "17.5").
    Negative hkl components render naturally (-1 → "h-1").
    """
    h, k, l = hkl
    return f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz"


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

    Returns:
        The path the npz was written to.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    if output_path is not None:
        output_path = Path(output_path)

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
        output_path=output_path,
        kernel_meta=kernel_meta,
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
    npz to the canonical path that `dfxm-forward`'s stage-0 preflight will
    read (`<fm.pkl_fpath>/<fm.pkl_fn>`), or to `--output <path>` if given.
    """
    import argparse
    import sys
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
            "Destination kernel npz path. Defaults to <pkl_fpath>/<pkl_fn> "
            "(the path dfxm-forward reads at import time)."
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
    valid_recip_keys = (valid_params - {"date", "output_path"}) | {"hkl", "keV"}
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

    a_lattice = 4.0495e-10  # Al lattice parameter, m
    if raw_hkl is not None and raw_keV is not None:
        try:
            hkl_tuple: tuple[int, int, int] = tuple(raw_hkl)
            theta = _validate_reflection(hkl_tuple, float(raw_keV), a_lattice)
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

    # Echo computed θ for sanity (Q4).
    theta_deg = float(np.degrees(theta))
    print(f"reflection: hkl={hkl_tuple}, keV={keV_for_filename:g} -> theta = {theta_deg:.4f} deg")

    # Build output path.
    if args.output is not None:
        output_path = args.output
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(fm.pkl_fpath) / _build_kernel_filename(hkl_tuple, keV_for_filename, date)

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
    written = generate_kernel(output_path=output_path, **reciprocal_kwargs)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(cli_main())
