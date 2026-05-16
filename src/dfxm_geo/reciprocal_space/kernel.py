"""Driver: generates a reciprocal-space resolution kernel pickle.

Configures the Monte Carlo integration parameters and calls
:func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`. Side
effects: writes ``pkl_files/Resq_i_<timestamp>.pkl`` and
``pkl_files/Resq_i_<timestamp>_vars.txt`` to the current working directory.

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
) -> Path:
    """Run the kernel-generation Monte Carlo and write the pickle to ``pkl_files/``.

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

    Returns:
        The path the pickle was written to.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    # Resolve the destination. When `output_path` is provided, it is the
    # canonical path; the sidecar lives next to it as `<stem>_vars.txt`.
    # When `output_path` is None, fall back to the legacy default
    # `pkl_files/Resq_i_<date>.pkl` (relative to CWD) — preserves the
    # `python -m dfxm_geo.reciprocal_space.kernel` workflow.
    if output_path is not None:
        output_path = Path(output_path)
        vars_path = output_path.with_name(output_path.stem + "_vars.txt")
    else:
        vars_path = Path("pkl_files") / f"Resq_i_{date}_vars.txt"

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
    )

    vars_used = {
        "Nrays": Nrays,
        "npoints1": npoints1,
        "npoints2": npoints2,
        "npoints3": npoints3,
        "qi1_range": qi1_range,
        "qi2_range": qi2_range,
        "qi3_range": qi3_range,
        "zeta_v_fwhm": zeta_v_fwhm,
        "zeta_h_fwhm": zeta_h_fwhm,
        "NA_rms": NA_rms,
        "eps_rms": eps_rms,
        "theta": theta,
        "D": D,
        "d1": d1,
        "phys_aper": phys_aper,
        "beamstop": beamstop,
        "bs_height": bs_height,
        "aperture": aperture,
        "knife_edge": knife_edge,
        "dphi_range": dphi_range,
    }

    vars_path.parent.mkdir(parents=True, exist_ok=True)
    vars_path.write_text(str(vars_used), encoding="utf-8")

    return output_path if output_path is not None else Path("pkl_files") / f"Resq_i_{date}.pkl"


# Stable reference to the original generate_kernel for CLI introspection.
# `cli_main` uses this (not the module-level `generate_kernel` name) when
# validating TOML keys, so that test monkeypatches replacing the public
# symbol don't shrink the inspected signature to the fake's `**kwargs`.
_generate_kernel_original = generate_kernel


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-bootstrap`.

    Reads a TOML config (e.g. `configs/default.toml`), parses the
    `[reciprocal]` block, and writes the resulting reciprocal-space kernel
    pickle to the canonical path that `dfxm-forward`'s stage-0 preflight will
    read (`<fm.pkl_fpath>/<fm.pkl_fn>`), or to `--output <path>` if given.
    """
    import argparse
    import sys
    import tomllib

    import dfxm_geo.direct_space.forward_model as fm

    parser = argparse.ArgumentParser(
        prog="dfxm-bootstrap",
        description=(
            "Generate the reciprocal-space resolution kernel pickle for "
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
            "Destination pickle path. Defaults to <pkl_fpath>/<pkl_fn> "
            "(the path dfxm-forward reads at import time)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pickle at the destination.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help=(
            "Skip silently (exit 0) when the destination pickle already exists, "
            "instead of returning the usual 'refusing to overwrite' error. "
            "Useful as an idempotent guard in cluster batch templates that don't "
            "want to hardcode the pickle filename."
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
    valid_recip_keys = valid_params - {"date", "output_path"}
    unknown = set(data["reciprocal"]) - valid_recip_keys
    if unknown:
        print(
            f"error: unknown [reciprocal] keys: {sorted(unknown)}; "
            f"valid keys are {sorted(valid_recip_keys)}.",
            file=sys.stderr,
        )
        return 1

    output_path = args.output if args.output is not None else Path(fm.pkl_fpath) / fm.pkl_fn

    if output_path.exists():
        if args.if_missing:
            print(f"kernel already present at {output_path}; skipping.")
            return 0
        if not args.force:
            print(
                f"refusing to overwrite existing pickle at {output_path}; pass --force to regenerate.",
                file=sys.stderr,
            )
            return 1

    kwargs = dict(data["reciprocal"])
    written = generate_kernel(output_path=output_path, **kwargs)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(cli_main())
