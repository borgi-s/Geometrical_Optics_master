"""dfxm-find-reflections: enumerate accessible reflections for a crystal mount.

Reproduces the paper's Table A.2 (Detlefs et al. 2025 / arXiv:2503.22022
Appendix A) for the mount + keV given in a config TOML. Wires the
(Phase-A-tested) crystal.oblique.find_reflections solver to the command line.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

from dfxm_geo.crystal.oblique import find_reflections
from dfxm_geo.crystal.reflections import ETA_MATCH_TOL, GROUP_TOL
from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dfxm-find-reflections",
        description="Enumerate Laue-accessible reflections (theta, eta, omega) for "
        "the crystal mount and beam energy in a config TOML.",
        epilog="The 'group' column assigns a kernel-sharing ID based on each row's "
        "solution-1 (theta, eta); +/- eta branches may consolidate further when "
        "branches are chosen explicitly in [[reflections]].",
    )
    parser.add_argument(
        "--config", required=True, help="TOML with [crystal] mount + [reciprocal] keV"
    )
    parser.add_argument("--keV", type=float, default=None, help="override [reciprocal] keV")
    parser.add_argument("--hkl-max", type=int, default=5, help="max |Miller index| (default 5)")
    parser.add_argument(
        "--theta-max-deg", type=float, default=16.25, help="max Bragg angle (default 16.25)"
    )
    parser.add_argument(
        "--eta-target-deg", type=float, default=None, help="keep only the group at this eta"
    )
    parser.add_argument(
        "--eta-tol-deg",
        type=float,
        default=float(np.degrees(ETA_MATCH_TOL)),
        help="eta-target tolerance in degrees (default: matches the [[reflections]] resolver tolerance)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        print(f"error: invalid TOML in {config_path}: {exc}", file=sys.stderr)
        return 1

    mount = _crystal_mount_from_toml(raw.get("crystal"))

    # Resolve keV: CLI flag > config value > hardcoded fallback (with warning).
    if args.keV is not None:
        keV = args.keV
    elif raw.get("reciprocal", {}).get("keV") is not None:
        keV = float(raw["reciprocal"]["keV"])
    else:
        print(
            "warning: no [reciprocal] keV in config and no --keV given; defaulting to 17.0 keV.",
            file=sys.stderr,
        )
        keV = 17.0

    kwargs: dict = {
        "theta_range": (0.0, float(np.deg2rad(args.theta_max_deg))),
        "hkl_max": args.hkl_max,
    }
    if args.eta_target_deg is not None:
        kwargs["eta_target"] = float(np.deg2rad(args.eta_target_deg))
        kwargs["eta_tol"] = float(np.deg2rad(args.eta_tol_deg))
    geoms = find_reflections(mount, keV, **kwargs)

    print(
        f"# mount: x={mount.mount_x} y={mount.mount_y} z={mount.mount_z}  a={mount.a:g} m  keV={keV:g}"
    )
    print(
        f"{'hkl':>10} {'theta_deg':>10} {'eta1_deg':>10} {'omega1_deg':>11} {'eta2_deg':>10} {'omega2_deg':>11} {'group':>6}"
    )
    print(
        "# group: kernel-sharing by solution-1 (theta, eta); +/- eta branches may consolidate when branches are chosen explicitly"
    )
    reps: list[tuple[float, float]] = []
    for g in geoms:
        theta = g.theta_1 if not np.isnan(g.theta_1) else g.theta_2
        eta = g.eta_1 if not np.isnan(g.eta_1) else g.eta_2
        gid = -1
        for i, (t_rep, e_rep) in enumerate(reps):
            if abs(theta - t_rep) <= GROUP_TOL and abs(eta - e_rep) <= GROUP_TOL:
                gid = i
                break
        if gid < 0:
            reps.append((float(theta), float(eta)))
            gid = len(reps) - 1
        hkl_str = " ".join(str(c) for c in g.hkl)
        print(
            f"{hkl_str:>10} {np.degrees(theta):>10.3f} "
            f"{np.degrees(g.eta_1):>10.3f} {np.degrees(g.omega_1):>11.3f} "
            f"{np.degrees(g.eta_2):>10.3f} {np.degrees(g.omega_2):>11.3f} {gid:>6}"
        )
    print(f"# {len(geoms)} reflections, {len(reps)} kernel group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
