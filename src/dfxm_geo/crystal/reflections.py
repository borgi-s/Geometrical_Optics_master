"""Resolve [[reflections]] / [reflections_auto] TOML tables into ReflectionRun records.

Pure logic over crystal.oblique — no I/O, no pipeline imports. See
docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta, find_reflections

# Matches the Phase-A bootstrap tolerance (kernel._validate_eta_against_compute_omega_eta):
# loose enough for paper-quoting precision, strict enough to catch typos.
ETA_MATCH_TOL = 1e-3
# Two solver outputs for a genuinely shared geometry agree to ~1e-12; 1e-6 rad
# groups them while keeping physically distinct geometries apart.
GROUP_TOL = 1e-6


@dataclass(frozen=True)
class ReflectionRun:
    """One fully resolved reflection of a multi-reflection run. Angles in radians."""

    hkl: tuple[int, int, int]
    keV: float
    eta: float
    theta: float
    omega: float
    group: int  # kernel-sharing group: same (theta, eta) within GROUP_TOL


def _resolve_entry(
    entry: dict,
    mount: CrystalMount,
    keV: float,
    default_eta: float | None,
) -> tuple[tuple[int, int, int], float, float, float]:
    """Resolve one [[reflections]] entry → (hkl, eta, theta, omega)."""
    if "hkl" not in entry:
        raise ValueError(f"[[reflections]] entry missing 'hkl': {entry!r}.")
    raw_hkl = entry["hkl"]
    if len(raw_hkl) != 3:
        raise ValueError(f"[[reflections]] hkl must have 3 components, got {raw_hkl!r}.")
    if any(int(c) != c for c in raw_hkl):
        raise ValueError(f"[[reflections]] hkl components must be integers, got {raw_hkl!r}.")
    h, k, l = (int(c) for c in raw_hkl)
    hkl: tuple[int, int, int] = (h, k, l)

    geom = compute_omega_eta(mount, hkl, keV)
    if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
        raise ValueError(
            f"Laue condition unsatisfiable for hkl={hkl} at keV={keV} with this mount. "
            "Use 'dfxm-find-reflections' to enumerate reachable reflections."
        )
    solutions = {
        1: (geom.eta_1, geom.theta_1, geom.omega_1),
        2: (geom.eta_2, geom.theta_2, geom.omega_2),
    }

    if "eta" in entry and "omega_solution" in entry:
        raise ValueError(
            f"[[reflections]] entry for hkl={hkl}: give 'eta' or 'omega_solution', "
            "not both (eta already selects the solution branch)."
        )
    eta_requested = entry.get("eta", None if "omega_solution" in entry else default_eta)
    if eta_requested is not None:
        eta_requested = float(eta_requested)
        for eta_i, theta_i, omega_i in solutions.values():
            if not np.isnan(eta_i) and abs(eta_i - eta_requested) <= ETA_MATCH_TOL:
                return hkl, float(eta_i), float(theta_i), float(omega_i)
        raise ValueError(
            f"eta={eta_requested:.6f} rad does not match either solution for hkl={hkl} "
            f"(η₁={geom.eta_1:.6f}, η₂={geom.eta_2:.6f}). No auto-correct; use "
            "'dfxm-find-reflections' to list valid (η, ω) pairs."
        )

    sol = int(entry.get("omega_solution", 1))
    if sol not in (1, 2):
        raise ValueError(f"omega_solution must be 1 or 2, got {sol} (hkl={hkl}).")
    eta_i, theta_i, omega_i = solutions[sol]
    if np.isnan(omega_i):
        raise ValueError(
            f"omega_solution={sol} does not exist for hkl={hkl} at keV={keV} "
            "(no real Laue branch). Use 'dfxm-find-reflections'."
        )
    return hkl, float(eta_i), float(theta_i), float(omega_i)


def _assign_groups(resolved: list[tuple[tuple[int, int, int], float, float, float]]) -> list[int]:
    """Dense group ids by (theta, eta) proximity, ordered by first appearance."""
    reps: list[tuple[float, float]] = []  # (theta, eta) representative per group
    groups: list[int] = []
    for _hkl, eta, theta, _omega in resolved:
        for gid, (t_rep, e_rep) in enumerate(reps):
            if abs(theta - t_rep) <= GROUP_TOL and abs(eta - e_rep) <= GROUP_TOL:
                groups.append(gid)
                break
        else:
            reps.append((theta, eta))
            groups.append(len(reps) - 1)
    return groups


def resolve_reflections(
    entries: list[dict],
    mount: CrystalMount,
    keV: float,
    *,
    default_eta: float | None = None,
) -> list[ReflectionRun]:
    """Resolve explicit [[reflections]] entries. See spec §5 for the rules."""
    if not entries:
        raise ValueError("[[reflections]] must contain at least one entry.")
    resolved = [_resolve_entry(e, mount, keV, default_eta) for e in entries]
    groups = _assign_groups(resolved)
    return [
        ReflectionRun(hkl=hkl, keV=float(keV), eta=eta, theta=theta, omega=omega, group=gid)
        for (hkl, eta, theta, omega), gid in zip(resolved, groups, strict=True)
    ]


def resolve_reflections_auto(
    auto: dict,
    mount: CrystalMount,
    keV: float,
) -> list[ReflectionRun]:
    """Expand [reflections_auto] via find_reflections.

    Typically one group when eta_target identifies a unique (theta, eta) geometry;
    multiple groups are returned if the target matches reflections at different Bragg angles.
    """
    if "eta_target" not in auto:
        raise ValueError("[reflections_auto] requires 'eta_target' (radians).")
    eta_target = float(auto["eta_target"])
    kwargs: dict = {"eta_target": eta_target, "eta_tol": ETA_MATCH_TOL}
    if "theta_max" in auto:
        kwargs["theta_range"] = (0.0, float(auto["theta_max"]))
    if "hkl_max" in auto:
        kwargs["hkl_max"] = int(auto["hkl_max"])
    geoms = find_reflections(mount, keV, **kwargs)
    if not geoms:
        raise ValueError(
            f"[reflections_auto] eta_target={eta_target:.6f} rad matched no reflections. "
            "Run 'dfxm-find-reflections' to inspect the accessible groups."
        )
    entries = []
    for g in geoms:
        # pick the solution branch whose eta matched the target
        sol = 1 if (not np.isnan(g.eta_1) and abs(g.eta_1 - eta_target) <= ETA_MATCH_TOL) else 2
        entries.append({"hkl": list(g.hkl), "omega_solution": sol})
    return resolve_reflections(entries, mount, keV)
