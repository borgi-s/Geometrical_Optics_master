"""Regression tests for the two GNB placement bugs (2026-06-20 followup).

Bug 1 — co-directional sets (same in-plane line-spacing direction u = n x xi and
equal density) were laid at IDENTICAL positions, fusing two distinct <110>
dislocations into a non-physical super-dislocation.  The fix interleaves
co-directional sets by a fractional offset (k + j/m) * d * u.

Bug 2 — the gnb branch placed cores with crystal_to_lab = Us, but the strain
field of every dislocation is rendered through the full chain that includes
Theta = R_y(theta_Bragg).  The self-consistent placement is Theta.T @ Us so the
cores live in the SAME lab frame as their fields.  (See
docs/superpowers/notes/2026-06-20-gnb-followup-bugs-and-final-task.md.)
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell

_CELL = UnitCell.cubic(4.05e-10)


def _line_dirs(pop) -> np.ndarray:
    """Per-dislocation geometric line direction (unit), in the population frame.

    line = cos(alpha) * b_hat + sin(alpha) * t_hat, alpha = 90 - rotation_deg,
    t_hat = Ud[:, 2] (the post-flip edge reference).  Verified equal to xi in the
    spike findings (section 3).
    """
    out = []
    for i in range(pop.Ud.shape[0]):
        b_hat = pop.Ud[i, :, 0]
        t_hat = pop.Ud[i, :, 2]
        a = np.deg2rad(90.0 - pop.rotation_deg[i])
        line = np.cos(a) * b_hat + np.sin(a) * t_hat
        out.append(line / np.linalg.norm(line))
    return np.asarray(out)


def test_no_two_dislocations_share_a_line():
    """No two distinct dislocations may occupy the same position AND line direction.

    Pre-fix this failed for leds_eq14 (set0 & set1 share xi=[1,0,-1] and were
    placed identically) and frankus (set0 & set1).  leds_eq11's two sets have
    different line directions so they never coincided.
    """
    for name in ("leds_eq11", "leds_eq14", "frankus"):
        pop = fw.build_wall_population(
            fw.RECIPES[name],
            theta_deg=0.05,
            extent_um=5.0,
            cell=_CELL,
            ny=0.334,
            crystal_to_lab=np.eye(3),
        )
        P = pop.positions_um
        L = _line_dirs(pop)
        n = P.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                same_line = abs(abs(float(L[i] @ L[j])) - 1.0) < 1e-6
                if same_line:
                    assert float(np.linalg.norm(P[i] - P[j])) > 1e-6, (
                        f"{name}: dislocations {i},{j} share both a line direction "
                        f"and a position (non-physical super-dislocation)"
                    )


def test_codirectional_sets_interleave_by_half_spacing():
    """leds_eq14 set0 & set1 (shared u, equal density) interleave at half-spacing.

    Both sets have line direction xi=[1,0,-1], hence the same spacing direction
    u = n x xi and (equal density) the same comb spacing d.  After the fix set1 is
    offset by d/2 from set0, so the merged comb's adjacent-line gaps are all d/2.
    Pre-fix the two combs coincide -> zero gaps.
    """
    rec = fw.RECIPES["leds_eq14"]
    pop = fw.build_wall_population(
        rec,
        theta_deg=0.05,
        extent_um=5.0,
        cell=_CELL,
        ny=0.334,
        crystal_to_lab=np.eye(3),
    )
    n_hat = fw._unit(fw._cartesian(rec.n, _CELL))
    u = fw._unit(np.cross(n_hat, fw._unit(fw._cartesian((1, 0, -1), _CELL))))
    rho_hat, _ = fw.solve_density_scale(rec, 0.05, _CELL)
    d_um = (1.0 / rho_hat[0]) * 1e6

    def proj_for_burgers(b_int) -> np.ndarray:
        b_hat = fw._unit(fw._cartesian(b_int, _CELL))
        mask = np.array(
            [np.allclose(pop.Ud[i, :, 0], b_hat, atol=1e-6) for i in range(pop.Ud.shape[0])]
        )
        return np.sort(pop.positions_um[mask] @ u)

    p0 = proj_for_burgers((1, 0, -1))  # set0
    p1 = proj_for_burgers((0, 1, -1))  # set1
    assert len(p0) > 1 and len(p1) > 1
    # each set's own comb spacing is d
    assert np.allclose(np.diff(p0), d_um, rtol=1e-6)
    assert np.allclose(np.diff(p1), d_um, rtol=1e-6)
    # merged comb: adjacent gaps all ~ d/2 (interleaved), and strictly positive
    gaps = np.diff(np.sort(np.concatenate([p0, p1])))
    assert float(np.min(gaps)) > 1e-9, "co-directional sets still coincide"
    assert np.isclose(float(np.median(gaps)), d_um / 2.0, rtol=0.1), (
        f"expected interleave at d/2={d_um / 2:.4g}, got median gap {np.median(gaps):.4g}"
    )


def _ry(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_gnb_population_placed_in_field_frame():
    """Bug 2: gnb cores must be placed with crystal_to_lab = Theta.T @ Us.

    The strain field of every dislocation is rendered through the chain
    rd = Ud.T @ Us.T @ S.T @ Theta @ (rl - offset), so the self-consistent lab
    frame for the position offsets is Theta.T @ Us (S = I, cubic mount). Placing
    with Us alone (the spike's simplified-geometry assumption) leaves the cores in
    a frame rotated by ~theta_Bragg from their fields.

    build_dislocation_population takes the run's Theta; the boundary-plane normal
    the positions occupy must therefore be Theta.T @ Us @ n_hat, NOT Us @ n_hat.
    """
    from dfxm_geo.config import CrystalConfig
    from dfxm_geo.direct_space.forward_model import Us, build_dislocation_population

    theta_b = np.deg2rad(8.972)  # (-1,1,-1) Al @ 17 keV, the gnb e2e reflection
    Theta = _ry(theta_b)
    crystal = CrystalConfig.from_dict(
        {"mode": "gnb", "gnb": {"recipe": "leds_eq11", "theta_deg": 0.05, "extent_um": 12.0}}
    )
    pop = build_dislocation_population(
        crystal, fov_lateral_um=50.0, rng=None, mount=None, theta=Theta
    )
    n_hat = fw._unit(fw._cartesian((1, 1, 1), _CELL))  # leds_eq11 boundary normal
    n_lab_field = fw._unit(Theta.T @ Us @ n_hat)  # the frame the field lives in
    n_lab_us_only = fw._unit(Us @ n_hat)  # the buggy (Theta-less) frame

    in_field_plane = float(np.max(np.abs(pop.positions_um @ n_lab_field)))
    in_us_plane = float(np.max(np.abs(pop.positions_um @ n_lab_us_only)))
    assert in_field_plane < 1e-6, (
        f"cores not in the field-frame boundary plane (max |pos.n| = {in_field_plane:.3g})"
    )
    assert in_us_plane > 1e-2, (
        f"cores still placed in the Theta-less Us frame (max |pos.n| = {in_us_plane:.3g}); "
        f"Theta was not applied"
    )
