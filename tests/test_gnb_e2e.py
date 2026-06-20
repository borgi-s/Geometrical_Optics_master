"""GNB (geometrically-necessary boundary) wall mode: wiring + frame + e2e.

These tests prove the ``crystal.mode == "gnb"`` branch of
``build_dislocation_population`` composes end-to-end:

  * the discriminating frame test (``test_placement_us_makes_field_lines_coherent``)
    pins the crystal->lab placement to the module-global ``Us`` (sample->grain
    rotation). This is the ONLY orthogonal map that keeps the dislocation FIELD
    lines (oriented by the fixed ``Us``: ``line_lab = Us @ xi_hat``) coplanar with
    the boundary plane the POSITIONS occupy (Task-1 spike). ``eye`` and ``Us.T``
    make the lines pierce the boundary by 18-60 deg (an incoherent wall). A
    positions-only assertion is NOT sufficient: it passes for any orthogonal R.

  * the analytic+oblique forward render (``test_gnb_forward_renders_finite``)
    runs the whole pipeline kernel-free (mirrors ``tests/test_bcc_e2e.py``).

  * the simplified-mode builder test (``test_gnb_simplified_builds_population``)
    exercises the ``mount is None`` cell-synthesis path without an MC kernel.

Kernel-free trick: analytic resolution backend + oblique FCC Al mount, exactly
as the BCC/HCP e2e tests use (no bootstrapped MC kernel needed). The η is
computed by ``compute_omega_eta`` and fed back to the config (the oblique
validator requires the exact η).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.pipeline import SimulationConfig, run_simulation

_AL_A = 4.05e-10  # Al lattice param, METRES
_HKL = (-1, 1, -1)
_KEV = 17.0
CUBIC = UnitCell.cubic(_AL_A)


def _eta(hkl=_HKL) -> float:
    """Valid oblique eta for ``hkl`` at 17 keV with the cubic identity FCC mount.

    Mirrors ``tests/test_hcp_e2e.py``: ``compute_omega_eta`` returns a
    ``ReflectionGeometry`` with ``eta_1``/``eta_2``; pick the finite branch.
    """
    mount = CrystalMount(
        lattice="cubic",
        a=_AL_A,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
        structure_type="fcc",
        material="Al",
    )
    geom = compute_omega_eta(mount, hkl, _KEV)
    eta = geom.eta_1 if not np.isnan(geom.eta_1) else geom.eta_2
    assert not np.isnan(eta), f"({hkl}) unreachable at {_KEV} keV"
    return float(eta)


def _gnb_toml(recipe="leds_eq11", hkl=_HKL, theta_deg=0.05, extent_um=12.0) -> str:
    eta = _eta(hkl)
    return f"""
[reciprocal]
hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]
keV = {_KEV}
backend = "analytic"
beamstop = false

[geometry]
mode = "oblique"
eta = {eta!r}

[crystal]
lattice = "cubic"
a = {_AL_A!r}
structure_type = "fcc"
material = "Al"
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "gnb"

[crystal.gnb]
recipe = "{recipe}"
theta_deg = {theta_deg}
extent_um = {extent_um}

[scan.phi]
value = 0.0

[io]
include_perfect_crystal = false
write_strain_provenance = false

[postprocess]
enabled = false
"""


def _run(base: Path, body: str) -> np.ndarray:
    base.mkdir(parents=True, exist_ok=True)
    (base / "gnb.toml").write_text(body, encoding="utf-8")
    cfg = SimulationConfig.from_toml(base / "gnb.toml")
    out = base / "out"
    run_simulation(cfg, out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        return f["/entry_0000/dfxm_sim_detector/image"][...]


def test_placement_us_makes_field_lines_coherent():
    # The field line orientation is ALWAYS Us @ xi_hat (the field is oriented by the
    # fixed Us). For a coherent wall the field lines must be perpendicular to the lab
    # boundary-plane normal the POSITIONS occupy (R @ n_hat). This holds ONLY for R = Us;
    # eye and Us.T make the lines pierce the boundary (Task-1 spike). Positions-only is
    # NOT sufficient (it passes for any orthogonal R).
    from dfxm_geo.direct_space.forward_model import Us

    r = fw.RECIPES["leds_eq11"]
    n_hat = fw._unit(fw._cartesian(r.n, CUBIC))
    field_lines = [fw._unit(Us @ fw._unit(fw._cartesian(s.xi, CUBIC))) for s in r.sets]

    def max_pierce(R):
        n_lab = fw._unit(R @ n_hat)
        return max(abs(float(L @ n_lab)) for L in field_lines)

    assert max_pierce(Us) < 1e-9  # correct placement: lines in-plane
    assert max_pierce(np.eye(3)) > 1e-2  # eye is wrong (spike: ~0.35)
    assert max_pierce(Us.T) > 1e-2  # Us.T is wrong (spike: ~0.88)
    # and the builder's positions lie in the boundary plane under the Us placement:
    pop = fw.build_wall_population(
        r, theta_deg=0.05, extent_um=12.0, cell=CUBIC, ny=0.334, crystal_to_lab=Us
    )
    n_lab = fw._unit(Us @ n_hat)
    assert np.max(np.abs(pop.positions_um @ n_lab)) < 1e-6


def test_gnb_forward_renders_finite(tmp_path):
    img = _run(tmp_path / "eq11", _gnb_toml("leds_eq11"))
    assert img.ndim == 3 and img.shape[0] == 1
    assert np.isfinite(img).all()
    assert float(img.std()) > 0.0  # non-degenerate


def test_gnb_simplified_builds_population_without_mount():
    from dfxm_geo.config import CrystalConfig
    from dfxm_geo.direct_space.forward_model import build_dislocation_population

    crystal = CrystalConfig.from_dict(
        {"mode": "gnb", "gnb": {"recipe": "leds_eq11", "theta_deg": 0.05, "extent_um": 12.0}}
    )
    # build_dislocation_population(crystal, fov_lateral_um, rng, *, mount=None)
    pop = build_dislocation_population(crystal, fov_lateral_um=50.0, rng=None, mount=None)
    assert pop.positions_um.shape[0] > 0
    assert pop.Ud.shape[0] == pop.positions_um.shape[0]
