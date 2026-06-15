"""Regression test: a g·b = 0 screw dislocation is exactly invisible.

In isotropic elasticity a pure screw whose Burgers vector is perpendicular to
the diffraction vector (g·b = 0) produces NO change in the local reciprocal
lattice vector and is therefore exactly invisible — independent of the slip
plane (Hirth & Lothe, *Theory of Dislocations*; Howie & Whelan 1961). The DFXM
local deviation is ``qs = Us · Hg · q̂`` with ``Hg = (Fg⁻¹)ᵀ − I``; for a screw
``Fg`` is nilpotent so ``Hg = −βᵀ`` exactly and ``qs ∝ (q̂ · b̂)``, which is
zero iff g·b = 0.

This guards the identify candidate-frame construction. The historical bug
(fixed 2026-06-14): ``ud_matrices`` placed the screw axis (Burgers + line, which
``Fd_find_mixed`` puts along frame column 0) along ``n × t``, which rotates with
the character angle. At the pure-screw angle (α = 90°, t ∥ b) that axis became
``n × b ⊥ b``, so the modelled screw carried Burgers ``n × b`` instead of ``b``;
its true extinction obeyed ``q · (n × b) = 0`` rather than ``q · b = 0``. For HCP
prismatic ⟨a⟩ screws on (0002) that lit up a g·b = 0 screw (``n × b ∥ c``); it
was *accidentally* correct for basal HCP and for cubic reflections where
``q · (n × b)`` happened to vanish. ``fixed_ud_matrices`` builds the
character-independent frame ``[b̂ | n̂ | t̂₀]`` so column 0 is the true Burgers.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import fixed_ud_matrices
from dfxm_geo.crystal.dislocations import MixedDislocSpec, find_hg_scene
from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.crystal.slip_systems import (
    burgers_in_plane,
    burgers_in_plane_int,
    hkil_to_hkl,
)

TI = dict(a=2.9505e-10, c=4.6826e-10)


def _screw_deviation_field(mount, plane_int, hkl, b_idx):
    """Max |qs| of the pure-screw (α = 90°) candidate field for one Burgers vector.

    Replicates the identify orchestrator's candidate construction (Cartesian
    n̂ = norm(B·plane), Cartesian unit Burgers, the FIXED frame, rotation_deg = 90),
    then contracts the resulting Hg with q̂ exactly as ``forward`` does
    (``qs = Us · Hg · q̂``). Us = Theta = identity isolates the frame math; the
    zero-ness of ``Hg·q̂`` is frame-invariant under those rotations.
    """
    cell = mount.cell
    if len(plane_int) == 4:
        plane_int = hkil_to_hkl(tuple(plane_int))
    if len(hkl) == 4:
        hkl = hkil_to_hkl(tuple(hkl))
    # q̂ and n̂ as build_forward_context / the orchestrator build them.
    q = cell.B @ np.asarray(hkl, dtype=float)
    q /= np.linalg.norm(q)
    if cell.is_cubic:
        n = np.asarray(plane_int, dtype=float)
        b_units = burgers_in_plane(mount.resolved_structure_type, tuple(plane_int))
    else:
        n = cell.B @ np.asarray(plane_int, dtype=float)
        ints = burgers_in_plane_int(mount.resolved_structure_type, tuple(plane_int)).astype(float)
        cart = (cell.A @ ints.T).T
        b_units = cart / np.linalg.norm(cart, axis=1, keepdims=True)
    n /= np.linalg.norm(n)
    # |b| only sets the field scale (extinction is scale-invariant). Use a (µm).
    b_mag = float(cell.a) * 1e6

    gb = abs(float(q @ b_units[b_idx]))

    Ud = fixed_ud_matrices(n, b_units[b_idx : b_idx + 1])[0]
    spec = MixedDislocSpec(Ud_mix=Ud, rotation_deg=90.0)
    # Ray grid in micrometres, displaced off the core singularity.
    rng = np.random.default_rng(0)
    rl = rng.normal(size=(3, 256)) * 2.0 + 1.5  # µm, no point at origin
    Hg, _ = find_hg_scene(rl, np.eye(3), [spec], np.eye(3), b=b_mag, ny=0.32)
    qs = Hg @ q  # (X, 3) — Us = I
    return gb, float(np.linalg.norm(qs, axis=1).max())


# Absolute deviation below this counts as "extinct" (a visible screw is ~1e-4).
_EXTINCT = 1e-9


def _hcp(families):
    return CrystalMount(
        lattice="hexagonal",
        a=TI["a"],
        c=TI["c"],
        structure_type="hcp",
        material="Ti",
        mount_x=(0, 0, 1),
        mount_y=(0, 1, 0),
        mount_z=(2, -1, 0),
        slip_families=families,
    )


def _fcc():
    return CrystalMount(
        lattice="cubic",
        a=4.05e-10,
        structure_type="fcc",
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


def _bcc():
    return CrystalMount(
        lattice="cubic",
        a=2.87e-10,
        structure_type="bcc",
        material="Fe",
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


class TestScrewGbExtinction:
    """A g·b = 0 screw must be exactly extinct on every slip plane and structure."""

    def test_hcp_basal_a_screw_extinct_on_0002(self):
        mount = _hcp(["{0001}<11-20>"])
        for b_idx in range(3):  # the 3 ⟨a⟩ basis directions (negatives mirror)
            gb, dev = _screw_deviation_field(mount, (0, 0, 1), (0, 0, 2), b_idx)
            assert gb == pytest.approx(0.0, abs=1e-9)
            assert dev < _EXTINCT, f"basal ⟨a⟩ screw b#{b_idx} not extinct: {dev:.3e}"

    def test_hcp_prismatic_a_screw_extinct_on_0002(self):
        """The bug's headline case: prismatic ⟨a⟩ screw on (0002) was VISIBLE."""
        mount = _hcp(["{10-10}<11-20>"])
        gb, dev = _screw_deviation_field(mount, (1, 0, -1, 0), (0, 0, 2), 0)
        assert gb == pytest.approx(0.0, abs=1e-9)
        assert dev < _EXTINCT, f"prismatic ⟨a⟩ screw not extinct: {dev:.3e}"

    def test_fcc_gb0_screw_extinct(self):
        """Cubic control: FCC {111} screw with q·b = 0 must be extinct."""
        # plane (1,1,1); b = [1,-1,0]/√2 is in-plane; q = (2,2,0): q·b = 0.
        mount = _fcc()
        b_units = burgers_in_plane("fcc", (1, 1, 1))
        b_idx = next(i for i, b in enumerate(b_units) if abs(b @ [2.0, 2.0, 0.0]) < 1e-9)
        gb, dev = _screw_deviation_field(mount, (1, 1, 1), (2, 2, 0), b_idx)
        assert gb == pytest.approx(0.0, abs=1e-9)
        assert dev < _EXTINCT, f"FCC g·b=0 screw not extinct: {dev:.3e}"

    def test_bcc_gb0_screw_extinct(self):
        """Cubic control: BCC {110} screw with q·b = 0 must be extinct."""
        mount = _bcc()
        # plane (1,-1,0); pick the in-plane ⟨111⟩ Burgers ⊥ q=(1,1,0): b=[1,1,-1] (q·b=2)?
        # choose q so that some in-plane b has q·b=0. For b=[1,1,1] in (1,-1,0),
        # q=(0,0,2): q·b = 2 (visible). Use q=(1,1,0): for b=[1,1,-2]? not ⟨111⟩.
        # Simplest: plane (1,-1,0) contains b=[1,1,1] and b=[1,1,-1]; q=(1,1,-2)
        # gives q·[1,1,1]=0 -> g·b=0 screw.
        b_units = burgers_in_plane("bcc", (1, -1, 0))
        q = np.array([1.0, 1.0, -2.0])
        b_idx = next((i for i, b in enumerate(b_units) if abs(b @ q) < 1e-9), None)
        assert b_idx is not None, "no g·b=0 ⟨111⟩ Burgers in (1,-1,0) for q=(1,1,-2)"
        gb, dev = _screw_deviation_field(mount, (1, -1, 0), (1, 1, -2), b_idx)
        assert gb == pytest.approx(0.0, abs=1e-9)
        assert dev < _EXTINCT, f"BCC g·b=0 screw not extinct: {dev:.3e}"

    def test_visible_screw_positive_control(self):
        """A g·b ≠ 0 screw MUST produce a large deviation (guards against a
        frame that trivially zeroes everything)."""
        mount = _hcp(["{0001}<11-20>"])
        # basal ⟨a⟩ screw seen by a prismatic-type reflection (10-10): g·b ≠ 0.
        gb, dev = _screw_deviation_field(mount, (0, 0, 1), (1, 0, -1, 0), 0)
        assert gb > 0.3, f"expected a visible reflection, got g·b={gb:.3f}"
        assert dev > 1e-6, f"visible screw deviation too small: {dev:.3e}"
