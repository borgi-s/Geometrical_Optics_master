"""Tests for compute_omega_eta — reproduces paper Table A.2 rows.

Paper §6.1 / Table A.2: Al, a = 4.0493 Å, mount (100)//x̂, (010)//ŷ, (001)//ẑ,
19.1 keV, theta < 16.25°. Reflection 1̄1̄3 has:
    ω₁ = 6.432°, ω₂ = 263.568°
    η₁ = 20.233°, η₂ = -20.233°
    θ = 15.417°  (the shared Bragg angle)
"""

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta


@pytest.fixture
def al_paper_mount() -> CrystalMount:
    return CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


def _as_deg(rad: float) -> float:
    return float(np.rad2deg(rad))


def test_table_a2_row_1bar_1bar_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 1̄1̄3 at 19.1 keV: matches paper Table A.2 first row."""
    geom = compute_omega_eta(al_paper_mount, hkl=(-1, -1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(6.432, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(263.568, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(20.233, abs=1e-3)
    assert _as_deg(geom.eta_2) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)
    assert _as_deg(geom.theta_2) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1bar_1_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 1̄13: ω₁=96.432, ω₂=353.568, η₁=20.233, θ=15.417 (group 1)."""
    geom = compute_omega_eta(al_paper_mount, hkl=(-1, 1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(96.432, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(353.568, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1_1_bar_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 11̄3: ω₁=173.568, ω₂=276.432, η₁=-20.233, θ=15.417."""
    geom = compute_omega_eta(al_paper_mount, hkl=(1, -1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(173.568, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(276.432, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1_1_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 113: ω₁=83.568, ω₂=186.432, η₁=-20.233, θ=15.417."""
    geom = compute_omega_eta(al_paper_mount, hkl=(1, 1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(83.568, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(186.432, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_unreachable_reflection_returns_nan(al_paper_mount: CrystalMount) -> None:
    """A reflection with sin(θ) > 1 at the given keV must return NaN-filled geometry."""
    # hkl=(20,20,20) is wildly high-order; at 19.1 keV sin(θ) > 1.
    geom = compute_omega_eta(al_paper_mount, hkl=(20, 20, 20), keV=19.1)
    assert np.isnan(geom.omega_1) and np.isnan(geom.omega_2)
    assert np.isnan(geom.eta_1) and np.isnan(geom.eta_2)


class TestNonCubicSolver:
    """M4 Stage 4.1: the Appendix-A solver is metric-general via C_s/U_mount."""

    MG_A = 3.2094e-10
    MG_C = 5.2108e-10

    def _hex_mount(self):
        return CrystalMount(
            lattice="hexagonal",
            a=self.MG_A,
            c=self.MG_C,
            mount_x=(2, -1, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )

    def test_hexagonal_theta_matches_textbook_d_spacing(self):
        import math

        # For hexagonal (2,-1,0): 1/d^2 = (4/3)(4 - 2 + 1)/a^2 = 4/a^2, d = a/2.
        geom = compute_omega_eta(self._hex_mount(), (2, -1, 0), 17.0)
        lam = 1.239841984e-9 / 17.0
        expected_theta = math.asin(lam / (2.0 * (self.MG_A / 2.0)))
        theta = geom.theta_1 if not np.isnan(geom.theta_1) else geom.theta_2
        assert theta == pytest.approx(expected_theta, rel=1e-10)

    def test_hexagonal_00l_unreachable_with_z_rotation_axis(self):
        # (0,0,2) is parallel to the rotation axis: rotating cannot bring it
        # onto the Ewald sphere -> both omega solutions NaN.
        geom = compute_omega_eta(self._hex_mount(), (0, 0, 2), 17.0)
        assert np.isnan(geom.omega_1) and np.isnan(geom.omega_2)
