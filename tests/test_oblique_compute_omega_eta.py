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
