"""cli_main "both, validated" check: config eta must match compute_omega_eta."""

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.reciprocal_space.kernel import _validate_eta_against_compute_omega_eta


@pytest.fixture
def mount() -> CrystalMount:
    return CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


def test_matching_eta_returns_theta_and_omega(mount: CrystalMount) -> None:
    """hkl=(-1,-1,3), keV=19.1, config eta = +20.233° → match η₁, returns θ."""
    eta_target = float(np.deg2rad(20.233))
    theta, omega = _validate_eta_against_compute_omega_eta(
        mount,
        hkl=(-1, -1, 3),
        keV=19.1,
        config_eta=eta_target,
        tol=1e-3,
    )
    assert np.isclose(np.rad2deg(theta), 15.417, atol=1e-3)
    assert np.isclose(np.rad2deg(omega), 6.432, atol=1e-3)


def test_mismatched_eta_raises_with_diff(mount: CrystalMount) -> None:
    """Eta=0 with hkl=(-1,-1,3) at 19.1 keV doesn't match either ±20.233° → error."""
    with pytest.raises(ValueError, match=r"does not match.*η"):
        _validate_eta_against_compute_omega_eta(
            mount,
            hkl=(-1, -1, 3),
            keV=19.1,
            config_eta=0.0,
            tol=1e-6,
        )


def test_unreachable_reflection_raises(mount: CrystalMount) -> None:
    """hkl with sin(θ)>1 at this keV → Laue unsatisfiable error."""
    with pytest.raises(ValueError, match="Laue.*unsatisfiable"):
        _validate_eta_against_compute_omega_eta(
            mount,
            hkl=(20, 20, 20),
            keV=19.1,
            config_eta=0.0,
            tol=1e-6,
        )
