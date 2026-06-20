import numpy as np
import pytest

from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell  # existing UnitCell
from dfxm_geo.crystal.slip_systems import burgers_magnitude_of

CUBIC = UnitCell.cubic(
    4.05e-10
)  # Al lattice param a in METRES (UnitCell.cubic takes metres; burgers_magnitude_of returns µm)


def test_unit_normalizes():
    assert np.allclose(np.linalg.norm(fw._unit([3.0, 0.0, 4.0])), 1.0)


def test_in_plane_basis_orthonormal_and_in_plane():
    n = fw._unit([1.0, 1.0, 1.0])
    e1, e2 = fw._in_plane_basis(n)
    assert abs(e1 @ n) < 1e-12 and abs(e2 @ n) < 1e-12
    assert abs(e1 @ e2) < 1e-12
    assert np.allclose([np.linalg.norm(e1), np.linalg.norm(e2)], 1.0)


def test_validate_accepts_good_recipe():
    r = fw.WallRecipe(
        name="t",
        n=(1, 1, 1),
        a=(1, 1, 1),
        sets=(
            fw.DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
    )
    r.validate(CUBIC)  # no raise


def test_validate_rejects_b_not_in_slip_plane():
    r = fw.WallRecipe(
        name="bad",
        n=(1, 1, 1),
        a=(1, 1, 1),
        sets=(
            fw.DislocationSet(b=(1, 1, 1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
    )
    with pytest.raises(ValueError, match="slip_plane"):
        r.validate(CUBIC)


def _eq11():
    return fw.WallRecipe(
        name="leds_eq11",
        n=(1, 1, 1),
        a=(1, 1, 1),
        sets=(
            fw.DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            fw.DislocationSet(b=(0, 1, -1), xi=(-1, 2, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
    )


def test_solver_residual_tiny_for_eq11():
    rho_hat, resid = fw.solve_density_scale(_eq11(), theta_deg=0.05, cell=CUBIC)
    assert resid < 1e-6
    assert np.all(rho_hat > 0)
    assert np.allclose(rho_hat[0], rho_hat[1])  # ratio 1:1


def test_spacing_matches_frank_relation():
    rho_hat, _ = fw.solve_density_scale(_eq11(), theta_deg=0.05, cell=CUBIC)
    b_m = burgers_magnitude_of((1, 0, -1), CUBIC, fraction=1.0) * 1e-6
    d_expected = b_m / (2 * np.sin(np.deg2rad(0.05) / 2))
    d_actual = 1.0 / rho_hat[0]
    assert d_actual == pytest.approx(d_expected, rel=0.05)


def test_frank_residual_matches_solver():
    r = _eq11()
    rho_hat, resid = fw.solve_density_scale(r, 0.05, CUBIC)
    assert fw.frank_residual(r, rho_hat, 0.05, CUBIC) < 1e-6
