import numpy as np
import pytest

from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell  # existing UnitCell

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
