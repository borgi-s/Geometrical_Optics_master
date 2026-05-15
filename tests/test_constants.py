"""Sanity checks for dfxm_geo.constants — values match the legacy hardcoded ones."""

import numpy as np

from dfxm_geo import constants


def test_burgers_vector_is_aluminium_lattice():
    """BURGERS_VECTOR = 2.862e-4 µm corresponds to Al."""
    assert constants.BURGERS_VECTOR == 2.862e-4


def test_poisson_ratio_is_aluminium():
    """POISSON_RATIO ≈ 0.334 for Al at RT."""
    assert constants.POISSON_RATIO == 0.334


def test_id06_theta_0_matches_legacy():
    """Bragg angle θ₀ matches the legacy 17.953° / 2 value."""
    expected = 17.953 / 2 * np.pi / 180
    assert expected == constants.ID06_THETA_0


def test_id06_grid_constants_are_ints():
    """Npixels and Nsub are integers (used as array dimensions)."""
    assert isinstance(constants.ID06_NPIXELS, int)
    assert isinstance(constants.ID06_NSUB, int)


def test_hkl_default_is_a_3_tuple():
    """Default Miller indices for the active reflection."""
    assert constants.HKL_DEFAULT == (-1, 1, -1)
    assert len(constants.HKL_DEFAULT) == 3
