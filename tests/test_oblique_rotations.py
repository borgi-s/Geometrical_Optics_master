"""R_lab_to_image rotation: builds the lab→image-detector rotation matrix.

At eta=0, R_lab_to_image must collapse bit-identically to the v2.2.0
implicit rotation R_y(-2θ) — verified at 50 θ values.
"""

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import R_lab_to_image


def _R_y_negative_2theta(theta: float) -> np.ndarray:
    """The v2.2.0 implicit lab→image-detector rotation (eta=0)."""
    c, s = np.cos(-2 * theta), np.sin(-2 * theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.mark.parametrize("theta", np.linspace(np.deg2rad(5.0), np.deg2rad(45.0), 50))
def test_eta_zero_collapses_to_v220_rotation(theta: float) -> None:
    R = R_lab_to_image(eta=0.0, theta=theta)
    np.testing.assert_array_almost_equal(R, _R_y_negative_2theta(theta), decimal=14)


def test_paper_figure3_rotation_is_orthogonal() -> None:
    """At (η=0.3531, θ=0.2691) — paper Figure 3B setup — R should still be a rotation."""
    R = R_lab_to_image(eta=0.3531, theta=0.2691)
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=14)
    np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=14)


def test_R_x_rotation_axis_is_lab_x() -> None:
    """Eta rotation is around lab x̂_l (the beam axis); x̂ is fixed by R_x(η)."""
    R = R_lab_to_image(eta=0.5, theta=0.0)
    np.testing.assert_array_almost_equal(
        R @ np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), decimal=14
    )
