"""Unit tests for dfxm_geo.viz.sample — 3D crystal-in-lab visualisation."""

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from dfxm_geo.viz.sample import euler_matrix, plot_crystal_in_lab


def test_euler_matrix_zero_is_identity():
    np.testing.assert_allclose(euler_matrix((0.0, 0.0, 0.0)), np.identity(3), atol=1e-12)


def test_euler_matrix_is_orthonormal():
    """Euler matrix has det=1 and is orthonormal (R @ R.T = I)."""
    R = euler_matrix((30.0, 45.0, 60.0))
    np.testing.assert_allclose(R @ R.T, np.identity(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_euler_matrix_order_kwarg():
    """The order kwarg controls the rotation composition; xyz != zyx in general."""
    angles = (10.0, 20.0, 30.0)
    R_xyz = euler_matrix(angles, order="xyz")
    R_zyx = euler_matrix(angles, order="zyx")
    assert not np.allclose(R_xyz, R_zyx)


def test_plot_crystal_in_lab_returns_figure():
    """Returns a matplotlib Figure; doesn't raise."""
    fig = plot_crystal_in_lab(sample_to_lab_R=np.identity(3))
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)
