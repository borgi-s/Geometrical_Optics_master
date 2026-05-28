"""Unit tests for dfxm_geo.viz.burgers — interactive 3D plotly viz."""

import sys

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors, rotated_t_vectors


def test_plot_slip_plane_3d_returns_figure_with_expected_traces():
    """Returns a plotly Figure with one surface (the plane) + N traces for vectors."""
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0, 90.0, 180.0]))

    fig = plot_slip_plane_3d(n, b, rotated)

    # 1 surface + 6 Burgers vectors + 3 rotated t-vectors at b_idx=0 = 10 traces.
    assert len(fig.data) == 10


def test_plot_slip_plane_3d_vertical_plane_no_nan():
    """A plane with n_z == 0 (e.g. (1,-1,0)) must render without inf/NaN.

    The old `z = (-n_x x - n_y y) / n_z` form divided by zero for planes
    parallel to the z-axis; the basis-vector parameterization must produce a
    finite surface.
    """
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)  # (110)-type: n_z == 0
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0]))

    fig = plot_slip_plane_3d(n, b, rotated)
    surface = fig.data[0]  # the plane is the first trace
    assert np.isfinite(np.asarray(surface.z)).all()
    assert np.isfinite(np.asarray(surface.x)).all()
    assert np.isfinite(np.asarray(surface.y)).all()


def test_plot_slip_plane_3d_zero_normal_raises():
    """A zero normal vector is invalid input → ValueError."""
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(np.array([1.0, 1.0, 1.0]) / np.sqrt(3), b, np.array([0.0]))
    with pytest.raises(ValueError, match="nonzero"):
        plot_slip_plane_3d(np.zeros(3), b, rotated)


def test_plot_slip_plane_3d_missing_plotly_raises_runtime_error(monkeypatch):
    """If plotly is not installed, raise a clear error pointing to the extras."""
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0]))

    with pytest.raises(RuntimeError, match="plotly is required"):
        plot_slip_plane_3d(n, b, rotated)
