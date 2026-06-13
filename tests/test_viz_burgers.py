"""Unit tests for dfxm_geo.viz.burgers — interactive 3D plotly viz."""

import sys

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors, rotated_t_vectors


def test_plot_slip_plane_3d_returns_figure_with_expected_traces():
    """Returns a plotly Figure with one surface (the plane) + N traces for vectors.

    Uses integer-Miller ⟨110⟩ FCC Burgers vectors (magnitude √2) directly —
    the new contract is that callers pass integer-Miller directions and the
    plotted endpoint reflects the natural length (not unit-normalised).
    """
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    # Integer-Miller ⟨110⟩ Burgers vectors and their negatives (FCC {111} plane).
    b_int = np.array(
        [[-1, 1, 0], [1, 0, -1], [0, 1, -1], [1, -1, 0], [-1, 0, 1], [0, -1, 1]],
        dtype=float,
    )
    # rotated_t_vectors expects unit-normalised Burgers directions.
    b_unit = b_int / np.sqrt(2)
    rotated = rotated_t_vectors(n, b_unit, np.array([0.0, 90.0, 180.0]))

    fig = plot_slip_plane_3d(n, b_int, rotated)

    # 1 surface + 6 Burgers vectors + 3 rotated t-vectors at b_idx=0 = 10 traces.
    assert len(fig.data) == 10

    # New contract: plotted Burgers arrow endpoints must have magnitude √2.
    for trace in fig.data[1 : 1 + len(b_int)]:
        bx, by, bz = trace.x[1], trace.y[1], trace.z[1]
        mag = float(np.sqrt(bx**2 + by**2 + bz**2))
        assert np.isclose(mag, np.sqrt(2), rtol=1e-6), (
            f"Expected FCC Burgers endpoint at √2, got {mag}"
        )


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


def test_plot_slip_plane_3d_fcc_burgers_scaled_to_sqrt2():
    """FCC ⟨110⟩ integer Burgers vectors are scaled by their own norm (√2).

    ``plot_slip_plane_3d`` accepts either unit-normalised or integer vectors;
    this test passes the integer ⟨110⟩ directions (magnitude √2) directly so
    the plotted endpoint should be at distance √2 from the origin.
    """
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    # Integer ⟨110⟩ Burgers vectors (magnitude √2) — the raw integer Miller form.
    b_int = np.array(
        [[-1, 1, 0], [1, 0, -1], [0, 1, -1], [1, -1, 0], [-1, 0, 1], [0, -1, 1]],
        dtype=float,
    )
    rotated = rotated_t_vectors(n, b_int / np.sqrt(2), np.array([0.0]))

    fig = plot_slip_plane_3d(n, b_int, rotated)

    # Burgers traces start at index 1 (index 0 is the slip-plane surface).
    # Each trace x=[0, bx], y=[0, by], z=[0, bz] → endpoint magnitude = |b|.
    for trace in fig.data[1 : 1 + len(b_int)]:
        bx, by, bz = trace.x[1], trace.y[1], trace.z[1]
        mag = float(np.sqrt(bx**2 + by**2 + bz**2))
        assert np.isclose(mag, np.sqrt(2), rtol=1e-6), (
            f"Expected FCC Burgers scaled to √2, got {mag}"
        )


def test_plot_slip_plane_3d_bcc_burgers_scaled_to_sqrt3():
    """BCC ⟨111⟩ integer Burgers vectors are scaled by their own norm (√3).

    Passes the integer ⟨111⟩ directions (magnitude √3) directly; after
    per-vector norm-scaling each plotted endpoint should be at distance √3.
    """
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    # Integer ⟨111⟩ Burgers directions (BCC primary family, magnitude √3).
    b_int = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=float,
    )
    n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # {110} plane normal
    b_unit = b_int / np.linalg.norm(b_int[0])
    rotated = rotated_t_vectors(n, b_unit, np.array([0.0]))

    fig = plot_slip_plane_3d(n, b_int, rotated)

    for trace in fig.data[1 : 1 + len(b_int)]:
        bx, by, bz = trace.x[1], trace.y[1], trace.z[1]
        mag = float(np.sqrt(bx**2 + by**2 + bz**2))
        assert np.isclose(mag, np.sqrt(3), rtol=1e-6), (
            f"Expected BCC Burgers scaled to √3, got {mag}"
        )


def test_plot_slip_plane_3d_110_normal_finite_mesh():
    """slip-plane surface for a {110}-type normal (n[2]==0) must be finite.

    The original ``zz = (-n[0]*xx - n[1]*yy) / n[2]`` divides by zero for
    normals like (1, 1, 0)/√2.  After the fix, the surface trace (index 0)
    must contain no inf or nan values.
    """
    pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # {110} plane — n[2] == 0
    b_int = np.array([[1.0, 1.0, 1.0]])  # single BCC ⟨111⟩ direction
    b_unit = b_int / np.sqrt(3)
    rotated = rotated_t_vectors(n, b_unit, np.array([0.0]))

    fig = plot_slip_plane_3d(n, b_int, rotated)

    # Index 0 is always the slip-plane surface.
    surface = fig.data[0]
    for coord in (surface.x.ravel(), surface.y.ravel(), surface.z.ravel()):
        assert np.all(np.isfinite(coord)), (
            f"Surface mesh contains non-finite values for {110} normal"
        )
