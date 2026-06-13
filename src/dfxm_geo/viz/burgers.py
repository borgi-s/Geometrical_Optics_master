"""Interactive 3D visualization of slip-plane geometry (plotly).

Requires the optional ``[identification]`` dep group. Plotly is imported
lazily inside `plot_slip_plane_3d` so the rest of the package imports
cleanly when plotly isn't installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_slip_plane_3d(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    rotated_vectors: np.ndarray,
) -> go.Figure:
    """Interactive 3D figure showing slip plane + Burgers vectors + rotated t-vectors.

    Caller decides whether to `.show()` (in a notebook) or `.write_html(path)`.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        burgers: shape (n_burgers, 3) — Burgers vectors plotted at their natural
            length.  Pass integer-Miller directions (e.g. ``[-1, 1, 0]`` for FCC
            ⟨110⟩, ``[1, 1, 1]`` for BCC ⟨111⟩) so the display length equals
            the integer norm (``√2`` / ``√3`` respectively).  Unit-normalised
            inputs are also accepted — in that case plotted lengths are ≈1.
        rotated_vectors: shape (n_angles, n_burgers, 3) — rotated t-vectors.

    Returns:
        plotly Figure.

    Raises:
        RuntimeError: if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "plotly is required for plot_slip_plane_3d. "
            "Install via: pip install 'dfxm-geo[identification]'"
        ) from exc

    n = np.asarray(slip_plane_normal, dtype=float)
    fig = go.Figure()

    # The plane (z = (-n_x x - n_y y) / n_z within a (-1, 1) box).
    xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    zz = (-n[0] * xx - n[1] * yy) / n[2]
    fig.add_trace(
        go.Surface(
            z=zz,
            x=xx,
            y=yy,
            showscale=False,
            colorscale=[[0, "black"], [1, "black"]],
            opacity=0.6,
        )
    )

    # Burgers vectors (red) — plotted at their natural length.
    # The caller controls the display scale by choosing whether to pass
    # integer-Miller vectors (|FCC ⟨110⟩| = √2, |BCC ⟨111⟩| = √3) or
    # unit-normalised directions.  The old hard-coded ``* √2`` assumed FCC
    # ⟨110⟩ unit inputs; using integer inputs generalises to any structure.
    for i, b_vec in enumerate(burgers):
        fig.add_trace(
            go.Scatter3d(
                x=[0, b_vec[0]],
                y=[0, b_vec[1]],
                z=[0, b_vec[2]],
                mode="lines+markers",
                name=f"b_{i}",
                marker=dict(size=4, color="red"),
                line=dict(color="red", width=4),
            )
        )

    # Rotated t-vectors at b_idx=0 (blue) — one per angle.
    for i, vec in enumerate(rotated_vectors[:, 0]):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode="lines+markers",
                name=f"t_{i}",
                marker=dict(size=4, color="blue"),
                line=dict(color="blue", width=3),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="x [001]",
            yaxis_title="y [010]",
            zaxis_title="z [100]",
            aspectmode="cube",
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
        ),
    )
    return fig
