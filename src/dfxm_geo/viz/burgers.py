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
        burgers: shape (n_burgers, 3) — Burgers vectors (typically 6).
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
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("slip_plane_normal must be a nonzero vector")
    fig = go.Figure()

    # Render the plane through the origin spanned by two in-plane basis vectors
    # orthogonal to n. The earlier `z = (-n_x x - n_y y) / n_z` form divided by
    # n_z and so produced inf/NaN for planes parallel to the z-axis — e.g. the
    # common cubic (110)-type normals with n_z == 0. The basis-vector
    # parameterization is well-defined for any orientation.
    n_hat = n / n_norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_hat, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    s, t = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    plane = s[..., None] * e1 + t[..., None] * e2  # (21, 21, 3)
    xx, yy, zz = plane[..., 0], plane[..., 1], plane[..., 2]
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

    # Burgers vectors (red) — branch convention scales by sqrt(2) for display.
    b_scaled = burgers * np.sqrt(2)
    for i, b_vec in enumerate(b_scaled):
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
