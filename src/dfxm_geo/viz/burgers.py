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

    # The plane within a (-1, 1) box.  Solve for the coordinate axis whose
    # component of n has the largest magnitude to avoid divide-by-zero for
    # normals like (1, 1, 0)/√2 (BCC {110} planes have n[2] == 0).
    #
    # Convention: label the three axes 0/1/2 (x/y/z).  k is the "dependent"
    # axis; the other two are meshed over [-1, 1].  The plane equation is:
    #   n[0]*coords[0] + n[1]*coords[1] + n[2]*coords[2] = 0
    # so  coords[k] = -sum_{j != k} n[j]*coords[j] / n[k].
    k = int(np.argmax(np.abs(n)))  # axis to solve for (largest |n_k| → no div-by-zero)
    j0, j1 = [ax for ax in (0, 1, 2) if ax != k]  # the two free axes

    u = np.linspace(-1, 1, 21)
    uu, vv = np.meshgrid(u, u)
    ww = -(n[j0] * uu + n[j1] * vv) / n[k]

    # Assign the three coordinate grids by axis index.
    coord_map: dict[int, np.ndarray] = {j0: uu, j1: vv, k: ww}
    xx, yy, zz = coord_map[0], coord_map[1], coord_map[2]

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
