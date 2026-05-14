"""3D visualisation of the crystal in lab coordinates.

Pure matplotlib (no plotly). Matches the ESRF_DTU branch's `plot_sample.py`
geometry helpers + draws a cube + axis arrows.
"""

from __future__ import annotations

import numpy as np


def euler_matrix(
    angles_deg: tuple[float, float, float],
    order: str = "xyz",
) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (degrees).

    Default order ``"xyz"`` composes as ``Rx @ Ry @ Rz`` (applied
    right-to-left to column vectors), matching the branch source.

    Args:
        angles_deg: ``(ax, ay, az)`` in degrees.
        order: Composition order, any permutation of "xyz" (e.g. "zyx").

    Returns:
        (3, 3) orthonormal rotation matrix.
    """
    ax, ay, az = np.radians(angles_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    R = np.identity(3)
    for axis in order:
        if axis == "x":
            R = Rx @ R
        elif axis == "y":
            R = Ry @ R
        elif axis == "z":
            R = Rz @ R
        else:
            raise ValueError(f"order must be a permutation of 'xyz'; got {order!r}")
    return R


def _draw_cube(
    ax,
    *,
    side: float = 1.2,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    R: np.ndarray | None = None,
    facecolor: tuple[float, float, float] = (0.6, 0.75, 1.0),
    edgecolor: str = "k",
    alpha: float = 0.65,
) -> None:
    """Draw an axis-aligned cube rotated by R and translated to center."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if R is None:
        R = np.identity(3)
    s = side / 2.0
    V = np.array(
        [
            [+s, +s, +s],
            [+s, +s, -s],
            [+s, -s, +s],
            [+s, -s, -s],
            [-s, +s, +s],
            [-s, +s, -s],
            [-s, -s, +s],
            [-s, -s, -s],
        ]
    )
    V = (R @ V.T).T + np.asarray(center)
    faces_idx = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    faces = [[V[i] for i in idx] for idx in faces_idx]
    poly = Poly3DCollection(
        faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=1.0, alpha=alpha
    )
    ax.add_collection3d(poly)


def plot_crystal_in_lab(
    sample_to_lab_R: np.ndarray | None = None,
    *,
    side: float = 1.2,
    show_axes: bool = True,
):
    """Return a matplotlib Figure showing the sample cube in lab coordinates.

    Args:
        sample_to_lab_R: (3, 3) rotation matrix from sample to lab frame.
            Default: identity (cube axis-aligned with lab).
        side: Cube side length. Default 1.2.
        show_axes: If True, draw RGB lab-frame axis arrows. Default True.

    Returns:
        matplotlib Figure. Caller decides whether to ``.show()`` or
        ``.savefig(path)``.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    if sample_to_lab_R is None:
        sample_to_lab_R = np.identity(3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_cube(ax, side=side, R=sample_to_lab_R)

    if show_axes:
        arrow_len = side * 0.8
        ax.quiver(0, 0, 0, arrow_len, 0, 0, color="r", arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, arrow_len, 0, color="g", arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, arrow_len, color="b", arrow_length_ratio=0.1)

    ax.set_xlabel("x (lab)")
    ax.set_ylabel("y (lab)")
    ax.set_zlabel("z (lab)")
    ax.set_box_aspect((1, 1, 1))
    return fig
