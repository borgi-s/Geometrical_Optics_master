"""Slip-plane geometry helpers for the dislocation-identification workflow.

Provides:
    burgers_vectors(slip_plane_normal): {111}-family Burgers vector basis.
    rotated_t_vectors(...): line-direction rotation around n.
    ud_matrices(...): construct Ud_mix from (rotated_t, n, b) basis.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 3 defines Ud = [b | n | t] columns.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

# Lookup table for the four {111}-family slip-plane normals. Each entry maps
# the slug "h k l" (with `-` for minus) to its three basis Burgers vectors,
# in the cubic crystal frame. Negatives are appended in `burgers_vectors`.
_BASIS_TABLE: dict[str, np.ndarray] = {
    "111": np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]], dtype=float),
    "1-11": np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], dtype=float),
    "11-1": np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]], dtype=float),
    "-111": np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]], dtype=float),
}


def _slug(slip_plane_normal: tuple[int, int, int]) -> str:
    """Convert (h, k, l) to a lookup key like '1-11'."""
    return "".join(str(c) if c >= 0 else f"-{abs(c)}" for c in slip_plane_normal)


def burgers_vectors(slip_plane_normal: tuple[int, int, int]) -> np.ndarray:
    """Return the 6 Burgers vectors associated with a {111}-family slip plane.

    Args:
        slip_plane_normal: One of (1,1,1), (1,-1,1), (1,1,-1), (-1,1,1).

    Returns:
        Array of shape (6, 3) — three basis vectors followed by their negatives.
        The branch convention divides the integer basis (e.g. ``[-1, 1, 0]``,
        magnitude √2) by √2, producing unit-magnitude direction vectors. The
        actual Burgers-vector magnitude is applied downstream via the
        `BURGERS_VECTOR` constant.

    Raises:
        ValueError: if slip_plane_normal is not one of the four {111} variants.
    """
    key = _slug(slip_plane_normal)
    if key not in _BASIS_TABLE:
        raise ValueError(
            f"slip_plane_normal {slip_plane_normal} is not one of the four "
            f"{{111}}-family variants {list(_BASIS_TABLE.keys())}"
        )
    basis = _BASIS_TABLE[key]
    return np.vstack([basis, -basis]) / np.sqrt(2)


def gb_cos(q_hkl: np.ndarray, b_vec: np.ndarray) -> float:
    """Normalized |cos∠(G, b)| = |G·b| / (|G||b|) — the g·b visibility scalar.

    0.0 means G ⊥ b (classic invisibility criterion g·b = 0 for screw
    dislocations); 1.0 means G ∥ b (maximum contrast).
    """
    q = np.asarray(q_hkl, dtype=float)
    b = np.asarray(b_vec, dtype=float)
    return float(abs(np.dot(q, b)) / (np.linalg.norm(q) * np.linalg.norm(b)))


def gb_visible(q_hkl: np.ndarray, b_vec: np.ndarray, threshold_deg: float) -> bool:
    """True when the dislocation is NOT within `threshold_deg` of invisibility.

    Bit-identical to the historical inline criterion in
    ``pipeline._passes_invisibility``: visible ⇔ cos∠(G,b) ≥ cos(90° − threshold).
    """
    return bool(gb_cos(q_hkl, b_vec) >= np.cos(np.deg2rad(90.0 - threshold_deg)))


def rotated_t_vectors(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """Rotate each in-plane initial line direction `t_0 = b × n` around `n`.

    For every (angle, Burgers vector) pair, computes
    ``R_n(angle) · (b × n)`` where ``R_n`` is rotation around `n` (degrees).
    Mirrors the branch source's `BurgersVectorsPlotter.calculate_rotated_vectors`.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        burgers: shape (n_burgers, 3) — the Burgers vectors.
        angles_deg: shape (n_angles,) — rotation angles in degrees.

    Returns:
        ndarray of shape (n_angles, n_burgers, 3) — rotated t-vectors.
    """
    n = np.asarray(slip_plane_normal, dtype=float)
    t_0 = np.cross(burgers, n)  # (n_burgers, 3)

    out = np.zeros((len(angles_deg), len(burgers), 3))
    for i, angle in enumerate(angles_deg):
        rot = _Rotation.from_rotvec(angle * n, degrees=True)
        out[i] = rot.apply(t_0)
    return out


def ud_matrices(
    slip_plane_normal: np.ndarray,
    rotated_vectors: np.ndarray,
) -> np.ndarray:
    """Construct Ud_mix matrices from (n × t, n, t) basis frames.

    Mirrors the branch source's `BurgersVectorsPlotter.calculate_ud_matrices`,
    which stacks ``(np.cross(n, t), n, t)`` as the three columns of each Ud
    matrix.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        rotated_vectors: shape (n_angles, n_burgers, 3) from `rotated_t_vectors`.

    Returns:
        ndarray of shape (n_angles, n_burgers, 3, 3).
    """
    n = np.asarray(slip_plane_normal, dtype=float)
    n_angles, n_burgers, _ = rotated_vectors.shape
    Ud = np.zeros((n_angles, n_burgers, 3, 3))
    for i in range(n_angles):
        for j in range(n_burgers):
            t = rotated_vectors[i, j]
            cross_nt = np.cross(n, t)
            Ud[i, j] = np.column_stack([cross_nt, n, t])
    return Ud
