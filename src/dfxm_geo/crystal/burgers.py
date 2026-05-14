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
