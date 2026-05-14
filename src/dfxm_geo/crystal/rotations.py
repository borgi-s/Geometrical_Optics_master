"""Rotation utilities for sample / crystal / lab frame transformations.

Functions:
    rotatedU(axis, alpha, U, coordtype) -> ndarray
        Rotate U around `axis` by `alpha` radians (Rodrigues' formula).
    fast_inverse2(A) -> ndarray
        Numba-JIT bulk inverse of an (N, 3, 3) array of 3x3 matrices.
"""

import numpy as np
from numba import jit


def rotatedU(
    axis: np.ndarray,
    alpha: float,
    U: np.ndarray,
    coordtype: str,
) -> np.ndarray:
    """Rotate U around `axis` by `alpha` radians.

    `axis` is interpreted in sample coords unless `coordtype == "cryst"`,
    in which case it is first transformed into sample coords via U.

    Args:
        axis: 3-vector rotation axis.
        alpha: rotation angle in radians (despite legacy comments).
        U: 3x3 rotation matrix to rotate.
        coordtype: "cryst" or "sample".

    Returns:
        Rotated 3x3 matrix.
    """
    if coordtype == "cryst":
        axis = np.dot(U, axis)  # convert to sample
    r1, r2, r3 = axis[0], axis[1], axis[2]
    cost = np.cos(alpha)
    onecost = 1 - cost
    sint = np.sin(alpha)
    R = np.array(
        [
            [
                r1 * r1 * onecost + cost,
                r1 * r2 * onecost + r3 * sint,
                r1 * r3 * onecost - r2 * sint,
            ],
            [
                r1 * r2 * onecost - r3 * sint,
                r2 * r2 * onecost + cost,
                r2 * r3 * onecost + r1 * sint,
            ],
            [
                r1 * r3 * onecost + r2 * sint,
                r2 * r3 * onecost - r1 * sint,
                r3 * r3 * onecost + cost,
            ],
        ]
    )
    return np.dot(R, U)


@jit("float64[:,:,:](float64[:,:,:])", nopython=True, fastmath=True)
def fast_inverse2(A):  # Try to rewrite this
    inv = np.empty_like(A)
    a = A[:, 0, 0]
    b = A[:, 0, 1]
    c = A[:, 0, 2]
    d = A[:, 1, 0]
    e = A[:, 1, 1]
    f = A[:, 1, 2]
    g = A[:, 2, 0]
    h = A[:, 2, 1]
    i = A[:, 2, 2]

    inv[:, 0, 0] = e * i - f * h
    inv[:, 1, 0] = -(d * i - f * g)
    inv[:, 2, 0] = d * h - e * g
    inv_det = 1 / (a * inv[:, 0, 0] + b * inv[:, 1, 0] + c * inv[:, 2, 0])

    inv[:, 0, 0] *= inv_det
    inv[:, 0, 1] = -inv_det * (b * i - c * h)
    inv[:, 0, 2] = inv_det * (b * f - c * e)
    inv[:, 1, 0] *= inv_det
    inv[:, 1, 1] = inv_det * (a * i - c * g)
    inv[:, 1, 2] = -inv_det * (a * f - c * d)
    inv[:, 2, 0] *= inv_det
    inv[:, 2, 1] = -inv_det * (a * h - b * g)
    inv[:, 2, 2] = inv_det * (a * e - b * d)
    return inv


def rotate_matrix_z_axis(matrix: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate `matrix` by `angle_degrees` around the lab z axis.

    Returns ``R_z(angle) @ matrix``. Matches the branch source's
    ``functions.rotate_matrix_z_axis`` from the ESRF_DTU port.

    Args:
        matrix: Shape (3, 3) — the matrix to left-rotate.
        angle_degrees: Rotation angle in degrees around the lab z axis.

    Returns:
        Rotated (3, 3) matrix.
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    Rz = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Rz @ matrix
