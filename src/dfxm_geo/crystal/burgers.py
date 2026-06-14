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

from dfxm_geo.crystal.slip_systems import burgers_in_plane

# Canonical ordering of the three positive basis Burgers vectors for each
# {111}-family plane, expressed as unit-normalized directions. This ordering
# is preserved from the original hand-written table so that downstream
# index-based selection (e.g. b_vector_indices = [0]) continues to select the
# same physical Burgers vector before and after the refactor.
# Values are bit-identical to the old `_BASIS_TABLE / sqrt(2)` table.
# NOTE: Do NOT extend this table with BCC/HCP entries — add them to the
# registry in slip_systems.py. Task 9 makes the orchestrator structure-aware
# and will supersede this shim entirely.
_ORDERED_BASES: dict[tuple[int, int, int], np.ndarray] = {
    (1, 1, 1): np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]], dtype=float) / np.sqrt(2),
    (1, -1, 1): np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], dtype=float) / np.sqrt(2),
    (1, 1, -1): np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]], dtype=float) / np.sqrt(2),
    (-1, 1, 1): np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]], dtype=float) / np.sqrt(2),
}


def burgers_vectors(slip_plane_normal: tuple[int, int, int]) -> np.ndarray:
    """Return the 6 Burgers vectors associated with a {111}-family slip plane.

    FCC-compat shim over the structure-family registry. Non-{111} planes raise
    (the identify workflow defaults to FCC). The ordering of the 6 vectors is
    preserved bit-identically from the original hand-written table so that
    index-based selection (``b_vector_indices``) selects the same vector before
    and after the refactor.

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
    # Validate via the registry (raises for non-{111} planes; future structure
    # types will override this whole shim in a later task).
    try:
        burgers_in_plane("fcc", slip_plane_normal)
    except ValueError as exc:
        raise ValueError(
            f"slip_plane_normal {slip_plane_normal} is not one of the four {{111}}-family variants"
        ) from exc
    basis = _ORDERED_BASES[tuple(slip_plane_normal)]  # type: ignore[index]
    return np.vstack([basis, -basis])


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


def fixed_ud_matrices(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
) -> np.ndarray:
    """Character-independent dislocation frames ``[b̂ | n̂ | t̂₀]`` — one per Burgers.

    Unlike :func:`ud_matrices` (whose column 0 is ``n × t`` and therefore rotates
    with the character angle), this frame is FIXED: column 0 is the Burgers
    direction ``b̂``, column 1 the slip-plane normal ``n̂``, column 2 the reference
    pure-edge line ``t̂₀ = b̂ × n̂``. ``Fd_find_mixed`` / ``find_hg_scene`` then
    encode the edge↔screw character ENTIRELY through ``rotation_deg``, so the
    modelled Burgers vector stays ``b̂`` for every character angle.

    This is required for physically correct screw contrast. At the pure-screw
    angle (``rotation_deg = 90°``) ``Fd_find_mixed`` places the screw axis (both
    the line and the Burgers vector) along column 0, so column 0 MUST be the true
    Burgers ``b̂`` for a ``g·b = 0`` screw to be invisible — the textbook
    isotropic-elasticity result (Hirth & Lothe; Howie & Whelan 1961). The earlier
    per-angle construction ``ud_matrices(n, rotated_t_vectors(n, b, angles))``
    rotated column 0 to ``n × t ≈ n × b ⊥ b`` at the screw angle, so the modelled
    screw carried Burgers ``n × b`` instead of ``b`` and a ``g·b = 0`` screw could
    light up (HCP prismatic ⟨a⟩ on (0002): ``n × b ∥ c``). See
    ``tests/test_screw_gb_extinction.py``.

    Equals ``ud_matrices(n, rotated_t_vectors(n, burgers, [0.0]))[0]`` (the α = 0
    frame) by construction, so identify pure-edge candidates stay byte-identical;
    only ``rotation_deg ≠ 0`` (screw / mixed) candidate fields change, and they
    change to the physically correct values.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal ``n`` (need not be
            unit length).
        burgers: shape (n_burgers, 3) — the in-plane Burgers vectors (need not be
            unit length).

    Returns:
        ndarray of shape (n_burgers, 3, 3): the fixed frames, column-stacked
        ``[b̂ | n̂ | t̂₀]``.
    """
    t_0 = rotated_t_vectors(slip_plane_normal, burgers, np.array([0.0]))  # (1, m, 3)
    return ud_matrices(slip_plane_normal, t_0)[0]  # (m, 3, 3)


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
