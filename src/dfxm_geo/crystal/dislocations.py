"""Dislocation displacement-gradient fields (edge dislocations).

Computes the Fdd / Fg tensors used by the DFXM forward model for
crystals containing one or more edge dislocations arranged in a wall.

Public functions:
    Fd_find — main entry: returns Fg given lab-frame coords + rotations.
    multi_dislocs_parallel — internal worker for parallel ndis>100 path.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from joblib import cpu_count
from numba import jit, njit

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO

# Module-level identity used as the default for the S kwarg in Fd_find.
# Defined here (not inline) to satisfy ruff B008 (no function calls in defaults).
_S_IDENTITY: np.ndarray = np.identity(3)


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _accumulate_bipolar_walls(
    rd: np.ndarray, dis: float, i_start: int, i_end: int, ny: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-JIT accumulator for the bipolar wall contributions to Fdd.

    Sums the i ∈ [i_start, i_end) contributions of an infinite-wall
    convention (odd i shifts +k*dis, even i shifts -k*dis, with k =
    ceil(i/2)). Returns the four nonzero Fdd components ``(f00, f01, f10,
    f11)`` as ``(N,)`` arrays. Caller adds these to the central-wall (i=0)
    Fdd in-place.

    Used from both branches of Fd_find: the sequential branch passes
    ``(1, ndis)`` to cover the whole wall, while the parallel branch's
    workers pass ``(chunk.start, chunk.stop)`` for their slice.

    Compiled with ``fastmath=True`` so the compiler may reassociate the
    accumulation; the result agrees with the original Python loop to
    within ~1e-12 abs / ~1e-10 rel on identity-transform inputs (verified
    by TestFdFindBipolarWall).
    """
    N = rd.shape[1]
    f00 = np.zeros(N)
    f01 = np.zeros(N)
    f10 = np.zeros(N)
    f11 = np.zeros(N)

    sqx = rd[0] * rd[0]
    for i in range(i_start, i_end):
        k = (i + 1) // 2
        offset = float(k) * dis if i % 2 == 1 else -float(k) * dis
        rd_y = rd[1] + offset
        sqy = rd_y * rd_y
        sum_sqxy = sqx + sqy
        denom = sum_sqxy * sum_sqxy
        nyfactor = 2.0 * ny * sum_sqxy

        f00 += -rd_y * (3.0 * sqx + sqy - nyfactor) / denom
        f01 += rd[0] * (3.0 * sqx + sqy - nyfactor) / denom
        f10 += -rd[0] * (3.0 * sqy + sqx - nyfactor) / denom
        f11 += rd_y * (sqx - sqy + nyfactor) / denom

    return f00, f01, f10, f11


def multi_dislocs_parallel(
    chunk: range,
    rd: np.ndarray,
    Fdd_shape: tuple[int, int, int],
    dis: float,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
    """Worker for Fd_find's parallel branch (ndis > 100).

    Delegates the per-i math to :func:`_accumulate_bipolar_walls` (numba-JIT)
    so each ThreadPoolExecutor worker runs the inner loop in compiled code
    with the GIL released. The chunk's index range is forwarded as
    ``(chunk.start, chunk.stop)``.

    Bipolar wall convention matches the sequential branch:
        odd  i (1, 3, 5, …) → +ceil(i/2) * dis
        even i (2, 4, 6, …) → -(i/2)    * dis
    Prior to 2026-05-12 this worker used a monotone one-sided wall pattern
    (``rd_new[1] -= i * dis``) that disagreed with the sequential branch
    by ~22% in Fdd Frobenius norm; production sims with ndis>100 hit this
    path, so reference results predating the fix used the one-sided wall.

    Args:
        chunk: ``range`` of dislocation indices to process. Only
            ``chunk.start`` and ``chunk.stop`` are read.
        rd: ``(3, N)`` coordinates in dislocation space.
        Fdd_shape: Shape of the Fdd tensor to be accumulated; ``(N, 3, 3)``.
        dis: Spacing between dislocations in the wall (µm).
        ny: Poisson ratio. Default = :data:`POISSON_RATIO`.

    Returns:
        ndarray of shape ``Fdd_shape`` with only the four nonzero
        components (``[0,0]``, ``[0,1]``, ``[1,0]``, ``[1,1]``) filled in.
    """
    f00, f01, f10, f11 = _accumulate_bipolar_walls(rd, dis, chunk.start, chunk.stop, ny)
    Fdd_chunk = np.zeros(Fdd_shape)
    Fdd_chunk[:, 0, 0] = f00
    Fdd_chunk[:, 0, 1] = f01
    Fdd_chunk[:, 1, 0] = f10
    Fdd_chunk[:, 1, 1] = f11
    return Fdd_chunk


def Fd_find(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float = 1,
    ndis: int = 1,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    misorientation: bool = False,
    t_vec: np.ndarray | None = None,
    *,
    S: np.ndarray = _S_IDENTITY,
) -> np.ndarray:
    """Calculate the displacement gradient field for an edge-dislocation wall.

    Args:
        rl: np.ndarray of shape (3, X). Coordinates in lab space.
        Ud: 3x3 rotation matrix (dislocation -> grain).
        Us: 3x3 rotation matrix (grain -> sample).
        Theta: 3x3 rotation matrix (sample -> lab).
        dis: Spacing between dislocations (µm). Default 1.
        ndis: Number of dislocations. Default 1.
        b: Burgers vector magnitude (µm). Default `BURGERS_VECTOR`.
        ny: Poisson ratio. Default `POISSON_RATIO`.
        misorientation: If True, branch into the misorientation path
            (sets up split sample-space rotations U_sr / U_sl).
        t_vec: Tangent vector, required if `misorientation` is True.
        S: 3x3 rotation matrix (sample-remount; default identity).

    Returns:
        ndarray of shape (X, 3, 3): Fg in grain space.
    """
    if misorientation:
        # The misorientation=True path has never worked. Two independent bugs
        # exist in pre-cleanup `main` and the CDD_Khaled clone:
        #   1. The piecewise rotation `rc = U_sl.T @ rs[:, left] & U_sr.T @ rs[:, right]`
        #      uses bitwise `&` on float matmul results, which raises TypeError
        #      before reaching anything downstream.
        #   2. Even if (1) is fixed, the unconditional fall-through below overwrites
        #      `rs`, `rc`, and `rd`, so any misorientation-derived state is discarded.
        # No reference behaviour exists. If this path is needed, open an issue with
        # the intended physics formulation (and a clear definition of what the lower
        # half-space rs[1] <= 0 should do — the current masks don't cover it).
        raise NotImplementedError(
            "Fd_find(misorientation=True) is not implemented — the pre-cleanup "
            "branch was dead-broken and has no reference behaviour to preserve."
        )

    rs = Theta @ rl
    rgon = S.T @ rs  # sample-remount (Purdue 2024); S = identity → rgon == rs
    rc = Us.T @ rgon
    rd = Ud.T @ rc

    Fdd = np.zeros([len(rd[-1]), 3, 3])
    alpha = 1e-20

    sqx = rd[0] * rd[0]
    sqy = rd[1] * rd[1]
    denom = (sqx + sqy) * (sqx + sqy) + alpha
    bfactor = b / (4 * np.pi * (1 - ny))
    nyfactor = 2 * ny * (sqx + sqy)

    # The `+nyfactor` on [1,1] (vs `-nyfactor` on the other three components)
    # applies the sign correction documented in Appendix A of
    # Borgi et al., J. Appl. Cryst. 2024 (doi.org/10.1107/S1600576724001183).
    # Pre-correction `main` had `-nyfactor` here; the fix was already in the
    # CDD_Khaled / Beam_Stop branch and is the version used by the paper.
    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] = rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] = rd[1] * (sqx - sqy + nyfactor) / denom

    if ndis > 100:
        njobs = cpu_count()
        chunk_size = int(np.ceil((ndis - 1) / njobs))

        chunks = [
            range(start, min(start + chunk_size, ndis)) for start in range(1, ndis, chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=njobs) as executor:
            results = list(
                executor.map(
                    multi_dislocs_parallel,
                    chunks,
                    [rd] * len(chunks),
                    [Fdd.shape] * len(chunks),
                    [dis] * len(chunks),
                )
            )

        for Fdd_i in results:
            Fdd += Fdd_i
    elif ndis > 1:
        # Numba-JIT accumulator over the bipolar walls (~2× faster than the
        # original Python loop at typical N; a NumPy-only vectorisation was
        # benchmarked and rejected — its (ndis-1, N) intermediates blew out
        # L2 cache and slowed things down by 3×).
        f00, f01, f10, f11 = _accumulate_bipolar_walls(rd, dis, 1, ndis, ny)
        Fdd[:, 0, 0] += f00
        Fdd[:, 0, 1] += f01
        Fdd[:, 1, 0] += f10
        Fdd[:, 1, 1] += f11

    Fdd *= bfactor
    Fdd += np.identity(3)
    return Ud @ Fdd @ Ud.T


def Fd_find_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    Ud_mix: np.ndarray,
    rotation_deg: float,
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    position_lab_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    S: np.ndarray = _S_IDENTITY,
) -> np.ndarray:
    """Displacement gradient Fg for a single mixed-character dislocation.

    Implements Eq. 1 of Borgi, Winther & Poulsen (2025), J. Appl. Cryst. 58,
    813-821, doi:10.1107/S1600576725002614::

        F_d = I + screw_matrix * cos(α_paper) + edge_matrix * sin(α_paper)

    where α_paper is the angle between Burgers vector and dislocation line
    (α_paper = 0° / 180° pure screw; 90° / 270° pure edge).

    **Parameterization note (differs from paper)**: ``rotation_deg`` is the
    angle (degrees) by which the line direction has been rotated around the
    slip-plane normal `n`, starting from the initial in-plane reference
    ``t_0 = b × n`` (which has α_paper=90°, pure edge). The two
    parameterizations satisfy ``α_paper = 90° - rotation_deg``, so:

        rotation_deg = 0   ⇔ α_paper = 90°  (pure edge)
        rotation_deg = 90° ⇔ α_paper = 0°   (pure screw)

    Naming preserves the convention of the branch source (`disloc_identify`)
    rather than the paper to keep callers unchanged.

    **Convenience equivalences (use these instead of separate functions):**

    - ``Fd_find_mixed(..., rotation_deg=0)`` is the pure-edge field
      (equivalent to ``Fd_find(..., ndis=1)`` with matching Ud).
    - ``Fd_find_mixed(..., rotation_deg=90)`` is the pure-screw field
      (only the screw out-of-plane terms ∂u_dx/∂y, ∂u_dx/∂z survive).

    We don't ship separate `Fd_find_edge` / `Fd_find_screw` wrappers; the
    ESRF_DTU branch had them but they're just `Fd_find_mixed` at the
    specific rotation angles above (Borgi 2025 Eq. 1's α=90° and α=0°
    limits respectively).

    Args:
        rl: Lab-frame coordinates, shape (3, X).
        Us: Sample-to-grain rotation (Eq. 5 of Borgi 2025), shape (3, 3).
        Ud_mix: Dislocation-to-grain rotation (Eq. 3 of Borgi 2025), shape (3, 3).
        rotation_deg: See parameterization note above.
        Theta: Lab-to-sample rotation (Eq. 7 of Borgi 2025), shape (3, 3).
        b: Burgers vector magnitude. Default `BURGERS_VECTOR` from constants.
        ny: Poisson ratio. Default `POISSON_RATIO` from constants.
        position_lab_um: Lab-frame offset (µm); shifts rl before transforming
            to dislocation coords so the core sits at this offset. Default 0.
        S: 3x3 rotation matrix (sample-remount; default identity).

    Returns:
        Fg of shape (X, 3, 3) in the grain frame, with the identity added.
    """
    if position_lab_um != (0.0, 0.0, 0.0):
        # rl is in micrometres (callers pass rl * 1e6); the offset is already
        # in micrometres, so subtract directly — no metre conversion.
        offset_um = np.asarray(position_lab_um).reshape(3, 1)
        rl = rl - offset_um

    # Eq. 8 of Borgi 2025: r_l = Θ^T · Us · Ud · r_d → r_d = Ud^T · Us^T · Θ · r_l.
    rs = Theta @ rl
    rgon = S.T @ rs  # sample-remount (Purdue 2024); S = identity → rgon == rs
    rc = Us.T @ rgon
    rd = Ud_mix.T @ rc

    Fdd = np.zeros([rd.shape[1], 3, 3])
    alpha = 1e-20

    sqx = rd[0] * rd[0]
    sqy = rd[1] * rd[1]
    denom = (sqx + sqy) * (sqx + sqy) + alpha
    bfactor = b / (4 * np.pi * (1 - ny))
    nyfactor = 2 * ny * (sqx + sqy)

    # Edge formula (Appendix-A sign correction already applied: +nyfactor on [1,1]).
    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] = rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] = rd[1] * (sqx - sqy + nyfactor) / denom

    Fdd *= bfactor
    Fdd *= np.cos(np.deg2rad(rotation_deg))

    # Screw out-of-plane contributions (∂u_dx/∂y, ∂u_dx/∂z).
    # denom1 = z² + y² preserved from branch source (Eq. 1 pure-screw matrix).
    sqz = rd[2] * rd[2]
    denom1 = sqz + sqy + alpha
    bfactor1 = b / (2 * np.pi)
    sin_rot = np.sin(np.deg2rad(rotation_deg))

    Fdd[:, 0, 1] += (-rd[2] / denom1) * bfactor1 * sin_rot
    Fdd[:, 0, 2] += (rd[1] / denom1) * bfactor1 * sin_rot

    Fdd += np.identity(3)
    return Ud_mix @ Fdd @ Ud_mix.T


@dataclass(frozen=True)
class MixedDislocSpec:
    """Specification for one mixed-character dislocation.

    Attributes:
        Ud_mix: Rotation matrix from dislocation to grain frame (Eq. 3 of
            Borgi 2025), shape (3, 3). Columns are (b, n, t) basis vectors.
        rotation_deg: Rotation angle (degrees) of the line direction `t`
            around the slip-plane normal `n`, starting from `t_0 = b × n`.
            See `Fd_find_mixed` docstring for the relation to the paper's α.
        position_lab_um: Lab-frame offset (µm) applied to ``rl`` so the
            dislocation core sits at the given (x, y, z). Default (0, 0, 0).
    """

    Ud_mix: np.ndarray
    rotation_deg: float
    position_lab_um: tuple[float, float, float] = (0.0, 0.0, 0.0)


def Fd_find_multi_dislocs_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    crystals: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    S: np.ndarray = _S_IDENTITY,
) -> np.ndarray:
    """Sum of mixed-dislocation contributions from N crystals.

    Generalises Eq. 1 of Borgi 2025 to multiple dislocations: each crystal's
    screw+edge contributions are summed (no per-crystal identity), and the
    identity is added once at the end. For N=1 this reduces to
    ``Fd_find_mixed``; for N=2 it is the case used by the multi-disloc Monte
    Carlo pipeline mode.

    Args:
        rl: Lab-frame coordinates, shape (3, X).
        Us: Sample-to-grain rotation, shape (3, 3).
        crystals: list of `MixedDislocSpec`, at least one.
        Theta: Lab-to-sample rotation, shape (3, 3).
        b: Burgers vector magnitude (µm).
        ny: Poisson ratio.
        S: 3x3 rotation matrix (sample-remount; default identity).

    Returns:
        Fg of shape (X, 3, 3) in the grain frame, with the identity added once.
    """
    if not crystals:
        raise ValueError("Fd_find_multi_dislocs_mixed requires at least one crystal")

    I = np.identity(3)
    Fg_sum = np.zeros((rl.shape[1], 3, 3))
    for spec in crystals:
        Fg_one = Fd_find_mixed(
            rl,
            Us,
            Ud_mix=spec.Ud_mix,
            rotation_deg=spec.rotation_deg,
            Theta=Theta,
            b=b,
            ny=ny,
            position_lab_um=spec.position_lab_um,
            S=S,
        )
        Fg_sum += Fg_one - I

    return Fg_sum + I


@njit(cache=True, nogil=True, fastmath=False)
def _population_hg_kernel(
    rl_um: np.ndarray,  # (3, X) float64, MICROMETRES
    M: np.ndarray,  # (N, 3, 3) float64 = Ud.T @ Us.T @ S.T @ Theta
    offset: np.ndarray,  # (N, 3) float64, micrometres
    Ud: np.ndarray,  # (N, 3, 3) float64
    cos_rot: np.ndarray,  # (N,) float64
    sin_rot: np.ndarray,  # (N,) float64
    b: float,
    ny: float,
    Hg_out: np.ndarray,  # (X, 3, 3) float64  (written in place)
) -> None:
    """Fused population displacement-gradient kernel: rl -> Hg, in one pass.

    Replaces the NumPy composition
        Fd_find_multi_dislocs_mixed (per-dislocation field + Ud rotation + sum)
        -> fast_inverse2 -> transpose - I.
    Looping dislocations *inside* the ray loop keeps it flat in memory whether
    N (ndis) is 4 or 200 (a NumPy broadcast would materialize (N,X,3,3)).
    fastmath=False keeps reassociation tame so parity holds at rtol=1e-12.
    See math reference in the Phase 1 plan
    (docs/superpowers/plans/2026-05-27-find-hg-numba-fusion.md).
    """
    X = rl_um.shape[1]
    N = M.shape[0]
    alpha = 1e-20
    bf = b / (4.0 * math.pi * (1.0 - ny))
    bf1 = b / (2.0 * math.pi)

    # Per-ray scratch reused across iterations (no per-ray heap allocation).
    G = np.zeros((3, 3))
    Tmp = np.zeros((3, 3))

    for x in range(X):
        rx = rl_um[0, x]
        ry = rl_um[1, x]
        rz = rl_um[2, x]

        # Fg accumulator = I + sum_d Ud_d @ G_d @ Ud_d.T
        f00 = 1.0
        f01 = 0.0
        f02 = 0.0
        f10 = 0.0
        f11 = 1.0
        f12 = 0.0
        f20 = 0.0
        f21 = 0.0
        f22 = 1.0

        for d in range(N):
            dx = rx - offset[d, 0]
            dy = ry - offset[d, 1]
            dz = rz - offset[d, 2]
            rd0 = M[d, 0, 0] * dx + M[d, 0, 1] * dy + M[d, 0, 2] * dz
            rd1 = M[d, 1, 0] * dx + M[d, 1, 1] * dy + M[d, 1, 2] * dz
            rd2 = M[d, 2, 0] * dx + M[d, 2, 1] * dy + M[d, 2, 2] * dz

            sqx = rd0 * rd0
            sqy = rd1 * rd1
            sqz = rd2 * rd2
            denom = (sqx + sqy) * (sqx + sqy) + alpha
            nyf = 2.0 * ny * (sqx + sqy)
            c = cos_rot[d]
            s = sin_rot[d]
            denom1 = sqz + sqy + alpha

            # Pure-field gradient G (= Fdd - I); only 5 entries are nonzero.
            G[0, 0] = -rd1 * (3.0 * sqx + sqy - nyf) / denom * bf * c
            G[0, 1] = rd0 * (3.0 * sqx + sqy - nyf) / denom * bf * c + (-rd2 / denom1) * bf1 * s
            G[0, 2] = (rd1 / denom1) * bf1 * s
            G[1, 0] = -rd0 * (3.0 * sqy + sqx - nyf) / denom * bf * c
            G[1, 1] = rd1 * (sqx - sqy + nyf) / denom * bf * c
            G[1, 2] = 0.0
            G[2, 0] = 0.0
            G[2, 1] = 0.0
            G[2, 2] = 0.0

            # Tmp = G @ Ud_d.T  ; Tmp[a,col] = sum_j G[a,j] * Ud[d,col,j]
            for a in range(3):
                for col in range(3):
                    acc = 0.0
                    for j in range(3):
                        acc += G[a, j] * Ud[d, col, j]
                    Tmp[a, col] = acc

            # contribution = Ud_d @ Tmp ; accumulate into Fg
            f00 += Ud[d, 0, 0] * Tmp[0, 0] + Ud[d, 0, 1] * Tmp[1, 0] + Ud[d, 0, 2] * Tmp[2, 0]
            f01 += Ud[d, 0, 0] * Tmp[0, 1] + Ud[d, 0, 1] * Tmp[1, 1] + Ud[d, 0, 2] * Tmp[2, 1]
            f02 += Ud[d, 0, 0] * Tmp[0, 2] + Ud[d, 0, 1] * Tmp[1, 2] + Ud[d, 0, 2] * Tmp[2, 2]
            f10 += Ud[d, 1, 0] * Tmp[0, 0] + Ud[d, 1, 1] * Tmp[1, 0] + Ud[d, 1, 2] * Tmp[2, 0]
            f11 += Ud[d, 1, 0] * Tmp[0, 1] + Ud[d, 1, 1] * Tmp[1, 1] + Ud[d, 1, 2] * Tmp[2, 1]
            f12 += Ud[d, 1, 0] * Tmp[0, 2] + Ud[d, 1, 1] * Tmp[1, 2] + Ud[d, 1, 2] * Tmp[2, 2]
            f20 += Ud[d, 2, 0] * Tmp[0, 0] + Ud[d, 2, 1] * Tmp[1, 0] + Ud[d, 2, 2] * Tmp[2, 0]
            f21 += Ud[d, 2, 0] * Tmp[0, 1] + Ud[d, 2, 1] * Tmp[1, 1] + Ud[d, 2, 2] * Tmp[2, 1]
            f22 += Ud[d, 2, 0] * Tmp[0, 2] + Ud[d, 2, 1] * Tmp[1, 2] + Ud[d, 2, 2] * Tmp[2, 2]

        # Analytic 3x3 inverse of Fg (mirrors fast_inverse2), then Hg = inv.T - I.
        c00 = f11 * f22 - f12 * f21
        c10 = -(f10 * f22 - f12 * f20)
        c20 = f10 * f21 - f11 * f20
        idet = 1.0 / (f00 * c00 + f01 * c10 + f02 * c20)
        i00 = c00 * idet
        i01 = -idet * (f01 * f22 - f02 * f21)
        i02 = idet * (f01 * f12 - f02 * f11)
        i10 = c10 * idet
        i11 = idet * (f00 * f22 - f02 * f20)
        i12 = -idet * (f00 * f12 - f02 * f10)
        i20 = c20 * idet
        i21 = -idet * (f00 * f21 - f01 * f20)
        i22 = idet * (f00 * f11 - f01 * f10)

        # Hg = transpose(inv) - I  =>  Hg[i,j] = inv[j,i] - (i==j)
        Hg_out[x, 0, 0] = i00 - 1.0
        Hg_out[x, 0, 1] = i10
        Hg_out[x, 0, 2] = i20
        Hg_out[x, 1, 0] = i01
        Hg_out[x, 1, 1] = i11 - 1.0
        Hg_out[x, 1, 2] = i21
        Hg_out[x, 2, 0] = i02
        Hg_out[x, 2, 1] = i12
        Hg_out[x, 2, 2] = i22 - 1.0


def find_hg_population(
    rl_um: np.ndarray,
    M: np.ndarray,
    offset: np.ndarray,
    Ud: np.ndarray,
    cos_rot: np.ndarray,
    sin_rot: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
    """NumPy-facing wrapper around ``_population_hg_kernel``.

    Allocates the (X, 3, 3) output and ensures C-contiguous float64 inputs the
    kernel expects. ``rl_um`` is the lab-frame ray grid in MICROMETRES.

    N=0 divergence: for an empty population (N=0) this kernel returns Hg=0
    (Fg=I), whereas the NumPy ``Fd_find_multi_dislocs_mixed`` path raises
    ValueError. N=0 is not reachable from ``build_dislocation_population``
    (always >=1 dislocation), so the two paths are equivalent for all real
    inputs.
    """
    X = rl_um.shape[1]
    Hg_out = np.empty((X, 3, 3), dtype=np.float64)
    _population_hg_kernel(
        np.ascontiguousarray(rl_um, dtype=np.float64),
        np.ascontiguousarray(M, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(Ud, dtype=np.float64),
        np.ascontiguousarray(cos_rot, dtype=np.float64),
        np.ascontiguousarray(sin_rot, dtype=np.float64),
        float(b),
        float(ny),
        Hg_out,
    )
    return Hg_out
