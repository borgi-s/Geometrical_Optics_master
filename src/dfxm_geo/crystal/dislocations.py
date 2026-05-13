"""Dislocation displacement-gradient fields (edge dislocations).

Computes the Fdd / Fg tensors used by the DFXM forward model for
crystals containing one or more edge dislocations arranged in a wall.

Public functions:
    Fd_find — main entry: returns Fg given lab-frame coords + rotations.
    multi_dislocs_parallel — internal worker for parallel ndis>100 path.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from joblib import cpu_count
from numba import jit  # noqa: F401  (kept for parity with legacy module)
from tqdm import tqdm

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO


def multi_dislocs_parallel(chunk, rd, Fdd_shape, dis, ny: float = POISSON_RATIO):
    """Worker for Fd_find's parallel branch (ndis > 100).

    Args:
        chunk: A chunk of dislocation indices to be processed.
        rd: An array containing x/y/z coordinates in dislocation space.
        Fdd_shape: Shape of the Fdd tensor to be accumulated.
        dis: Spacing between dislocations in the wall (µm).
        ny: Poisson ratio. Default = POISSON_RATIO.

    Returns:
        ndarray: Fdd chunk with the accumulated tensor contributions.
    """
    Fdd_chunk = np.zeros(Fdd_shape)
    for i in tqdm(chunk, desc=f"Running: {chunk}"):
        rd_new = np.copy(rd[:2])
        # Bipolar wall pattern matching the sequential branch in Fd_find:
        #   odd i  (1, 3, 5, ...)  → walls at +1*dis, +2*dis, +3*dis, ...
        #   even i (2, 4, 6, ...)  → walls at -1*dis, -2*dis, -3*dis, ...
        # Prior to 2026-05-12 this loop did `rd_new[1] -= i * dis`, producing a
        # one-sided wall that disagreed with the sequential branch by ~22% in
        # Fdd Frobenius norm. Production sims with ndis>100 hit this path, so
        # any reference results predating the fix were generated with a
        # one-sided wall configuration.
        if i % 2 == 1:
            rd_new[1] += ((i + 1) // 2) * dis
        else:
            rd_new[1] -= (i // 2) * dis

        sqx = rd_new[0] * rd_new[0]
        sqy = rd_new[1] * rd_new[1]
        denom = (sqx + sqy) * (sqx + sqy)
        nyfactor = 2 * ny * (sqx + sqy)

        Fdd_chunk[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 0, 1] += rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
        Fdd_chunk[:, 1, 1] += rd_new[1] * (sqx - sqy + nyfactor) / denom
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
    rc = Us.T @ rs
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
        print(chunks)

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
    elif ndis <= 100:
        count1, count2 = 1, 1
        for i in tqdm(range(1, ndis)):
            rd_new = np.copy(rd[:2])

            if i % 2 == 0:
                rd_new[1] -= count1 * dis
                count1 += 1

            if i % 2 == 1:
                rd_new[1] += count2 * dis
                count2 += 1

            sqx = rd_new[0] * rd_new[0]
            sqy = rd_new[1] * rd_new[1]
            denom = (sqx + sqy) * (sqx + sqy)
            nyfactor = 2 * ny * (sqx + sqy)

            Fdd[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 0, 1] += rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
            Fdd[:, 1, 1] += rd_new[1] * (sqx - sqy + nyfactor) / denom

    Fdd *= bfactor
    Fdd += np.identity(3)
    return Ud @ Fdd @ Ud.T
