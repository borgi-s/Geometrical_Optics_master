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
        rd_new[1] -= i * dis

        sqx = rd_new[0] * rd_new[0]
        sqy = rd_new[1] * rd_new[1]
        denom = (sqx + sqy) * (sqx + sqy)
        nyfactor = 2 * ny * (sqx + sqy)

        Fdd_chunk[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 0, 1] += rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
        Fdd_chunk[:, 1, 1] += rd_new[1] * (sqx - sqy - nyfactor) / denom
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
    # Lazy import to avoid a circular-import risk during package init.
    from dfxm_geo.crystal.rotations import rotatedU

    if misorientation:
        m_ori = b / (dis * 1e-6)
        U_sr = rotatedU(t_vec, m_ori / 2, Us, 1)
        U_sl = rotatedU(t_vec, -m_ori / 2, Us, 1)

        rs = Theta @ rl
        left_half1 = (rs[0] < 0) & (rs[1] > 0)
        right_half1 = (rs[0] >= 0) & (rs[1] > 0)
        rc = U_sl.T @ rs[:, left_half1] & U_sr.T @ rs[:, right_half1]
        rd = Ud.T @ rc

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

    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] = rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] = rd[1] * (sqx - sqy - nyfactor) / denom

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
            Fdd[:, 1, 1] += rd_new[1] * (sqx - sqy - nyfactor) / denom

    Fdd *= bfactor
    Fdd += np.identity(3)
    return Ud @ Fdd @ Ud.T
