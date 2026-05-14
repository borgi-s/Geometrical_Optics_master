"""Sample-remount rotation matrices (Purdue 2024 paper).

S_i is applied between the sample frame (after Theta) and the crystal frame
(before Us.T) in the Fd_find rotation chain. Operationally: S rotates the
entire sample relative to the goniometer, simulating physical remounting at
a symmetry-equivalent orientation.

S1 = identity (no remount); S2, S3, S4 are three specific cubic-symmetry
proper rotations from the Purdue 2024 paper. They are ported verbatim from
the branch source; their numerical traces are not all equal — S2 and S4
have trace 1/3 (~109.47 deg rotations) and S3 has trace 5/3 (~70.53 deg)
— so they are NOT three rotations about a single axis; they are independent
group elements selected for the paper's specific remount scenarios.
"""

import numpy as np

S1: np.ndarray = np.identity(3)

S2: np.ndarray = np.array(
    [
        [1 / 3, -2 / 3, -2 / 3],
        [2 / 3, -1 / 3, 2 / 3],
        [-2 / 3, -2 / 3, 1 / 3],
    ]
)

S3: np.ndarray = np.array(
    [
        [1 / 3, -2 / 3, 2 / 3],
        [2 / 3, 2 / 3, 1 / 3],
        [-2 / 3, 1 / 3, 2 / 3],
    ]
)

S4: np.ndarray = np.array(
    [
        [1 / 3, 2 / 3, 2 / 3],
        [2 / 3, 1 / 3, -2 / 3],
        [-2 / 3, 2 / 3, -1 / 3],
    ]
)

SAMPLE_REMOUNT_OPTIONS: dict[str, np.ndarray] = {
    "S1": S1,
    "S2": S2,
    "S3": S3,
    "S4": S4,
}
