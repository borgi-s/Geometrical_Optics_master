"""Strain-field (Hg) loading and caching utilities."""

import numpy as np

from dfxm_geo.crystal.dislocations import Fd_find
from dfxm_geo.crystal.rotations import fast_inverse2


def load_or_generate_Hg(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float,
    ndis: int,
    file_path: str | None = None,
) -> np.ndarray:
    """Return the displacement gradient field Hg, loading from disk if cached.

    If `file_path` is given and the file exists, load Fg from it. Otherwise
    compute Fg via `Fd_find` (or identity if ndis == 0) and optionally save
    it to `file_path`. Hg is derived from Fg by bulk inversion + transpose.
    """
    if file_path is not None:
        fname = file_path.rsplit("/", 1)[-1]
        try:
            Fg = np.load(file_path)
            print(f"Loaded Fg from {fname}")
        except FileNotFoundError:
            print(f"File '{fname}' not found. \nGenerating a new Fg array.")
            if ndis == 0:
                Fg = np.zeros([len(rl[-1]), 3, 3])
                Fg += np.identity(Fg.shape[1])
            else:
                Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)

            if file_path is not None:
                np.save(file_path, Fg)
                print(f"Saved Fg to {fname}")
    else:
        if ndis == 0:
            Fg = np.zeros([len(rl[-1]), 3, 3])
            Fg += np.identity(Fg.shape[1])
        else:
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)

    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)

    return Hg
