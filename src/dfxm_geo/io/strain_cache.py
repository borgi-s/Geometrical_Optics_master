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

    If `file_path` is given and the file exists *and* its leading dimension
    matches `rl.shape[1]`, load Fg from it. A shape mismatch (e.g. cache was
    written under a different detector ray grid) regenerates Fg and overwrites
    the cache rather than silently corrupting the result. Hg is derived from
    Fg by bulk inversion + transpose.
    """
    expected_n = rl.shape[1]
    Fg: np.ndarray | None = None

    if file_path is not None:
        fname = file_path.rsplit("/", 1)[-1]
        try:
            candidate = np.load(file_path)
        except FileNotFoundError:
            print(f"File '{fname}' not found. \nGenerating a new Fg array.")
        else:
            if candidate.shape[0] != expected_n:
                print(
                    f"Cached Fg in {fname} has shape[0]={candidate.shape[0]} "
                    f"but rl needs {expected_n} rays; regenerating."
                )
            else:
                print(f"Loaded Fg from {fname}")
                Fg = candidate

    if Fg is None:
        if ndis == 0:
            Fg = np.zeros([expected_n, 3, 3])
            Fg += np.identity(Fg.shape[1])
        else:
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)
        if file_path is not None:
            np.save(file_path, Fg)
            print(f"Saved Fg to {file_path.rsplit('/', 1)[-1]}")

    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)

    return Hg
