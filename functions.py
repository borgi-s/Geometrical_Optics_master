"""DEPRECATED — re-exports from `dfxm_geo.crystal.*` and `dfxm_geo.io`.

This module is kept as a compatibility shim during Phase 4 of the
cleanup. New code should import directly from:

    dfxm_geo.crystal.dislocations   # Fd_find, multi_dislocs_parallel
    dfxm_geo.crystal.rotations      # rotatedU, fast_inverse2
    dfxm_geo.io                     # check_folder
    dfxm_geo.io.strain_cache        # load_or_generate_Hg

The dead helpers `square`, `m_norm`, `repeat` (zero call sites in the
repo) were removed during the Phase 4.3 move. Recover from git history
if you need them.
"""

from dfxm_geo.crystal.dislocations import Fd_find, multi_dislocs_parallel
from dfxm_geo.crystal.rotations import fast_inverse2, rotatedU
from dfxm_geo.io import check_folder
from dfxm_geo.io.strain_cache import load_or_generate_Hg

__all__ = [
    "Fd_find",
    "check_folder",
    "fast_inverse2",
    "load_or_generate_Hg",
    "multi_dislocs_parallel",
    "rotatedU",
]
