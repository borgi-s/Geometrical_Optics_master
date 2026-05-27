# scripts/profile_find_hg.py
"""Profile Find_Hg_from_population in isolation (Phase 1 baseline).

Runs the random_dislocations population path used by configs/profile_rocking_random.toml
(ndis=4) and dumps a cProfile breakdown + a wall-clock median over repeats.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import CrystalConfig, RandomDislocationsConfig


def _make_population(ndis: int = 4, seed: int = 42):
    crystal = CrystalConfig(
        mode="random_dislocations",
        random_dislocations=RandomDislocationsConfig(
            ndis=ndis, sigma=5.0, min_distance=4.0, seed=seed
        ),
    )
    rng = np.random.default_rng(seed)
    return fm.build_dislocation_population(crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=rng)


def main() -> None:
    pop = _make_population()
    # Warm up (numba/cache, allocation pools) before timing.
    fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)

    repeats = 5
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
        times.append(time.perf_counter() - t0)
    print(f"median wall: {np.median(times) * 1e3:.1f} ms  (n={repeats})")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
