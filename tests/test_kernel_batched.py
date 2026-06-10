"""Chunked (low-peak-memory) kernel generation.

`generate_kernel(batch_size=...)` processes the Monte Carlo rays in batches,
accumulating into the shared Resq_i histogram, so peak memory is ~O(batch_size)
instead of ~O(Nrays). This lets the full Nrays=1e8 bootstrap run on a
memory-constrained laptop.

Correctness contract:
- A single batch (batch_size >= Nrays) reproduces the unbatched kernel
  bit-for-bit (same RNG stream, same draw order).
- Multi-batch generation is deterministic for a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dfxm_geo.reciprocal_space.kernel import generate_kernel

# Small, fast settings: 20k rays on a coarse grid. beamstop=False keeps the
# RNG stream identical between the batched and unbatched paths (no wire draws).
_COMMON = dict(
    Nrays=20_000,
    npoints1=50,
    npoints2=40,
    npoints3=40,
    seed=0,
    beamstop=False,
)


def test_single_batch_matches_unbatched(tmp_path: Path) -> None:
    """batch_size >= Nrays must give a bit-identical kernel to the unbatched path."""
    p_un = generate_kernel(output_path=tmp_path / "unbatched.npz", **_COMMON)
    p_b = generate_kernel(output_path=tmp_path / "batched.npz", batch_size=20_000, **_COMMON)
    un = np.load(p_un)["Resq_i"]
    ba = np.load(p_b)["Resq_i"]
    np.testing.assert_array_equal(un, ba)


def test_batched_is_deterministic(tmp_path: Path) -> None:
    """Multi-batch generation at a fixed seed is reproducible, and a real kernel."""
    p1 = generate_kernel(output_path=tmp_path / "a.npz", batch_size=5_000, **_COMMON)
    p2 = generate_kernel(output_path=tmp_path / "b.npz", batch_size=5_000, **_COMMON)
    k1 = np.load(p1)["Resq_i"]
    k2 = np.load(p2)["Resq_i"]
    np.testing.assert_array_equal(k1, k2)
    # sanity: a normalised, non-degenerate kernel
    assert k1.max() == 1.0
    assert np.count_nonzero(k1) > 0
