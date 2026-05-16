"""Equivalence tests for the bincount-based scatter accumulator in forward().

forward() used to call ``np.add.at(im_1, tuple(indices.T), pro)`` to splat
the per-ray probability into the (chi, phi)-rocking image. Phase 8 replaced
that with ``np.bincount(_flat_indices[idx], weights=prob_f32, ...)``. These
tests guard the equivalence on representative synthetic data so a future
edit can't silently regress the algorithm.

Both accumulators visit elements in input-array order, so for the same
inputs they produce bit-identical float64 outputs. The tests below assert
exact equality, not allclose — any drift here would be a real bug.
"""

from __future__ import annotations

import numpy as np


def _bincount_scatter(im: np.ndarray, flat_indices: np.ndarray, weights: np.ndarray) -> None:
    """The accumulator forward() uses. Mirrored here for the equivalence check."""
    contribution = np.bincount(flat_indices, weights=weights, minlength=im.size)
    im += contribution.reshape(im.shape)


def _np_add_at_scatter(im: np.ndarray, flat_indices: np.ndarray, weights: np.ndarray) -> None:
    """The reference (pre-Phase 8) accumulator. The two must match exactly."""
    rows, cols = np.divmod(flat_indices, im.shape[1])
    np.add.at(im, (rows, cols), weights)


def test_bincount_scatter_matches_np_add_at_on_production_size() -> None:
    """At realistic detector size and valid-ray count, the two scatter
    accumulators produce bit-identical float64 output."""
    rng = np.random.default_rng(20260516)
    H, W = 510, 170  # production detector after Nsub=2 on NN1=340, NN2=1020
    n_valid = 200_000  # plausible idx.sum() in real runs
    flat_indices = rng.integers(0, H * W, size=n_valid, dtype=np.int64)
    weights = rng.standard_normal(n_valid).astype(np.float32)

    im_ref = np.zeros((H, W))
    _np_add_at_scatter(im_ref, flat_indices, weights)

    im_new = np.zeros((H, W))
    _bincount_scatter(im_new, flat_indices, weights)

    np.testing.assert_array_equal(im_new, im_ref)


def test_bincount_scatter_handles_duplicate_indices() -> None:
    """Many writes to the same cell (Nsub>1 collapses sub-pixels) must
    accumulate identically under both algorithms."""
    rng = np.random.default_rng(0)
    H, W = 10, 5  # only 50 cells; 1000 writes => heavy duplication
    flat_indices = rng.integers(0, H * W, size=1000, dtype=np.int64)
    weights = rng.standard_normal(1000).astype(np.float32)

    im_ref = np.zeros((H, W))
    _np_add_at_scatter(im_ref, flat_indices, weights)

    im_new = np.zeros((H, W))
    _bincount_scatter(im_new, flat_indices, weights)

    np.testing.assert_array_equal(im_new, im_ref)


def test_bincount_scatter_empty_input() -> None:
    """Zero valid rays (e.g. a (phi, chi) cell where every qi falls outside
    the Resq_i grid) must produce an all-zero contribution, not crash."""
    H, W = 510, 170
    flat_indices = np.array([], dtype=np.int64)
    weights = np.array([], dtype=np.float32)

    im = np.zeros((H, W))
    _bincount_scatter(im, flat_indices, weights)
    assert np.all(im == 0.0)


def test_forward_model_flat_indices_match_indices_2d() -> None:
    """The precomputed _flat_indices module constant is just C-order
    flattening of the existing (ZI, YI) `indices` array."""
    import dfxm_geo.direct_space.forward_model as fm

    num_cols = fm.NN1 // fm.Nsub
    expected = fm.indices[:, 0].astype(np.int64) * num_cols + fm.indices[:, 1].astype(np.int64)
    np.testing.assert_array_equal(fm._flat_indices, expected)
