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
    H, W = 510, 170  # production detector: NN2//Nsub x NN1//Nsub, Nsub-invariant
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


def test_mc_lut_scatter_matches_numpy_chain() -> None:
    """The numba _mc_lut_scatter is bit-identical to the numpy chain it
    replaced: 3x np.floor().astype(int16) -> in-bounds mask -> Resq_i gather ->
    np.bincount(weights=float32). Exercises in-bounds rays plus every flavour
    of out-of-bounds (negative index and index >= npoints on each axis), which
    must be dropped exactly as the bool mask dropped them.
    """
    import dfxm_geo.direct_space.forward_model as fm

    rng = np.random.default_rng(20260527)
    np1, np2, np3 = 7, 5, 4
    Resq = rng.standard_normal((np1, np2, np3))  # synthetic float64 LUT
    H, W = 6, 3

    qi1_start, qi1_step = -0.5, 0.1
    qi2_start, qi2_step = -0.3, 0.07
    qi3_start, qi3_step = -0.2, 0.05
    # Draw finite qi spanning well past the grid on both ends so a healthy
    # fraction of rays are rejected (index < 0 and index >= npoints).
    n_rays = 8000
    qi = np.empty((3, n_rays))
    qi[0] = rng.uniform(qi1_start - 0.3, qi1_start + np1 * qi1_step + 0.3, n_rays)
    qi[1] = rng.uniform(qi2_start - 0.3, qi2_start + np2 * qi2_step + 0.3, n_rays)
    qi[2] = rng.uniform(qi3_start - 0.3, qi3_start + np3 * qi3_step + 0.3, n_rays)
    prob_z = rng.uniform(0.1, 1.0, n_rays)
    flat_indices = rng.integers(0, H * W, size=n_rays, dtype=np.int64)

    # numpy reference: the exact chain forward_from_static used to run.
    i1 = np.floor((qi[0] - qi1_start) / qi1_step).astype(np.int16)
    i2 = np.floor((qi[1] - qi2_start) / qi2_step).astype(np.int16)
    i3 = np.floor((qi[2] - qi3_start) / qi3_step).astype(np.int16)
    idx = (i1 >= 0) * (i2 >= 0) * (i3 >= 0) * (i1 < np1) * (i2 < np2) * (i3 < np3)
    assert 0 < idx.sum() < n_rays, "test should have both in- and out-of-bounds rays"
    prob = Resq[i1[idx], i2[idx], i3[idx]] * prob_z[idx]
    ref = np.bincount(flat_indices[idx], weights=prob.astype(np.float32), minlength=H * W).reshape(
        H, W
    )

    # numba kernel scatters directly into a flat image view.
    im_flat = np.zeros(H * W)
    fm._mc_lut_scatter(
        qi,
        prob_z,
        flat_indices,
        np.ascontiguousarray(Resq.reshape(-1)),
        im_flat,
        qi1_start,
        qi1_step,
        qi2_start,
        qi2_step,
        qi3_start,
        qi3_step,
        np1,
        np2,
        np3,
        np2 * np3,
        np3,
    )
    np.testing.assert_array_equal(im_flat.reshape(H, W), ref)


def test_forward_model_flat_indices_match_indices_2d() -> None:
    """The precomputed _flat_indices module constant is just C-order
    flattening of the existing (ZI, YI) `indices` array."""
    import dfxm_geo.direct_space.forward_model as fm

    num_cols = fm.NN1 // fm.Nsub
    expected = fm.indices[:, 0].astype(np.int64) * num_cols + fm.indices[:, 1].astype(np.int64)
    np.testing.assert_array_equal(fm._flat_indices, expected)
