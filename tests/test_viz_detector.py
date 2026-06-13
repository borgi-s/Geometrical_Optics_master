"""Tests for the weak-beam display reduction."""

import numpy as np

from dfxm_geo.viz.detector import subtract_background


def test_subtract_background_floors_at_zero_and_keeps_peaks():
    rng = np.random.default_rng(0)
    img = rng.normal(110.0, 3.0, size=(64, 64))  # pure floor
    img[10, 10] = 5000.0  # one bright dislocation-like pixel
    out = subtract_background(img, k=2.0)
    assert out.min() == 0.0
    # the bright feature survives, reduced by roughly mean + 2 sigma
    thr = img.mean() + 2.0 * img.std()
    assert np.isclose(out[10, 10], 5000.0 - thr)
    # almost all floor pixels are zeroed (one-sided 2 sigma keeps ~2 %)
    assert (out == 0.0).mean() > 0.9


def test_subtract_background_does_not_mutate_input_and_handles_stacks():
    img = np.full((3, 8, 8), 100.0)
    img[:, 4, 4] = 200.0
    before = img.copy()
    out = subtract_background(img)
    assert np.array_equal(img, before)
    assert out.shape == img.shape
