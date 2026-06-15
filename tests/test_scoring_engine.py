# tests/test_scoring_engine.py
import numpy as np

from dfxm_geo.scoring import engine


def test_preprocess_subtracts_background_and_normalizes():
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 5.0, size=(32, 32))
    img[10:14, 10:14] += 200.0  # a bright feature
    pp = engine.preprocess(img, k=2.0)
    assert pp.min() >= 0.0  # negatives clipped
    assert pp[11, 11] > 0.0  # feature survives
    assert pp[0, 0] == 0.0  # background floor removed
    assert abs(pp.std() - 1.0) < 0.5  # roughly unit-scaled


def test_preprocess_flat_image_is_zero():
    img = np.full((8, 8), 7.0)
    pp = engine.preprocess(img)
    assert np.all(pp == 0.0)  # zero std -> all-zero, no divide-by-zero
