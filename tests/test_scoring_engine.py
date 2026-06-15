# tests/test_scoring_engine.py
import numpy as np
import pytest

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


def test_cross_correlation_self_is_max():
    rng = np.random.default_rng(1)
    a = rng.normal(50, 10, size=(40, 40))
    a[5:9, 20:24] += 150.0
    self_score = engine.cross_correlation_peak(a, a, normalize="symmetric")
    assert abs(self_score - 1.0) < 1e-9


def test_cross_correlation_translation_invariant():
    rng = np.random.default_rng(2)
    a = rng.normal(50, 10, size=(40, 40))
    a[5:9, 20:24] += 150.0
    b = np.roll(a, shift=(7, -5), axis=(0, 1))  # shifted copy
    score = engine.cross_correlation_peak(a, b, normalize="symmetric")
    assert score > 0.95  # circular xcorr is shift-invariant


def test_cross_correlation_dissimilar_is_low():
    rng = np.random.default_rng(3)
    a = np.zeros((40, 40))
    a[5:9, 20:24] = 100.0  # compact blob
    b = rng.normal(0.0, 10.0, size=(40, 40))  # structureless noise: no coherent feature
    score = engine.cross_correlation_peak(a, b, normalize="symmetric")
    assert 0.0 <= score < 0.6


def test_resolve_backend_torch_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert engine._resolve_backend("auto") == "numpy"
    with pytest.raises(RuntimeError):
        engine._resolve_backend("torch")
