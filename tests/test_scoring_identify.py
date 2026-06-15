# tests/test_scoring_identify.py
import numpy as np

from dfxm_geo.scoring.identify import IdentifiabilityResult, Identifier, RankedMatch
from dfxm_geo.scoring.types import CandidateLabel, CandidateLibrary, GridSpec


def _lib():
    rng = np.random.default_rng(7)
    base = rng.normal(40, 6, size=(20, 20))
    f0 = base.copy()
    f0[3:6, 3:6] += 120
    f1 = base.copy()
    f1[14:17, 14:17] += 120
    frames = np.stack([f0, f1]).astype(np.float32)
    labels = [
        CandidateLabel((1, 1, 1), (1, 0, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 1, "m.h5"),
        CandidateLabel((1, 1, 1), (0, 1, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 2, "m.h5"),
    ]
    return CandidateLibrary(frames, labels, GridSpec((0.1, 0.1), (20, 20)))


def test_rank_returns_sorted_matches():
    lib = _lib()
    ident = Identifier(lib, backend="numpy")
    target = lib.frames[1].copy()
    matches = ident.rank(target, target_pitch_um=(0.1, 0.1), top_k=2)
    assert isinstance(matches[0], RankedMatch)
    assert matches[0].label.burgers == (0, 1, 1)  # best match = frame 1
    assert matches[0].score >= matches[1].score
    assert matches[0].scan_index == 2


def test_rank_resamples_off_grid_target():
    lib = _lib()
    ident = Identifier(lib, backend="numpy")
    # target at 2x coarser pitch but same physical feature as frame 0
    coarse = lib.frames[0][::2, ::2].copy()
    matches = ident.rank(coarse, target_pitch_um=(0.2, 0.2), top_k=1)
    assert matches[0].label.burgers == (1, 0, 1)


def test_study_distinct_classes_top1_perfect(tmp_path):
    lib = _lib()  # two clearly different frames/classes
    ident = Identifier(lib, backend="numpy")
    res = ident.study()
    assert isinstance(res, IdentifiabilityResult)
    assert res.matrix.shape == (2, 2)
    assert res.top1_accuracy == 1.0  # each recovers its own class (LOO)
    res.save(tmp_path)
    assert (tmp_path / "matrix.npy").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "labels.json").exists()


def test_study_degenerate_classes_confused():
    rng = np.random.default_rng(8)
    base = rng.normal(40, 6, size=(20, 20))
    f = base.copy()
    f[3:6, 3:6] += 120
    frames = np.stack([f, f.copy(), np.roll(f, (9, 9), axis=(0, 1))]).astype(np.float32)
    from dfxm_geo.scoring.types import CandidateLabel, CandidateLibrary, GridSpec

    labels = [
        CandidateLabel((1, 1, 1), (1, 0, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 1, "m"),
        CandidateLabel(
            (1, 1, 1), (0, 1, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 2, "m"
        ),  # identical frame, diff class
        CandidateLabel((1, 1, 1), (1, 1, 0), 0.0, 0.8, True, (0.0, 2.0, 0.0), 3, "m"),
    ]
    lib = CandidateLibrary(frames, labels, GridSpec((0.1, 0.1), (20, 20)))
    res = Identifier(lib, backend="numpy").study()
    assert res.top1_accuracy < 1.0  # the two identical frames confuse
