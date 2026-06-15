# Dislocation cross-correlation scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `dfxm_geo.scoring` subpackage that ranks DFXM contrast images against a forward-simulated candidate library by normalized FFT cross-correlation, exposing an all-pairs identifiability study and a single-target ranking query.

**Architecture:** A pure-array cross-correlation engine (numpy default, optional torch) sits under an HDF5 identify-output loader and a physical-resampling target loader; a high-level `Identifier` composes them into the study and ranking APIs. Files are split by responsibility: `types`, `engine`, `library`, `target`, `identify`.

**Tech Stack:** Python, numpy, scipy (`scipy.fft`, `scipy.ndimage.zoom`), h5py, matplotlib (already package deps); torch optional.

Spec: `docs/superpowers/specs/2026-06-15-disloc-cross-corr-scorer-design.md`

All commits end with the repo's `Co-Authored-By` trailer (omitted from the commands below for brevity).

---

## Prerequisites: isolated worktree environment

> **Why:** the shared `GM-reworked/.venv` is editable-installed against the main
> checkout, which a parallel session (`fix/m4-post43-followups`) is using. Running
> `pip install -e .` from this worktree against that shared venv would repoint it
> and break the other session. Create a worktree-local venv instead.

- [ ] **Step 1: Create and populate an isolated venv in the worktree**

Run (from the worktree root `wt-disloc-scorer/`):
```bash
py -3 -m venv .venv-wt
./.venv-wt/Scripts/python.exe -m pip install -U pip
./.venv-wt/Scripts/python.exe -m pip install -e ".[dev]"
```
Expected: editable install succeeds; `./.venv-wt/Scripts/python.exe -c "import dfxm_geo, h5py, scipy, numpy"` prints nothing and exits 0.

- [ ] **Step 2: Baseline sanity (scoped, not the full suite)**

Run: `./.venv-wt/Scripts/python.exe -m pytest tests/ -q -k "io_hdf5 or version" -x`
Expected: PASS (a quick smoke that the worktree imports and a small slice of the suite is green). The full suite is not required here; new tests are added per task. If you run the full suite, first `rm -f direct_space/deformation_gradient_tensors/Fg_*.npy` (stale-cache flakiness).

In every test command below, `PY` means `./.venv-wt/Scripts/python.exe`.

---

## File structure

- Create `src/dfxm_geo/scoring/__init__.py` — public API re-exports
- Create `src/dfxm_geo/scoring/types.py` — `GridSpec`, `CandidateLabel`, `CandidateLibrary`
- Create `src/dfxm_geo/scoring/engine.py` — preprocessing, FFT cross-corr, `score_matrix`, `score_target`
- Create `src/dfxm_geo/scoring/library.py` — `load_library`
- Create `src/dfxm_geo/scoring/target.py` — `resample_to_grid`
- Create `src/dfxm_geo/scoring/identify.py` — `Identifier`, `IdentifiabilityResult`, `RankedMatch`
- Create `scripts/run_identify_study.py` — thin runner
- Create `tests/test_scoring_types.py`, `tests/test_scoring_engine.py`, `tests/test_scoring_target.py`, `tests/test_scoring_library.py`, `tests/test_scoring_identify.py`

---

## Task 1: Package skeleton + data types

**Files:**
- Create: `src/dfxm_geo/scoring/__init__.py`
- Create: `src/dfxm_geo/scoring/types.py`
- Test: `tests/test_scoring_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_types.py
import numpy as np
import pytest
from dfxm_geo.scoring.types import GridSpec, CandidateLabel, CandidateLibrary


def _label(plane=(1, 1, 1), b=(1, 0, 1), alpha=0.0):
    return CandidateLabel(
        slip_plane_normal=plane, burgers=b, rotation_deg=alpha,
        gb_cos=0.8, gb_visible=True, q_hkl=(0.0, 2.0, 0.0),
        scan_index=1, source_file="m.h5",
    )


def test_class_key_modes():
    lbl = _label(plane=(1, 1, 1), b=(1, 0, 1), alpha=30.0)
    assert lbl.class_key("plane_burgers") == ((1, 1, 1), (1, 0, 1))
    assert lbl.class_key("burgers") == ((1, 0, 1),)
    assert lbl.class_key("plane_burgers_alpha") == ((1, 1, 1), (1, 0, 1), 30.0)
    assert hash(lbl.class_key("plane_burgers"))  # hashable -> usable as dict key


def test_class_key_unknown_mode():
    with pytest.raises(ValueError):
        _label().class_key("nonsense")


def test_library_len():
    frames = np.zeros((2, 4, 4), dtype=np.float32)
    lib = CandidateLibrary(frames=frames, labels=[_label(), _label()],
                           grid=GridSpec(pitch_um=(0.1, 0.1), shape=(4, 4)))
    assert len(lib) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfxm_geo.scoring'`

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/__init__.py
"""Cross-correlation dislocation identification scorer."""
from .types import GridSpec, CandidateLabel, CandidateLibrary

__all__ = ["GridSpec", "CandidateLabel", "CandidateLibrary"]
```

```python
# src/dfxm_geo/scoring/types.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Object-plane sampling of a candidate/target image."""

    pitch_um: tuple[float, float]   # (dy, dx) micrometres per pixel
    shape: tuple[int, int]          # (H, W)


@dataclass(frozen=True)
class CandidateLabel:
    slip_plane_normal: tuple[int, int, int]
    burgers: tuple[int, int, int]
    rotation_deg: float
    gb_cos: float
    gb_visible: bool
    q_hkl: tuple[float, float, float]
    scan_index: int
    source_file: str

    def class_key(self, mode: str = "plane_burgers") -> tuple:
        if mode == "plane_burgers":
            return (self.slip_plane_normal, self.burgers)
        if mode == "burgers":
            return (self.burgers,)
        if mode == "plane_burgers_alpha":
            return (self.slip_plane_normal, self.burgers, self.rotation_deg)
        raise ValueError(f"unknown class_key mode: {mode!r}")


@dataclass
class CandidateLibrary:
    frames: np.ndarray              # (N, H, W) float32
    labels: list[CandidateLabel]
    grid: GridSpec

    def __len__(self) -> int:
        return len(self.labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_types.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/__init__.py src/dfxm_geo/scoring/types.py tests/test_scoring_types.py
git commit -m "feat(scoring): package skeleton + GridSpec/CandidateLabel/CandidateLibrary types"
```

---

## Task 2: Engine — preprocessing

**Files:**
- Create: `src/dfxm_geo/scoring/engine.py`
- Test: `tests/test_scoring_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_engine.py
import numpy as np
import pytest
from dfxm_geo.scoring import engine


def test_preprocess_subtracts_background_and_normalizes():
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 5.0, size=(32, 32))
    img[10:14, 10:14] += 200.0          # a bright feature
    pp = engine.preprocess(img, k=2.0)
    assert pp.min() >= 0.0               # negatives clipped
    assert pp[11, 11] > 0.0              # feature survives
    assert pp[0, 0] == 0.0               # background floor removed
    assert abs(pp.std() - 1.0) < 0.5     # roughly unit-scaled


def test_preprocess_flat_image_is_zero():
    img = np.full((8, 8), 7.0)
    pp = engine.preprocess(img)
    assert np.all(pp == 0.0)             # zero std -> all-zero, no divide-by-zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_engine.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'dfxm_geo.scoring.engine'`

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/engine.py
from __future__ import annotations

import numpy as np


def preprocess(img: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Background-subtract (mean + k*std), clip negatives to 0, normalize by std.

    Matches the Borgi 2025 method: the floor uses the raw image statistics, then
    the clipped result is scaled by its own std so different images are comparable.
    """
    arr = np.asarray(img, dtype=np.float64)
    sub = arr - (arr.mean() + k * arr.std())
    sub[sub < 0] = 0.0
    s = sub.std()
    if s == 0.0:
        return sub
    return sub / s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/engine.py tests/test_scoring_engine.py
git commit -m "feat(scoring): engine.preprocess (background subtract + normalize)"
```

---

## Task 3: Engine — single-pair cross-correlation peak + backend resolver

**Files:**
- Modify: `src/dfxm_geo/scoring/engine.py`
- Test: `tests/test_scoring_engine.py`

- [ ] **Step 1: Write the failing test (append)**

```python
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
    b = np.roll(a, shift=(7, -5), axis=(0, 1))   # shifted copy
    score = engine.cross_correlation_peak(a, b, normalize="symmetric")
    assert score > 0.95                          # circular xcorr is shift-invariant


def test_cross_correlation_dissimilar_is_low():
    rng = np.random.default_rng(3)
    a = np.zeros((40, 40)); a[5:9, 20:24] = 100.0
    b = np.zeros((40, 40)); b[30:34, 2:6] = 100.0
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_engine.py -k "cross_correlation or resolve_backend" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'cross_correlation_peak'`

- [ ] **Step 3: Write the implementation (append to engine.py)**

```python
def _resolve_backend(backend: str) -> str:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("backend='torch' requested but torch is not installed") from exc
        return "torch"
    if backend == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "torch"
        except ImportError:
            pass
        return "numpy"
    raise ValueError(f"unknown backend: {backend!r}")


def _peak_from_ffts(fa: np.ndarray, fb: np.ndarray) -> float:
    cc = np.fft.ifft2(fa * np.conj(fb))
    return float(np.abs(cc).max())


def cross_correlation_peak(
    a: np.ndarray, b: np.ndarray, *, normalize: str = "symmetric", k: float = 2.0
) -> float:
    """Translation-invariant similarity of two images (numpy reference path)."""
    fa = np.fft.fft2(preprocess(a, k))
    fb = np.fft.fft2(preprocess(b, k))
    cross = _peak_from_ffts(fa, fb)
    if normalize == "none":
        return cross
    auto_a = _peak_from_ffts(fa, fa)
    auto_b = _peak_from_ffts(fb, fb)
    if normalize == "symmetric":
        denom = np.sqrt(auto_a * auto_b)
        return cross / denom if denom > 0 else 0.0
    if normalize == "diagonal":
        return cross / auto_a if auto_a > 0 else 0.0
    raise ValueError(f"unknown normalize mode: {normalize!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_engine.py -k "cross_correlation or resolve_backend" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/engine.py tests/test_scoring_engine.py
git commit -m "feat(scoring): engine cross_correlation_peak + backend resolver"
```

---

## Task 4: Engine — all-pairs score matrix

**Files:**
- Modify: `src/dfxm_geo/scoring/engine.py`
- Test: `tests/test_scoring_engine.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def _three_frames():
    rng = np.random.default_rng(4)
    base = rng.normal(50, 8, size=(24, 24))
    a = base.copy(); a[5:8, 5:8] += 120
    b = np.roll(a, (3, 2), axis=(0, 1))          # similar to a (shifted)
    c = base.copy(); c[18:21, 18:21] += 120      # different feature
    return np.stack([a, b, c])


def test_score_matrix_symmetric_and_unit_diagonal():
    frames = _three_frames()
    M = engine.score_matrix(frames, normalize="symmetric")
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)                    # symmetric
    assert M[0, 1] > M[0, 2]                      # a~b more similar than a~c


def test_score_matrix_diagonal_mode_rows_self_one():
    frames = _three_frames()
    M = engine.score_matrix(frames, normalize="diagonal")
    assert np.allclose(np.diag(M), 1.0)


def test_score_matrix_none_mode_raw_peaks():
    frames = _three_frames()
    M = engine.score_matrix(frames, normalize="none")
    assert np.allclose(M, M.T)
    assert (M >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_engine.py -k score_matrix -v`
Expected: FAIL with `AttributeError: ... 'score_matrix'`

- [ ] **Step 3: Write the implementation (append to engine.py)**

```python
def _normalize_matrix(C: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return C
    d = np.diag(C).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "symmetric":
            denom = np.sqrt(np.outer(d, d))
            return np.where(denom > 0, C / denom, 0.0)
        if mode == "diagonal":
            return np.where(d[:, None] > 0, C / d[:, None], 0.0)
    raise ValueError(f"unknown normalize mode: {mode!r}")


def score_matrix(
    frames: np.ndarray, *, normalize: str = "symmetric", backend: str = "auto", k: float = 2.0
) -> np.ndarray:
    """All-pairs normalized cross-correlation matrix (N, N)."""
    backend = _resolve_backend(backend)
    pp = np.stack([preprocess(f, k) for f in frames])
    if backend == "torch":
        C = _raw_matrix_torch(pp)
    else:
        C = _raw_matrix_numpy(pp)
    return _normalize_matrix(C, normalize)


def _raw_matrix_numpy(pp: np.ndarray) -> np.ndarray:
    n = pp.shape[0]
    F = np.fft.fft2(pp)                            # (n, H, W) complex
    C = np.zeros((n, n))
    for i in range(n):
        cc = np.fft.ifft2(F[i] * np.conj(F[i:]))   # broadcast i vs j>=i
        peaks = np.abs(cc).reshape(n - i, -1).max(axis=1)
        C[i, i:] = peaks
        C[i:, i] = peaks
    return C
```

(The torch variant `_raw_matrix_torch` is added in Task 13; `score_matrix` only
reaches it when `_resolve_backend` returns `"torch"`, which requires CUDA + torch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_engine.py -k score_matrix -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/engine.py tests/test_scoring_engine.py
git commit -m "feat(scoring): engine.score_matrix (symmetric/diagonal/none, upper-triangle)"
```

---

## Task 5: Engine — single-target scoring

**Files:**
- Modify: `src/dfxm_geo/scoring/engine.py`
- Test: `tests/test_scoring_engine.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def test_score_target_recovers_matching_frame():
    frames = _three_frames()
    target = frames[2].copy()                     # equals candidate 2
    scores = engine.score_target(target, frames, normalize="symmetric")
    assert scores.shape == (3,)
    assert int(np.argmax(scores)) == 2
    assert scores[2] > 0.99


def test_score_target_none_mode_shape():
    frames = _three_frames()
    scores = engine.score_target(frames[0], frames, normalize="none")
    assert scores.shape == (3,)
    assert (scores >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_engine.py -k score_target -v`
Expected: FAIL with `AttributeError: ... 'score_target'`

- [ ] **Step 3: Write the implementation (append to engine.py)**

```python
def score_target(
    target: np.ndarray, frames: np.ndarray, *,
    normalize: str = "symmetric", backend: str = "auto", k: float = 2.0,
) -> np.ndarray:
    """Score one target image against every library frame; returns (N,)."""
    backend = _resolve_backend(backend)            # torch path shares the numpy math here
    pp_t = preprocess(target, k)
    pp = np.stack([preprocess(f, k) for f in frames])
    ft = np.fft.fft2(pp_t)
    F = np.fft.fft2(pp)
    cross = np.abs(np.fft.ifft2(ft[None] * np.conj(F))).reshape(F.shape[0], -1).max(axis=1)
    if normalize == "none":
        return cross
    auto_t = _peak_from_ffts(ft, ft)
    auto_f = np.array([_peak_from_ffts(F[j], F[j]) for j in range(F.shape[0])])
    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize == "symmetric":
            denom = np.sqrt(auto_t * auto_f)
            return np.where(denom > 0, cross / denom, 0.0)
        if normalize == "diagonal":
            return np.where(auto_t > 0, cross / auto_t, 0.0)
    raise ValueError(f"unknown normalize mode: {normalize!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_engine.py -k score_target -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/engine.py tests/test_scoring_engine.py
git commit -m "feat(scoring): engine.score_target (one target vs library)"
```

---

## Task 6: Target — physical resample onto the library grid

**Files:**
- Create: `src/dfxm_geo/scoring/target.py`
- Test: `tests/test_scoring_target.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_target.py
import numpy as np
from dfxm_geo.scoring.target import resample_to_grid, _crop_or_pad
from dfxm_geo.scoring.types import GridSpec


def test_crop_or_pad_crops_centered():
    a = np.arange(36).reshape(6, 6).astype(float)
    out = _crop_or_pad(a, (4, 4))
    assert out.shape == (4, 4)
    assert out[0, 0] == a[1, 1]                    # centered crop


def test_crop_or_pad_pads_centered():
    a = np.ones((2, 2))
    out = _crop_or_pad(a, (4, 4))
    assert out.shape == (4, 4)
    assert out.sum() == 4.0                        # only original ones survive
    assert out[0, 0] == 0.0 and out[1, 1] == 1.0


def test_resample_matches_native_scale():
    # a blob rendered at 2x coarser pitch, resampled to the fine grid, should
    # closely match the natively-fine blob (same physical extent).
    grid = GridSpec(pitch_um=(0.5, 0.5), shape=(40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    fine = np.exp(-(((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 4.0 ** 2)))
    yy2, xx2 = np.mgrid[0:20, 0:20]
    coarse = np.exp(-(((yy2 - 10) ** 2 + (xx2 - 10) ** 2) / (2 * 2.0 ** 2)))
    out = resample_to_grid(coarse, src_pitch_um=(1.0, 1.0), grid=grid)
    assert out.shape == (40, 40)
    # peak near centre, correlation with the native fine blob is high
    num = float((out * fine).sum())
    den = float(np.sqrt((out ** 2).sum() * (fine ** 2).sum()))
    assert num / den > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_target.py -v`
Expected: FAIL with `ModuleNotFoundError: dfxm_geo.scoring.target`

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/target.py
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .types import GridSpec


def _crop_or_pad(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    h, w = a.shape
    out = np.zeros((H, W), dtype=a.dtype)
    sy0 = max((h - H) // 2, 0)
    sx0 = max((w - W) // 2, 0)
    dy0 = max((H - h) // 2, 0)
    dx0 = max((W - w) // 2, 0)
    th = min(h, H)
    tw = min(w, W)
    out[dy0:dy0 + th, dx0:dx0 + tw] = a[sy0:sy0 + th, sx0:sx0 + tw]
    return out


def resample_to_grid(
    img: np.ndarray, src_pitch_um: tuple[float, float], grid: GridSpec
) -> np.ndarray:
    """Resample an image from its object-plane pitch onto the library grid.

    Zooms by src_pitch/dst_pitch so a feature of a given micrometre extent keeps
    that extent, then center-crops or zero-pads to the library shape.
    """
    arr = np.asarray(img, dtype=np.float64)
    sy, sx = src_pitch_um
    dy, dx = grid.pitch_um
    if dy <= 0 or dx <= 0 or sy <= 0 or sx <= 0:
        raise ValueError("pixel pitches must be positive")
    zoomed = zoom(arr, (sy / dy, sx / dx), order=1)
    return _crop_or_pad(zoomed, grid.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_target.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/target.py tests/test_scoring_target.py
git commit -m "feat(scoring): target.resample_to_grid (physical resample + crop/pad)"
```

---

## Task 7: Library loader — single master

**Files:**
- Create: `src/dfxm_geo/scoring/library.py`
- Test: `tests/test_scoring_library.py`

- [ ] **Step 1: Write the failing test (with a fixture that mirrors the real layout)**

```python
# tests/test_scoring_library.py
import numpy as np
import h5py
import pytest
from dfxm_geo.scoring.library import load_library
from dfxm_geo.scoring.types import GridSpec


def _write_master(tmp_path, name, candidates, psize_m=1.0e-7):
    """Build an identify master + per-scan detector files mirroring the real
    BLISS layout: frames live in a per-scan file linked via ExternalLink."""
    master = tmp_path / name
    with h5py.File(master, "w") as h:
        for i, c in enumerate(candidates, start=1):
            entry = h.create_group(f"{i}.1")
            s = entry.create_group("sample")
            s["slip_plane_normal"] = np.asarray(c["plane"], dtype=np.int32)
            s["burgers"] = np.asarray(c["b"], dtype=np.int32)
            s["rotation_deg"] = float(c["alpha"])
            g = entry.create_group("dfxm_geo")
            g["gb_cos"] = float(c.get("gb_cos", 0.8))
            g["gb_visible"] = np.int8(c.get("gb_visible", 1))
            g["q_hkl"] = np.asarray(c.get("q_hkl", [0.0, 2.0, 0.0]), dtype=np.float64)
            g["psize"] = float(psize_m)
            # per-scan detector file + external link
            scan = tmp_path / f"{name}_scan{i:04d}.h5"
            with h5py.File(scan, "w") as d:
                d.create_dataset("/entry_0000/dfxm_sim_detector/image",
                                 data=c["frame"][None].astype(np.float32))
            inst = entry.create_group("instrument").create_group("dfxm_sim_detector")
            inst["data"] = h5py.ExternalLink(scan.name, "/entry_0000/dfxm_sim_detector/image")
    return master


def _blob(shape, yx):
    a = np.zeros(shape, dtype=np.float32)
    y, x = yx
    a[y:y + 3, x:x + 3] = 100.0
    return a


def test_load_single_master(tmp_path):
    cands = [
        {"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((12, 12), (2, 2))},
        {"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 30.0, "frame": _blob((12, 12), (7, 7))},
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    lib = load_library(master)
    assert len(lib) == 2
    assert lib.frames.shape == (2, 12, 12)
    assert lib.frames.dtype == np.float32
    assert lib.labels[0].slip_plane_normal == (1, 1, 1)
    assert lib.labels[0].burgers == (1, 0, 1)
    assert lib.labels[1].rotation_deg == 30.0
    assert lib.labels[0].scan_index == 1
    assert lib.grid == GridSpec(pitch_um=(0.1, 0.1), shape=(12, 12))   # psize 1e-7 m -> 0.1 um


def test_load_missing_psize_raises(tmp_path):
    cands = [{"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))}]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    with h5py.File(master, "r+") as h:
        del h["1.1/dfxm_geo/psize"]
    with pytest.raises(ValueError, match="psize"):
        load_library(master)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_library.py -k "single_master or missing_psize" -v`
Expected: FAIL with `ModuleNotFoundError: dfxm_geo.scoring.library`

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/library.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import h5py
import numpy as np

from .types import CandidateLabel, CandidateLibrary, GridSpec

_DET_LINK = "instrument/dfxm_sim_detector/data"

PathLike = Union[str, Path]


def _expand_paths(paths: Union[PathLike, Sequence[PathLike]]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return sorted(p.glob("**/dfxm_identify.h5"))
        return [p]
    out: list[Path] = []
    for item in paths:
        out.extend(_expand_paths(item))
    return out


def _identify_entries(h: h5py.File) -> list[str]:
    entries = [k for k in h.keys() if k[:1].isdigit() and k.endswith(".1")]
    return sorted(entries, key=lambda k: int(k.split(".")[0]))


def _read_label(h: h5py.File, entry: str, fp: Path) -> CandidateLabel:
    s = h[entry]["sample"]
    g = h[entry]["dfxm_geo"]
    return CandidateLabel(
        slip_plane_normal=tuple(int(x) for x in np.asarray(s["slip_plane_normal"])),
        burgers=tuple(int(x) for x in np.asarray(s["burgers"])),
        rotation_deg=float(s["rotation_deg"][()]),
        gb_cos=float(g["gb_cos"][()]),
        gb_visible=bool(int(g["gb_visible"][()])),
        q_hkl=tuple(float(x) for x in np.asarray(g["q_hkl"])),
        scan_index=int(entry.split(".")[0]),
        source_file=str(fp),
    )


def _read_frame(h: h5py.File, entry: str, reduction: str) -> np.ndarray:
    data = np.asarray(h[entry][_DET_LINK], dtype=np.float64)
    if data.ndim == 2:
        return data
    n = data.shape[0]
    if n == 1:
        return data[0]
    if reduction in ("auto", "max"):
        return data.max(axis=0)
    if reduction == "mean":
        return data.mean(axis=0)
    if reduction == "single":
        raise ValueError(f"{entry} has {n} frames but frame_reduction='single'")
    raise ValueError(f"unknown frame_reduction: {reduction!r}")


def _read_grid(h: h5py.File, entry: str, shape: tuple[int, ...]) -> GridSpec:
    g = h[entry]["dfxm_geo"]
    if "psize" not in g:
        raise ValueError(f"{entry}/dfxm_geo lacks 'psize'; cannot derive object-plane grid")
    pitch_um = float(g["psize"][()]) * 1.0e6      # metres -> micrometres (square object-plane pixels)
    return GridSpec(pitch_um=(pitch_um, pitch_um), shape=(int(shape[-2]), int(shape[-1])))


def load_library(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    include_invisible: bool = False,
    frame_reduction: str = "auto",
) -> CandidateLibrary:
    """Load candidate frames + labels from one or more identify masters."""
    files = _expand_paths(paths)
    if not files:
        raise ValueError(f"no identify masters found at {paths!r}")
    frames: list[np.ndarray] = []
    labels: list[CandidateLabel] = []
    grid: GridSpec | None = None
    for fp in files:
        with h5py.File(fp, "r") as h:
            for entry in _identify_entries(h):
                label = _read_label(h, entry, fp)
                if not include_invisible and not label.gb_visible:
                    continue
                arr = _read_frame(h, entry, frame_reduction)
                this_grid = _read_grid(h, entry, arr.shape)
                if grid is None:
                    grid = this_grid
                elif this_grid != grid:
                    raise ValueError(
                        f"non-uniform grid at {fp}:{entry}: {this_grid} vs {grid}"
                    )
                frames.append(arr.astype(np.float32))
                labels.append(label)
    if not frames:
        raise ValueError("library is empty (no visible candidates after filtering)")
    assert grid is not None
    return CandidateLibrary(frames=np.stack(frames), labels=labels, grid=grid)
```

> **Implementation note (verify once against real data):** `_read_grid` assumes
> `psize` is the object-plane pixel pitch in metres and that object-plane pixels
> are square. Before the validation run, open one real `dfxm_identify.h5` and
> confirm `psize` units and that pitch * shape matches the configured FOV
> (`xl_range`/`yl_range`). If it differs, adjust `_read_grid` only; nothing else
> depends on the unit choice (tests use a self-consistent fixture).

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_library.py -k "single_master or missing_psize" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/library.py tests/test_scoring_library.py
git commit -m "feat(scoring): library.load_library single-master parse (frames+labels+grid)"
```

---

## Task 8: Library loader — filters, reduction, directory, uniform-grid guard

**Files:**
- Test: `tests/test_scoring_library.py` (the implementation from Task 7 already covers these; this task locks the behavior with tests)

- [ ] **Step 1: Write the failing/locking tests (append)**

```python
def test_invisible_filter(tmp_path):
    cands = [
        {"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "gb_visible": 1, "frame": _blob((8, 8), (1, 1))},
        {"plane": (1, 1, 1), "b": (1, -1, 0), "alpha": 0.0, "gb_visible": 0, "frame": _blob((8, 8), (4, 4))},
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    assert len(load_library(master)) == 1                       # invisible dropped
    assert len(load_library(master, include_invisible=True)) == 2


def test_directory_concatenation(tmp_path):
    d1 = tmp_path / "seed00001"; d1.mkdir()
    d2 = tmp_path / "seed00002"; d2.mkdir()
    _write_master(d1, "dfxm_identify.h5",
                  [{"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))}])
    _write_master(d2, "dfxm_identify.h5",
                  [{"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 0.0, "frame": _blob((8, 8), (4, 4))}])
    lib = load_library(tmp_path)
    assert len(lib) == 2


def test_multiframe_reduction(tmp_path):
    master = tmp_path / "dfxm_identify.h5"
    with h5py.File(master, "w") as h:
        e = h.create_group("1.1")
        s = e.create_group("sample")
        s["slip_plane_normal"] = np.asarray((1, 1, 1), np.int32)
        s["burgers"] = np.asarray((1, 0, 1), np.int32)
        s["rotation_deg"] = 0.0
        g = e.create_group("dfxm_geo")
        g["gb_cos"] = 0.8; g["gb_visible"] = np.int8(1)
        g["q_hkl"] = np.asarray([0.0, 2.0, 0.0]); g["psize"] = 1.0e-7
        scan = tmp_path / "scan1.h5"
        stack = np.zeros((3, 8, 8), np.float32); stack[1, 2, 2] = 50.0
        with h5py.File(scan, "w") as d:
            d["/entry_0000/dfxm_sim_detector/image"] = stack
        e.create_group("instrument").create_group("dfxm_sim_detector")["data"] = \
            h5py.ExternalLink("scan1.h5", "/entry_0000/dfxm_sim_detector/image")
    lib = load_library(master, frame_reduction="max")
    assert lib.frames.shape == (1, 8, 8)
    assert lib.frames[0, 2, 2] == 50.0                          # max-projection kept the peak


def test_non_uniform_grid_raises(tmp_path):
    cands = [
        {"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))},
        {"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 0.0, "frame": _blob((10, 10), (4, 4))},
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    with pytest.raises(ValueError, match="non-uniform grid"):
        load_library(master)
```

- [ ] **Step 2: Run tests**

Run: `PY -m pytest tests/test_scoring_library.py -v`
Expected: PASS (all 6 in the file). If any fail, fix `library.py` (not the tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_scoring_library.py
git commit -m "test(scoring): lock library filters/reduction/dir/uniform-grid behavior"
```

---

## Task 9: Identifier.rank + RankedMatch

**Files:**
- Create: `src/dfxm_geo/scoring/identify.py`
- Test: `tests/test_scoring_identify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_identify.py
import numpy as np
from dfxm_geo.scoring.identify import Identifier, RankedMatch, IdentifiabilityResult
from dfxm_geo.scoring.types import GridSpec, CandidateLabel, CandidateLibrary


def _lib():
    rng = np.random.default_rng(7)
    base = rng.normal(40, 6, size=(20, 20))
    f0 = base.copy(); f0[3:6, 3:6] += 120
    f1 = base.copy(); f1[14:17, 14:17] += 120
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
    assert matches[0].label.burgers == (0, 1, 1)        # best match = frame 1
    assert matches[0].score >= matches[1].score
    assert matches[0].scan_index == 2


def test_rank_resamples_off_grid_target():
    lib = _lib()
    ident = Identifier(lib, backend="numpy")
    # target at 2x coarser pitch but same physical feature as frame 0
    coarse = lib.frames[0][::2, ::2].copy()
    matches = ident.rank(coarse, target_pitch_um=(0.2, 0.2), top_k=1)
    assert matches[0].label.burgers == (1, 0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_identify.py -k rank -v`
Expected: FAIL with `ModuleNotFoundError: dfxm_geo.scoring.identify`

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/identify.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from .engine import score_matrix, score_target
from .target import resample_to_grid
from .types import CandidateLabel, CandidateLibrary


@dataclass(frozen=True)
class RankedMatch:
    score: float
    label: CandidateLabel
    scan_index: int
    source_file: str


@dataclass
class IdentifiabilityResult:
    matrix: np.ndarray
    labels: list[CandidateLabel]
    top1_accuracy: float
    per_class_accuracy: dict
    confusion: np.ndarray
    class_order: list

    def save(self, out_dir: Union[str, Path], *, plots: bool = False) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "matrix.npy", self.matrix)
        np.save(out / "confusion.npy", self.confusion)
        labels_json = [
            {
                "slip_plane_normal": list(lbl.slip_plane_normal),
                "burgers": list(lbl.burgers),
                "rotation_deg": lbl.rotation_deg,
                "gb_cos": lbl.gb_cos,
                "gb_visible": lbl.gb_visible,
                "q_hkl": list(lbl.q_hkl),
                "scan_index": lbl.scan_index,
                "source_file": lbl.source_file,
            }
            for lbl in self.labels
        ]
        (out / "labels.json").write_text(json.dumps(labels_json, indent=2))
        metrics = {
            "top1_accuracy": self.top1_accuracy,
            "per_class_accuracy": {str(k): v for k, v in self.per_class_accuracy.items()},
            "class_order": [str(k) for k in self.class_order],
            "n_candidates": len(self.labels),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if plots:
            self._save_plots(out)

    def _save_plots(self, out: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.matrix, cmap="gnuplot", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        ax.set_title("Normalized cross-correlation matrix")
        fig.savefig(out / "heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        iu = np.triu_indices_from(self.matrix, k=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.asarray(self.matrix)[iu].ravel(), bins=100, color="steelblue")
        ax.set_xlabel("cross-correlation")
        ax.set_ylabel("frequency")
        fig.savefig(out / "score_hist.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


class Identifier:
    def __init__(
        self, library: CandidateLibrary, *,
        normalize: str = "symmetric", k: float = 2.0,
        backend: str = "auto", class_key_mode: str = "plane_burgers",
    ) -> None:
        self.library = library
        self.normalize = normalize
        self.k = k
        self.backend = backend
        self.class_key_mode = class_key_mode

    def rank(
        self, target_img: np.ndarray, target_pitch_um: tuple[float, float], *, top_k: int = 10
    ) -> list[RankedMatch]:
        t = resample_to_grid(target_img, target_pitch_um, self.library.grid)
        scores = score_target(
            t, self.library.frames, normalize=self.normalize, backend=self.backend, k=self.k
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RankedMatch(
                score=float(scores[i]),
                label=self.library.labels[i],
                scan_index=self.library.labels[i].scan_index,
                source_file=self.library.labels[i].source_file,
            )
            for i in order
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_identify.py -k rank -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/identify.py tests/test_scoring_identify.py
git commit -m "feat(scoring): Identifier.rank + RankedMatch + IdentifiabilityResult.save"
```

---

## Task 10: Identifier.study (all-pairs identifiability)

**Files:**
- Modify: `src/dfxm_geo/scoring/identify.py`
- Test: `tests/test_scoring_identify.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def test_study_distinct_classes_top1_perfect(tmp_path):
    lib = _lib()                                    # two clearly different frames/classes
    ident = Identifier(lib, backend="numpy")
    res = ident.study()
    assert isinstance(res, IdentifiabilityResult)
    assert res.matrix.shape == (2, 2)
    assert res.top1_accuracy == 1.0                 # each recovers its own class (LOO)
    res.save(tmp_path)
    assert (tmp_path / "matrix.npy").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "labels.json").exists()


def test_study_degenerate_classes_confused():
    rng = np.random.default_rng(8)
    base = rng.normal(40, 6, size=(20, 20))
    f = base.copy(); f[3:6, 3:6] += 120
    frames = np.stack([f, f.copy(), np.roll(f, (9, 9), axis=(0, 1))]).astype(np.float32)
    from dfxm_geo.scoring.types import GridSpec, CandidateLabel, CandidateLibrary
    labels = [
        CandidateLabel((1, 1, 1), (1, 0, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 1, "m"),
        CandidateLabel((1, 1, 1), (0, 1, 1), 0.0, 0.8, True, (0.0, 2.0, 0.0), 2, "m"),  # identical frame, diff class
        CandidateLabel((1, 1, 1), (1, 1, 0), 0.0, 0.8, True, (0.0, 2.0, 0.0), 3, "m"),
    ]
    lib = CandidateLibrary(frames, labels, GridSpec((0.1, 0.1), (20, 20)))
    res = Identifier(lib, backend="numpy").study()
    assert res.top1_accuracy < 1.0                  # the two identical frames confuse
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_identify.py -k study -v`
Expected: FAIL with `AttributeError: 'Identifier' object has no attribute 'study'`

- [ ] **Step 3: Write the implementation (append the method to `Identifier`)**

```python
    def study(self) -> IdentifiabilityResult:
        C = score_matrix(
            self.library.frames, normalize=self.normalize, backend=self.backend, k=self.k
        )
        keys = [lbl.class_key(self.class_key_mode) for lbl in self.library.labels]
        class_order = sorted(set(keys), key=str)
        idx_of = {key: i for i, key in enumerate(class_order)}
        n = len(keys)
        confusion = np.zeros((len(class_order), len(class_order)), dtype=int)
        per_tot = {key: 0 for key in class_order}
        per_ok = {key: 0 for key in class_order}
        correct = 0
        M = np.array(C, dtype=float, copy=True)
        np.fill_diagonal(M, -np.inf)                 # leave-one-out: never match self
        for i in range(n):
            j = int(np.argmax(M[i]))
            true, pred = keys[i], keys[j]
            confusion[idx_of[true], idx_of[pred]] += 1
            per_tot[true] += 1
            if pred == true:
                correct += 1
                per_ok[true] += 1
        top1 = correct / n if n else 0.0
        per_class = {
            key: (per_ok[key] / per_tot[key] if per_tot[key] else 0.0) for key in class_order
        }
        return IdentifiabilityResult(
            matrix=C, labels=self.library.labels, top1_accuracy=top1,
            per_class_accuracy=per_class, confusion=confusion, class_order=class_order,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PY -m pytest tests/test_scoring_identify.py -v`
Expected: PASS (all in file)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/identify.py tests/test_scoring_identify.py
git commit -m "feat(scoring): Identifier.study (LOO top-1 + per-class + confusion)"
```

---

## Task 11: Public API + runner script + mypy gate

**Files:**
- Modify: `src/dfxm_geo/scoring/__init__.py`
- Create: `scripts/run_identify_study.py`
- Test: `tests/test_scoring_identify.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def test_public_api_surface():
    import dfxm_geo.scoring as sc
    for name in ["GridSpec", "CandidateLabel", "CandidateLibrary", "preprocess",
                 "cross_correlation_peak", "score_matrix", "score_target",
                 "resample_to_grid", "load_library", "Identifier",
                 "IdentifiabilityResult", "RankedMatch"]:
        assert hasattr(sc, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY -m pytest tests/test_scoring_identify.py -k public_api -v`
Expected: FAIL (missing attributes on the package)

- [ ] **Step 3: Write the implementation**

```python
# src/dfxm_geo/scoring/__init__.py
"""Cross-correlation dislocation identification scorer."""
from .engine import cross_correlation_peak, preprocess, score_matrix, score_target
from .identify import IdentifiabilityResult, Identifier, RankedMatch
from .library import load_library
from .target import resample_to_grid
from .types import CandidateLabel, CandidateLibrary, GridSpec

__all__ = [
    "GridSpec", "CandidateLabel", "CandidateLibrary",
    "preprocess", "cross_correlation_peak", "score_matrix", "score_target",
    "resample_to_grid", "load_library",
    "Identifier", "IdentifiabilityResult", "RankedMatch",
]
```

```python
# scripts/run_identify_study.py
"""Run the all-pairs identifiability study over identify master(s).

Example:
    python scripts/run_identify_study.py path/to/masters_dir out/ --plots
"""
from __future__ import annotations

import argparse

from dfxm_geo.scoring import Identifier, load_library


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DFXM dislocation identifiability study")
    p.add_argument("library", help="identify master .h5 or a directory of masters")
    p.add_argument("out_dir", help="output directory for matrix/metrics/plots")
    p.add_argument("--normalize", default="symmetric",
                   choices=["symmetric", "diagonal", "none"])
    p.add_argument("--class-key", default="plane_burgers",
                   choices=["plane_burgers", "burgers", "plane_burgers_alpha"])
    p.add_argument("--include-invisible", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--plots", action="store_true")
    a = p.parse_args(argv)

    lib = load_library(a.library, include_invisible=a.include_invisible)
    ident = Identifier(lib, normalize=a.normalize, backend=a.backend,
                       class_key_mode=a.class_key)
    res = ident.study()
    res.save(a.out_dir, plots=a.plots)
    print(f"top-1 accuracy: {res.top1_accuracy:.4f} over {len(lib)} candidates")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + mypy**

Run: `PY -m pytest tests/test_scoring_identify.py -k public_api -v`
Expected: PASS

Run: `PY -m mypy src/dfxm_geo/scoring/`
Expected: `Success: no issues found` (0 errors). Fix any typing issues in `scoring/` only.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/__init__.py scripts/run_identify_study.py tests/test_scoring_identify.py
git commit -m "feat(scoring): public API exports + run_identify_study runner; mypy clean"
```

---

## Task 12: Full scoring-suite green + whole-package mypy

**Files:** none (verification task)

- [ ] **Step 1: Run the full scoring test set**

Run: `PY -m pytest tests/test_scoring_*.py -v`
Expected: PASS (all scoring tests).

- [ ] **Step 2: Whole-package mypy unchanged**

Run: `PY -m mypy src/dfxm_geo/`
Expected: 0 errors (baseline was 0/45 at the screw-fix commit; scoring adds 0).

- [ ] **Step 3: Confirm no regression in a fast slice**

Run: `rm -f direct_space/deformation_gradient_tensors/Fg_*.npy && PY -m pytest tests/ -q -k "not slow"`
Expected: same pass/skip/xfail counts as the screw-fix baseline plus the new scoring tests, 0 failures. (If a stale-cache failure appears, re-run after the `rm`.)

- [ ] **Step 4: Commit (if any fixups were needed)**

```bash
git add -A
git commit -m "test(scoring): full scoring suite green; whole-package mypy 0 errors"
```

---

## Task 13 (optional): torch GPU backend

Only do this if a torch+CUDA box is available. The numpy path is the default and
fully correct; this task adds a faster backend for large all-pairs runs.

**Files:**
- Modify: `src/dfxm_geo/scoring/engine.py`
- Test: `tests/test_scoring_engine.py`

- [ ] **Step 1: Write the test (auto-skips without torch+CUDA)**

```python
def test_torch_matrix_matches_numpy():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    frames = _three_frames()
    M_np = engine.score_matrix(frames, normalize="symmetric", backend="numpy")
    M_t = engine.score_matrix(frames, normalize="symmetric", backend="torch")
    assert np.allclose(M_np, M_t, atol=1e-4)
```

- [ ] **Step 2: Run test**

Run: `PY -m pytest tests/test_scoring_engine.py -k torch_matrix -v`
Expected: SKIP (no CUDA) or, with torch+CUDA, FAIL (`_raw_matrix_torch` missing).

- [ ] **Step 3: Implement `_raw_matrix_torch` (append to engine.py)**

```python
def _raw_matrix_torch(pp: np.ndarray) -> np.ndarray:
    import torch
    n = pp.shape[0]
    t = torch.as_tensor(pp, dtype=torch.float32, device="cuda")
    F = torch.fft.fft2(t)
    C = np.zeros((n, n))
    for i in range(n):
        cc = torch.fft.ifft2(F[i] * torch.conj(F[i:])).abs()
        peaks = cc.reshape(n - i, -1).amax(dim=1).cpu().numpy()
        C[i, i:] = peaks
        C[i:, i] = peaks
    return C
```

- [ ] **Step 4: Run test**

Run: `PY -m pytest tests/test_scoring_engine.py -k torch_matrix -v`
Expected: PASS or SKIP.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/scoring/engine.py tests/test_scoring_engine.py
git commit -m "feat(scoring): optional torch/CUDA backend for score_matrix"
```

---

## Self-review

**Spec coverage:**
- Cross-corr engine (preprocess + metric + normalize + backend): Tasks 2-5, 13. ✓
- HDF5 library loader (frames+labels+grid, filters, reduction, directory, uniform-grid guard): Tasks 7-8. ✓
- Physical target resample: Task 6. ✓
- Identifiability study (matrix, LOO top-1, per-class, confusion, save+plots): Tasks 9-10. ✓
- Single-target ranking: Task 9. ✓
- numpy default, torch optional, noarch-safe (no required new dep): Tasks 3/4/13, prerequisites. ✓
- No new `[project.scripts]` (runner is a plain script): Task 11. ✓
- Error handling (missing field, empty library, non-uniform grid, torch absent): Tasks 3, 7, 8. ✓
- Tests for translation invariance, resample, loader, metrics: Tasks 3-10. ✓

**Spec deviations (intentional, minor):**
- Added `types.py` (spec listed 4 modules) so `target.py` does not import from
  `library.py` just for `GridSpec`. Cleaner dependency graph.
- Plotting moved from `study(plots=...)` to `IdentifiabilityResult.save(plots=...)`
  (the spec sketched `study(plots=)`); the plot artifacts are a save-time concern.

**Placeholder scan:** no TBD/TODO; every code step has complete code. The one
flagged runtime check (`_read_grid` psize units) has concrete code plus a
verify-once note; it is not a placeholder.

**Type consistency:** `score_matrix`/`score_target`/`cross_correlation_peak` share
`normalize`/`backend`/`k` names; `CandidateLabel` fields match between `types.py`,
the loader, the fixture, and `IdentifiabilityResult.save`; `class_key` modes match
between `types.py` and the `--class-key` CLI choices.
