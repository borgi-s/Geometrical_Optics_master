# Dislocation identification scorer (cross-correlation) design

Date: 2026-06-15
Branch: `feature/disloc-cross-corr-scorer` (worktree off the screw-fix commit `1252487`)
Status: design approved, pending spec review

## 1. Motivation

Dislocation identification in DFXM works by matching an observed contrast image
against a library of forward-simulated candidates (one per slip system, Burgers
vector, and dislocation line-direction "character" angle) and ranking them by
image similarity. The original method (Borgi 2025) lived in a single ad-hoc
cluster script (`disloc_identify.py`) that generated its own image library with
the pre-cleanup globals-era forward model and scored an all-pairs FFT
cross-correlation matrix on the GPU. That script no longer runs against the
current `dfxm_geo` package (its `direct_space.forward_model` / `image_processor`
/ `functions` imports were deleted in the cleanup), it hardcodes cluster paths,
and several of its later cells are broken even on their own terms.

There is currently no reusable scorer anywhere in the package. The identify
pipeline produces a rich, labelled HDF5 candidate library but nothing consumes
it for similarity scoring. We also need this scorer to re-validate the
identifiability numbers and Borgi 2025 figures that were computed with the
pre-screw-fix identify physics (the g.b=0 screw extinction bug, fixed in
`1252487`).

This design specifies a small, reusable scoring subpackage that:

1. reads the modern identify HDF5 output (frames plus labels),
2. computes a normalized FFT cross-correlation similarity,
3. produces an all-pairs identifiability study (matrix plus top-1 / confusion
   metrics), and
4. ranks a single target image (simulated or a pre-reduced measured frame)
   against the library.

## 2. Goals and non-goals

### Goals

- A clean cross-correlation engine that operates on in-memory arrays, with no
  HDF5 or filesystem knowledge, unit-testable in isolation.
- A loader that turns one or more `dfxm_identify.h5` masters into a labelled
  in-memory candidate library.
- Physical resampling of a measured (or simulated) target onto the library grid
  so similarity compares true physical feature sizes.
- An all-pairs identifiability study and a single-target ranking query, both
  built on the shared engine.
- numpy/scipy default backend; optional torch (GPU) acceleration when installed.
- Keep the package `noarch: python` and lean: torch is an optional dependency,
  never required.

### Non-goals (v1)

- Measured-scan reduction. The caller reduces a raw BLISS scan (frame selection,
  projection, ROI crop) to a single 2D contrast image and supplies it plus its
  object-plane pixel pitch. The brittle, dataset-specific "which frame / which
  ROI" logic stays out of the scorer.
- A `.npy`-folder legacy adapter (the old cluster input format). Deferred until
  there is legacy data to score.
- Sub-pixel registration beyond the translation invariance that FFT
  cross-correlation already provides.
- Multi-grid studies. One study assumes one uniform object-plane grid; different
  reflections or configs with different grids are separate studies.
- A new `[project.scripts]` console entry point (avoids the conda-forge
  feedstock entry-point sync burden). A `scripts/` runner is provided instead;
  promotion to a packaged CLI is a deliberate later release step.

## 3. Grounding facts (current pipeline)

From exploration of the worktree at commit `1252487`:

- Identify output: master `dfxm_identify.h5` with one BLISS entry `/N.1/` per
  candidate. Pixel frames live in per-scan LIMA files at
  `/entry_0000/dfxm_sim_detector/image`, reachable from the master via the
  external link `/N.1/instrument/dfxm_sim_detector/data`. dtype is float32
  (raw) or uint16 (after the noise model). Shape `(N_frames, H, W)`, typically
  `(1, 510, 170)` for single-mode candidates.
- Per-candidate labels: `/N.1/sample/slip_plane_normal` (int32 (3,)),
  `/N.1/sample/burgers` (int32 (3,)), `/N.1/sample/rotation_deg` (float).
  Physics: `/N.1/dfxm_geo/gb_cos` (float), `/N.1/dfxm_geo/gb_visible` (int8),
  `/N.1/dfxm_geo/q_hkl` (3,), plus `psize`, `zl_rms`, `theta`. The embedded
  TOML is at `/dfxm_geo/config_toml`.
- The object-plane field of view is set by the config `xl_range` / `yl_range`
  (the old script used `extent = [-yl_range, yl_range, -xl_range, xl_range]`
  in micrometres). The loader derives the object-plane pixel pitch from the
  candidate provenance; the exact field plumbing (a recorded `psize` / range,
  versus parsing `config_toml`) is confirmed during implementation, with a clear
  error if a required field is absent.
- Measured data: BLISS HDF5, detector `pco_ff`, frames `(N, 2048, 2048)` uint16
  at `/<entry>/measurement/pco_ff` (and `/<entry>/instrument/pco_ff/image`);
  detector pixel size at `/<entry>/instrument/pco_ff/x_pixel_size`,
  `y_pixel_size`. The much larger 2048^2 detector grid versus the 510x170 sim
  grid is exactly why physical resampling is required.

## 4. Architecture

```
src/dfxm_geo/scoring/
  __init__.py     # public API re-exports
  engine.py       # preprocessing + FFT cross-corr + normalization (numpy default, torch optional)
  library.py      # HDF5 identify master(s) -> CandidateLibrary
  target.py       # load + physically resample a 2D target onto the library grid
  identify.py     # Identifier: all-pairs study + single-target ranking; result types
scripts/
  run_identify_study.py   # thin runner: dir of masters -> matrix + metrics + plots
tests/
  test_scoring_engine.py
  test_scoring_library.py
  test_scoring_target.py
  test_scoring_identify.py
```

Dependency direction: `identify.py` depends on `engine.py`, `library.py`,
`target.py`. `engine.py` depends only on numpy/scipy (and optionally torch).
`library.py` depends on h5py. `target.py` depends on numpy/scipy.

## 5. Data structures

```python
@dataclass(frozen=True)
class GridSpec:
    pitch_um: tuple[float, float]   # object-plane (dy, dx) micrometres per pixel
    shape: tuple[int, int]          # (H, W)

@dataclass(frozen=True)
class CandidateLabel:
    slip_plane_normal: np.ndarray   # int (3,)
    burgers: np.ndarray             # int (3,)
    rotation_deg: float             # character angle alpha
    gb_cos: float
    gb_visible: bool
    q_hkl: np.ndarray               # (3,)
    scan_index: int                 # N from /N.1
    source_file: str
    # derived grouping key for identifiability; default (slip_plane_normal, burgers)
    def class_key(self, mode="plane_burgers") -> tuple: ...

@dataclass
class CandidateLibrary:
    frames: np.ndarray              # (N, H, W) float32, one reduced image per candidate
    labels: list[CandidateLabel]    # length N
    grid: GridSpec
```

The class key controls what counts as a correct identification. Default groups by
`(slip_plane_normal, burgers)`, treating the character angle alpha as a
within-class nuisance variable. The key mode is configurable (for example
`plane_burgers_alpha` for an exact-pose study, or `burgers` only).

## 6. Engine (`engine.py`)

Pure-array functions, no IO.

```python
def preprocess(img: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Background subtract (mean + k*std), clip negatives to 0, divide by std."""

def _fft_frames(frames, backend) -> Any:        # precompute FFT of each preprocessed frame once
def cross_correlation_peak(fa, fb, backend) -> float:
    """Peak magnitude of the circular FFT cross-correlation: |IFFT(fa * conj(fb))|.max().
    Circular cross-correlation makes the score translation invariant."""

def score_matrix(
    frames: np.ndarray,
    *,
    normalize: str = "symmetric",   # "symmetric" | "diagonal" | "none"
    backend: str = "auto",          # "auto" | "numpy" | "torch"
    k: float = 2.0,
) -> np.ndarray:                    # (N, N)
    """Preprocess all frames, precompute their FFTs once, fill the upper triangle
    (the raw peak matrix is symmetric), mirror, then normalize."""

def score_target(
    target: np.ndarray,             # already on the library grid
    frames: np.ndarray,
    *,
    normalize: str = "symmetric",
    backend: str = "auto",
    k: float = 2.0,
) -> np.ndarray:                    # (N,)
```

Normalization modes:

- `symmetric`: `C[i,j] / sqrt(C[i,i] * C[j,j])`. Self-match is 1.0, symmetric,
  comparable across pairs. Default.
- `diagonal`: `C[i,j] / C[i,i]` (legacy row-by-its-own-autopeak; asymmetric).
  Provided to reproduce the original Borgi 2025 figures.
- `none`: raw peak values.

Performance: precompute each frame's FFT once (N transforms, not N^2). Each pair
is one complex multiply plus one inverse transform. The raw peak matrix is
symmetric, so only the upper triangle is computed. Backend `auto` uses torch
with CUDA if importable, otherwise numpy / `scipy.fft`. `torch` requested
explicitly without torch installed raises; `auto` falls back to numpy silently.

## 7. Library loader (`library.py`)

```python
def load_library(
    paths: str | Path | Sequence[str | Path],
    *,
    include_invisible: bool = False,
    frame_reduction: str = "auto",   # "auto" | "single" | "max" | "mean"
) -> CandidateLibrary:
```

- `paths` may be a single master `.h5`, a directory (globs `dfxm_identify.h5`
  masters, useful because a real campaign is one master per seed), or an
  explicit list. Results are concatenated.
- Per `/N.1`: read frames through the external link, read labels from
  `/N.1/sample` and `/N.1/dfxm_geo`, derive the object-plane `GridSpec` from
  provenance. A missing required field raises an error naming the file and entry.
- `frame_reduction`: a 1-frame candidate is used as-is; a multi-frame (rocking)
  candidate is reduced by `max` or `mean` projection. `auto` means single if one
  frame, else max-projection.
- Candidates with `gb_visible == 0` are dropped unless `include_invisible`
  (invisible contrast is near-blank and produces degenerate scores).
- The loader asserts a single uniform `GridSpec` across the whole library.
  Mixed grids raise; mixing reflections or configs is a separate study.

## 8. Target loader and resampling (`target.py`)

```python
def resample_to_grid(
    img: np.ndarray,
    src_pitch_um: tuple[float, float],   # object-plane (dy, dx) micrometres per pixel
    grid: GridSpec,
) -> np.ndarray:
    """Zoom by src_pitch / grid.pitch_um (scipy.ndimage.zoom), then center-crop
    or zero-pad to grid.shape."""
```

The caller supplies the target's object-plane pixel pitch (magnification already
accounted for). For a measured frame this is `detector_pitch / magnification`.
The resampled target is then preprocessed and scored by the engine on the same
grid as every library candidate, so a contrast lobe of a given micrometre extent
matches the same extent in the library.

## 9. High-level API (`identify.py`)

```python
class Identifier:
    def __init__(self, library: CandidateLibrary, *,
                 normalize: str = "symmetric", k: float = 2.0,
                 backend: str = "auto", class_key_mode: str = "plane_burgers"): ...

    def study(self, *, plots: bool = False) -> IdentifiabilityResult: ...
    def rank(self, target_img: np.ndarray, target_pitch_um: tuple[float, float],
             *, top_k: int = 10) -> list[RankedMatch]: ...

@dataclass
class IdentifiabilityResult:
    matrix: np.ndarray              # (N, N) normalized scores
    labels: list[CandidateLabel]
    top1_accuracy: float            # leave-one-out
    per_class_accuracy: dict[tuple, float]
    confusion: np.ndarray           # by class
    class_order: list[tuple]
    def save(self, out_dir: str | Path) -> None: ...   # matrix.npy, labels.json, metrics.json, optional plots

@dataclass(frozen=True)
class RankedMatch:
    score: float
    label: CandidateLabel
    scan_index: int
    source_file: str
```

- `study()` computes the all-pairs matrix and the leave-one-out top-1 accuracy:
  for each row i, take the argmax over j != i and check whether the predicted
  class key matches the true class key. It also reports per-class accuracy and a
  by-class confusion matrix. `save()` writes `matrix.npy`, `labels.json`,
  `metrics.json`, and (if `plots=True`) a heatmap and a score histogram using
  matplotlib.
- `rank()` preprocesses and resamples the target onto the library grid, scores
  it against all candidates, sorts descending, and returns the top_k matches with
  their labels.

## 10. Error handling

- Non-uniform library grid: raise (cannot cross-correlate different shapes).
- Missing label or provenance field: raise, naming the file and `/N.1`.
- Missing target pitch: raise.
- Empty library after the invisibility filter: raise.
- `backend="torch"` without torch installed: raise. `backend="auto"`: silent
  numpy fallback.

## 11. Testing

- engine: identical images score 1.0; a shifted copy scores ~1.0 (translation
  invariance via circular cross-correlation); random/orthogonal patterns score
  low; matrix symmetry holds; `diagonal` mode gives 1.0 on each row's self.
- target: a synthetic Gaussian blob rendered at 2x pitch, resampled to 1x,
  scores high against the native 1x version; center-crop and zero-pad behave for
  larger and smaller FOV.
- library: a tiny synthetic identify HDF5 fixture (2 to 3 candidates, 5x5
  frames, the external-link layout) parses frames, labels, and grid; the
  invisibility filter drops `gb_visible==0`; a directory of two masters
  concatenates.
- identify: a toy library of clearly distinct classes gives top-1 = 100%; a
  degenerate library with two identical classes shows the mixup in the confusion
  matrix.
- Determinism, and `mypy src/dfxm_geo/` stays at 0 errors.

## 12. Packaging and dependencies

- New runtime dependency: none required beyond what the package already has
  (numpy, scipy, h5py, matplotlib). torch is optional and only used when present.
- No change to `[project.scripts]`, so no conda-forge feedstock entry-point sync
  is needed for this work.
- The subpackage is pure Python, preserving `noarch: python`.

## 13. Validation use (driving the work)

Once built, regenerate a candidate library with the screw-fixed identify physics
(commit `1252487`) and run `Identifier.study()` to re-check the identifiability
numbers and the Borgi 2025 figures that were produced with the pre-fix screw
fields. This is the concrete acceptance use of the scorer, run separately after
the module and its tests are green.
