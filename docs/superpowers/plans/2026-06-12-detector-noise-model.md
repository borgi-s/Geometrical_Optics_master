# Realistic Detector Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detector frames written to HDF5 become statistically indistinguishable from raw ID03 `pco_ff` frames: uint16 ADU on the measured noise floor, photon statistics at a calibrated absolute level, `exposure_time` as a physical knob; plus a weak-beam background-subtraction display helper.

**Architecture:** A pure `DetectorModel` module (measured PCO Edge 4.2 bi preset) converts sampling-normalized forward intensity → ADU at the existing post-write seam (`_maybe_apply_poisson_noise` → `_apply_detector_model`), now also called from forward mode. A new `[detector]` config block replaces `[noise]` everywhere (breaking, no back-compat). Spec: `docs/superpowers/specs/2026-06-12-detector-noise-model-design.md`.

**Tech Stack:** numpy (Generator/SeedSequence spawn), h5py, frozen dataclasses, pytest, mypy. Run everything with the venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`, cwd `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`. Branch: `feature/detector-noise-model`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/dfxm_geo/detector.py` | Create | `DetectorModel` frozen dataclass, `SensorMap`, `PCO_EDGE_4P2_ID03` preset, `resolve_model()` |
| `src/dfxm_geo/viz/detector.py` | Create | `subtract_background()` weak-beam display reduction |
| `src/dfxm_geo/direct_space/forward_model.py` | Modify | sampling-invariance normalization in `forward_from_static` (4 return points, 2 insertion sites) |
| `src/dfxm_geo/config.py` | Modify | `DetectorConfig` dataclass; parse `[detector]` in BOTH `SimulationConfig.from_toml` and `load_identification_config`; reject `[noise]`; delete `IdentificationNoiseConfig` |
| `src/dfxm_geo/orchestrator.py` | Modify | `_maybe_apply_poisson_noise` → `_apply_detector_model`; new forward-mode call; `config.noise` → `config.detector` at RNG sites 1065/1080/1306/1320 |
| `src/dfxm_geo/io/hdf5.py` | Modify | `replace_detector_image()` helper (dtype-changing in-place dataset swap + provenance attrs) |
| `src/dfxm_geo/pipeline.py` | Modify | facade re-exports: drop `IdentificationNoiseConfig`/`_maybe_apply_poisson_noise`, add `DetectorConfig`/`_apply_detector_model` |
| `src/dfxm_geo/cli.py` | Modify | `--seed` overrides `cfg.detector.rng_seed` |
| `src/dfxm_geo/io/darling_compat.py` (or wherever `DarlingReader` lives — `grep -r "class DarlingReader" src/`) | Modify | cast loaded stacks to float32 so the reader contract survives uint16 sources |
| `src/dfxm_geo/data/configs/*.toml` (3), `configs/profile_identify_*.toml` (3) | Modify | `[noise]` → `[detector]` |
| `tests/test_detector_model.py` | Create | unit tests for `detector.py` |
| `tests/test_viz_detector.py` | Create | tests for `subtract_background` |
| `tests/test_forward_sampling_invariance.py` | Create | Nsub=1 vs Nsub=2 invariance (slow) |
| `tests/test_detector_config.py` | Create | `[detector]` parsing, `[noise]` rejection, CLI seed |
| `tests/test_apply_detector_model.py` | Create | seam unit + e2e (uint16, floor, labels noiseless, per-reflection independence) |
| ~24 existing test files | Modify | mechanical `IdentificationNoiseConfig` → `DetectorConfig` sweep (Task 7 has the mapping table) |
| `docs/detector-noise-model.md` | Create | model equation, fit table, counts_scale derivation record |
| `docs/calibration/*.py` | Create | archived fitting scripts (from `..\noise_scratch\`) + `derive_counts_scale.py` |
| `docs/output-format.md` | Modify | uint16 + `[detector]` provenance attrs |

**Ordering constraint:** Tasks 1–5 are independent of the config break. Task 6 (config) + Task 7 (sweep) MUST land back-to-back — the suite is red in between; do not run gates between them. Tasks 8–10 wire the seam. Tasks 11–14 are configs/calibration/docs.

**Measured constants used throughout** (provenance: spec §2):
offset_base=102.5 ADU, dark_rate=7.5 ADU/s, read_noise_var_base=6.3 ADU², read_noise_var_rate=11.0 ADU²/s, gain=2.14 ADU/photon-equivalent, fpn_sigma=1.8 ADU, tail_fraction=0.015, tail_scale=10.0 ADU (reproduces the +10 ADU 0.57 % and +50 ADU 0.011 % census), edge_rows=16, edge_peak=20.0 ADU, edge_decay=4.0 rows, full well 65535.

---

### Task 1: `viz/detector.py` — weak-beam background subtraction

**Files:**
- Create: `src/dfxm_geo/viz/detector.py`
- Test: `tests/test_viz_detector.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_viz_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm_geo.viz.detector'`

- [ ] **Step 3: Implement**

```python
"""Detector-frame display helpers.

The weak-beam reduction implemented here mirrors standard experimental
post-processing of DFXM detector images: estimate the background level as
``mean + k*std`` over the frame and clamp what remains at zero. Validated
against ID03 pco_ff weak-beam frames (subtracts ~117 ADU at 1 s exposure);
see docs/detector-noise-model.md. In strong-beam condition this zeros
nearly the whole image by construction — the GO model is weak-beam-only.
"""

import numpy as np


def subtract_background(image: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Subtract ``mean + k*std`` and floor at zero. Returns a new float array.

    Works on a single frame or a stack; statistics are computed over the
    whole input (per-run background, matching how the reduction is applied
    at the beamline viewer).
    """
    img = np.asarray(image, dtype=np.float64)
    threshold = img.mean() + k * img.std()
    return np.clip(img - threshold, 0.0, None)
```

- [ ] **Step 4: Run the test — PASS expected.** Also run `mypy src/dfxm_geo/viz/detector.py` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/viz/detector.py tests/test_viz_detector.py
git commit -m "feat(viz): weak-beam background subtraction (mean + k*sigma, floored)"
```

---

### Task 2: `detector.py` — model dataclass + preset + registry

**Files:**
- Create: `src/dfxm_geo/detector.py`
- Test: `tests/test_detector_model.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the realistic detector model."""

import numpy as np
import pytest

from dfxm_geo.detector import (
    FULL_WELL,
    PCO_EDGE_4P2_ID03,
    DetectorModel,
    SensorMap,
    resolve_model,
)


def test_preset_matches_measured_calibration():
    m = PCO_EDGE_4P2_ID03
    assert m.name == "pco_edge_4.2_id03"
    assert m.gain == pytest.approx(2.14)
    assert m.offset(0.0) == pytest.approx(102.5)
    assert m.offset(1.0) == pytest.approx(110.0)
    # noise sigma: sqrt(6.3) ~ 2.51 at t=0, sqrt(6.3 + 11*0.5) ~ 3.43 at 0.5 s
    assert m.noise_sigma(0.0) == pytest.approx(2.51, abs=0.01)
    assert m.noise_sigma(0.5) == pytest.approx(3.43, abs=0.01)


def test_resolve_model_registry():
    assert resolve_model("pco_edge_4.2_id03") is PCO_EDGE_4P2_ID03
    assert resolve_model("ideal") is None
    with pytest.raises(ValueError, match="unknown detector model"):
        resolve_model("pco_edge_99")


def test_full_well_is_uint16_max():
    assert FULL_WELL == np.iinfo(np.uint16).max
```

- [ ] **Step 2: Run to verify failure** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the module skeleton**

```python
"""Realistic detector model: measured pco_ff noise + ADU conversion.

Converts sampling-normalized forward-model intensity into uint16 ADU frames
that are statistically indistinguishable from raw ID03 PCO Edge 4.2 bi
(scintillator-coupled sCMOS) frames.

Calibration provenance — fitted 2026-06-12 from Sina's ID03 beamtime data
(true darks + 61-frame photon-transfer sweep); method and fit figures in
docs/detector-noise-model.md, scripts in docs/calibration/, design in
docs/superpowers/specs/2026-06-12-detector-noise-model-design.md.
"""

from dataclasses import dataclass

import numpy as np

FULL_WELL = 65535  # uint16 saturation, matches the real LIMA frames


@dataclass(frozen=True)
class SensorMap:
    """Per-pixel fixed-pattern state for one synthetic 'camera'.

    Generated once per run (same map for every scan and reflection — it is
    *fixed*-pattern), reproducible from the config seed.
    """

    fpn_offset: np.ndarray  # (ny, nx) float64 ADU, added to the time-dependent offset


@dataclass(frozen=True, kw_only=True)
class DetectorModel:
    """Noise/conversion parameters of a physical detector. All ADU units."""

    name: str
    offset_base: float  # ADU at t=0
    dark_rate: float  # ADU per second of exposure
    read_noise_var_base: float  # ADU^2 at t=0
    read_noise_var_rate: float  # ADU^2 per second (dark shot noise)
    gain: float  # ADU per incident photon-equivalent
    fpn_sigma: float  # Gaussian core of the fixed-pattern offset
    tail_fraction: float  # fraction of warm/hot pixels (exponential tail)
    tail_scale: float  # ADU, exponential tail scale
    edge_rows: int  # rows at each frame edge with elevated offset
    edge_peak: float  # ADU boost at the outermost row
    edge_decay: float  # e-folding of the boost, in rows

    def offset(self, exposure_time: float) -> float:
        """Mean dark level in ADU at the given exposure."""
        return self.offset_base + self.dark_rate * exposure_time

    def noise_sigma(self, exposure_time: float) -> float:
        """Temporal (read + dark shot) noise sigma in ADU at the given exposure."""
        return float(
            np.sqrt(self.read_noise_var_base + self.read_noise_var_rate * exposure_time)
        )


PCO_EDGE_4P2_ID03 = DetectorModel(
    name="pco_edge_4.2_id03",
    offset_base=102.5,
    dark_rate=7.5,
    read_noise_var_base=6.3,
    read_noise_var_rate=11.0,
    gain=2.14,
    fpn_sigma=1.8,
    tail_fraction=0.015,
    tail_scale=10.0,
    edge_rows=16,
    edge_peak=20.0,
    edge_decay=4.0,
)

_MODELS = {PCO_EDGE_4P2_ID03.name: PCO_EDGE_4P2_ID03}


def resolve_model(name: str) -> "DetectorModel | None":
    """Map a config model name to a DetectorModel; ``"ideal"`` → None."""
    if name == "ideal":
        return None
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown detector model {name!r}; expected 'ideal' or one of "
            f"{sorted(_MODELS)}"
        ) from None
```

- [ ] **Step 4: Run tests — PASS.** `mypy src/dfxm_geo/detector.py` → 0 errors.

- [ ] **Step 5: Commit** — `feat(detector): DetectorModel dataclass + measured PCO Edge 4.2 bi preset`

---

### Task 3: `make_sensor_map` — synthetic fixed-pattern state

**Files:**
- Modify: `src/dfxm_geo/detector.py`
- Test: `tests/test_detector_model.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_detector_model.py`)

```python
def test_sensor_map_statistics_match_dark_census():
    rng = np.random.default_rng(7)
    m = PCO_EDGE_4P2_ID03
    sm = m.make_sensor_map((512, 512), rng)
    fpn = sm.fpn_offset
    assert fpn.shape == (512, 512)
    interior = fpn[m.edge_rows : -m.edge_rows, :]
    # Gaussian core sigma ~ fpn_sigma (robust estimate, tail excluded)
    mad = np.median(np.abs(interior - np.median(interior)))
    assert 1.4826 * mad == pytest.approx(m.fpn_sigma, rel=0.2)
    # warm/hot census: ~0.57 % above +10 ADU, ~0.011 % above +50 ADU (spec §2)
    frac10 = (interior > 10.0).mean()
    frac50 = (interior > 50.0).mean()
    assert 0.003 < frac10 < 0.012
    assert 1e-5 < frac50 < 5e-4
    # edge rows elevated relative to interior
    assert fpn[0, :].mean() > interior.mean() + 0.5 * m.edge_peak


def test_sensor_map_is_reproducible_and_seed_sensitive():
    m = PCO_EDGE_4P2_ID03
    a = m.make_sensor_map((64, 64), np.random.default_rng(3)).fpn_offset
    b = m.make_sensor_map((64, 64), np.random.default_rng(3)).fpn_offset
    c = m.make_sensor_map((64, 64), np.random.default_rng(4)).fpn_offset
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
```

- [ ] **Step 2: Run — FAIL** (`AttributeError: ... no attribute 'make_sensor_map'`).

- [ ] **Step 3: Implement** (method on `DetectorModel`)

```python
    def make_sensor_map(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> SensorMap:
        """Synthesize a fixed-pattern offset map by sampling the measured
        distributions (spec §3.1): Gaussian core + exponential warm/hot tail
        + elevated edge rows. Synthetic rather than the literal 2048^2
        calibration map so it scales to any simulated detector size.
        """
        ny, nx = shape
        fpn = rng.normal(0.0, self.fpn_sigma, size=shape)
        tail = rng.random(shape) < self.tail_fraction
        fpn[tail] += rng.exponential(self.tail_scale, size=int(tail.sum()))
        rows = np.arange(ny, dtype=np.float64)
        dist = np.minimum(rows, ny - 1 - rows)  # distance to nearest edge
        boost = self.edge_peak * np.exp(-dist / self.edge_decay)
        boost[dist >= self.edge_rows] = 0.0
        return SensorMap(fpn_offset=fpn + boost[:, None])
```

- [ ] **Step 4: Run tests — PASS.** If the census bounds fail, the tail constants are wrong — fix the implementation, do NOT widen the bounds beyond the ranges above (they bracket the measured 0.57 % / 0.011 %).

- [ ] **Step 5: Commit** — `feat(detector): synthetic sensor map (FPN + warm/hot tail + edge rows)`

---

### Task 4: `DetectorModel.apply` — ideal ADU → noisy uint16

**Files:**
- Modify: `src/dfxm_geo/detector.py`
- Test: `tests/test_detector_model.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_statistics_match_photon_transfer():
    """Mean and variance must reproduce the measured model:
    mean = s + offset(t), var = gain*s + noise_sigma(t)^2 (+ tiny rounding var)."""
    m = PCO_EDGE_4P2_ID03
    rng = np.random.default_rng(11)
    flat = SensorMap(fpn_offset=np.zeros((200, 200)))
    for t, s in [(1.0, 0.0), (1.0, 50.0), (1.0, 1000.0), (0.1, 300.0)]:
        ideal = np.full((10, 200, 200), s)
        out = m.apply(ideal, exposure_time=t, rng=rng, sensor=flat)
        assert out.dtype == np.uint16
        vals = out.astype(np.float64)
        expected_mean = s + m.offset(t)
        expected_var = m.gain * s + m.noise_sigma(t) ** 2 + 1.0 / 12.0
        assert vals.mean() == pytest.approx(expected_mean, rel=0.01)
        assert vals.var() == pytest.approx(expected_var, rel=0.05)


def test_apply_clamps_at_full_well_and_zero():
    m = PCO_EDGE_4P2_ID03
    flat = SensorMap(fpn_offset=np.zeros((4, 4)))
    rng = np.random.default_rng(0)
    hot = m.apply(np.full((1, 4, 4), 1e9), 1.0, rng, flat)
    assert (hot == FULL_WELL).all()
    # negative ideal input (shouldn't happen, but) must not wrap below zero
    cold = m.apply(np.full((1, 4, 4), -1e6), 1.0, rng, flat)
    assert (cold <= 200).all()


def test_apply_is_deterministic_for_fixed_rng():
    m = PCO_EDGE_4P2_ID03
    flat = SensorMap(fpn_offset=np.zeros((8, 8)))
    ideal = np.full((2, 8, 8), 500.0)
    a = m.apply(ideal, 1.0, np.random.default_rng(5), flat)
    b = m.apply(ideal, 1.0, np.random.default_rng(5), flat)
    assert np.array_equal(a, b)


def test_apply_adds_sensor_map_offset():
    m = PCO_EDGE_4P2_ID03
    sm = SensorMap(fpn_offset=np.full((16, 16), 40.0))
    out = m.apply(np.zeros((50, 16, 16)), 1.0, np.random.default_rng(1), sm)
    assert out.astype(float).mean() == pytest.approx(m.offset(1.0) + 40.0, rel=0.02)
```

- [ ] **Step 2: Run — FAIL** (`no attribute 'apply'`).

- [ ] **Step 3: Implement**

```python
    def apply(
        self,
        ideal_adu: np.ndarray,
        exposure_time: float,
        rng: np.random.Generator,
        sensor: SensorMap,
    ) -> np.ndarray:
        """Ideal expected-signal ADU (above offset) → noisy uint16 frames.

        Noise composition (spec §2): gain-scaled Poisson on the photon
        count, plus the exposure-dependent offset, the fixed-pattern map,
        and Gaussian read/dark-shot noise; rounded and clamped to uint16.
        ``gain * Poisson(s / gain)`` reproduces the measured variance
        ``gain * s`` exactly (scintillator excess noise is inside the
        fitted gain by construction).
        """
        photons = np.clip(ideal_adu, 0.0, None) / self.gain
        signal = self.gain * rng.poisson(photons).astype(np.float64)
        noisy = (
            signal
            + self.offset(exposure_time)
            + sensor.fpn_offset
            + rng.normal(0.0, self.noise_sigma(exposure_time), size=ideal_adu.shape)
        )
        return np.clip(np.rint(noisy), 0, FULL_WELL).astype(np.uint16)
```

- [ ] **Step 4: Run the whole file — PASS.** `mypy src/dfxm_geo/detector.py` → 0.

- [ ] **Step 5: Commit** — `feat(detector): apply() — gain-Poisson + floor + read noise -> uint16`

---

### Task 5: sampling-invariance normalization in `forward_from_static`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (function `forward_from_static`, lines ~798–899)
- Test: `tests/test_forward_sampling_invariance.py`

Background (spec §3.2(a), explorer-verified): each detector pixel accumulates `Nsub² · NN3` voxel contributions, and `prob_z` is an UNNORMALIZED Gaussian whose per-column sum scales with `NN3`. Dividing the finished image by `Nsub² · prob_z[:NN3].sum()` makes the level invariant under `Nsub`/`NN3` changes (the `prob_z`-sum form makes the z-quadrature exact). Grid ordering: `mgrid[xl, yl, zl].reshape(3, -1)` → z varies fastest → the first `NN3` entries of `prob_z` are exactly one column's z-weights.

- [ ] **Step 1: Write the failing test**

```python
"""Nsub-invariance of the normalized forward image (spec §3.2(a))."""

import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import (
    ForwardContext,
    GeometryContext,
    InstrumentContext,
    ResolutionContext,
    forward_from_static,
)
from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution


def _make_ctx(npixels: int, nsub: int, theta: float) -> ForwardContext:
    """Hand-built mini context replicating the module grid math
    (forward_model.py lines ~140-195 and build_geometry_context ~645-654);
    hand-built because build_instrument_context() snapshots module globals
    and build_geometry_context reads module-level yl_range/zl_range."""
    psize = 40e-9
    zl_rms = 0.15e-6 / 2.35
    nn1 = npixels // 3 * nsub
    nn2 = npixels * nsub
    nn3 = npixels // 30 * nsub
    yl_start = -psize * npixels / 2 + psize / (2 * nsub)
    yl_range = -yl_start
    zl_range = 0.5 * zl_rms * 6
    YI = (np.arange(nn1) // nsub).repeat(nn3 * nn2)
    ZI = np.tile((np.arange(nn2) // nsub).repeat(nn3), nn1)
    flat_indices = ZI.astype(np.int64) * (nn1 // nsub) + YI.astype(np.int64)
    instr = InstrumentContext(
        psize=psize, zl_rms=zl_rms, Npixels=npixels, Nsub=nsub,
        NN1=nn1, NN2=nn2, NN3=nn3,
        Ud=np.identity(3), Us=np.identity(3),
        flat_indices=flat_indices, yl_start=yl_start,
        xl_steps=nn1, yl_steps=nn2, zl_steps=nn3,
    )
    Theta_ = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    xl_start = yl_start / np.tan(2 * theta) / 3
    xl_range = -xl_start
    rl = np.vstack(
        np.mgrid[
            -xl_range : xl_range : complex(nn1),
            -yl_range : yl_range : complex(nn2),
            -zl_range : zl_range : complex(nn3),
        ]
    ).reshape(3, -1)
    geom = GeometryContext(
        theta_0=theta, Theta=Theta_, xl_start=xl_start, xl_range=xl_range,
        rl=rl, prob_z=np.exp(-0.5 * (rl[2] / zl_rms) ** 2),
    )
    # Analytic backend: grid-free, no kernel file needed. Constructor args
    # mirror _load_analytic_resolution (forward_model.py ~570-599) — copy the
    # exact kwargs that call uses (zeta_v_fwhm/zeta_h_fwhm/NA_rms/eps_rms).
    analytic = AnalyticResolution(
        theta=theta, zeta_v_fwhm=0.53e-3, zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35, eps_rms=0.014 / 2.35,
    )
    res = ResolutionContext(
        Resq_i=None, qi1_start=0.0, qi1_step=0.0, qi2_start=0.0, qi2_step=0.0,
        qi3_start=0.0, qi3_step=0.0, npoints1=None, npoints2=None, npoints3=None,
        analytic_eval=analytic, loaded_kernel_path=None,
    )
    q = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0)
    return ForwardContext(instrument=instr, geometry=geom, resolution=res, q_hkl=q)


@pytest.mark.slow
def test_normalized_image_invariant_under_nsub():
    """base_qc = 0 puts every ray at the acceptance peak (the analytic
    backend is peak-normalized in deviation space), so the image reduces to
    the pure accumulation Σ p_Q(Theta@ang)·prob_z per pixel — exactly the
    quantity the normalization must make sampling-invariant. No Hg physics
    needed; invariance is a property of the accumulation, not the scene."""
    theta = 0.31  # ~17.8 deg, Al(111)-like
    images = {}
    for nsub in (1, 2):
        ctx = _make_ctx(90, nsub, theta)
        n = ctx.geometry.rl.shape[1]
        base_qc = np.zeros((3, n))
        images[nsub] = np.asarray(forward_from_static(base_qc, ctx))
    im1, im2 = images[1], images[2]
    assert im1.shape == im2.shape  # (npixels, npixels//3) regardless of Nsub
    assert im1.sum() > 0  # rays sit at the acceptance peak by construction
    # Without normalization the level ratio would be Nsub^2*(NN3 ratio) = 8.
    ratio = im2.sum() / im1.sum()
    assert ratio == pytest.approx(1.0, abs=0.05)
```

NOTE for the implementer: if `AnalyticResolution.__init__` requires kwargs beyond the five shown, copy the full instantiation from the analytic loader near `forward_model.py:570–599` verbatim (the test comment says so). If `im1.sum()` is ~0 even with `base_qc = 0`, the analytic peak is not at qi=0 for this geometry — find the peak with `analytic(np.zeros((3, 1)))` vs small offsets and adjust `phi` accordingly; do NOT paper over it by removing the `sum() > 0` assert.

- [ ] **Step 2: Run — FAIL** (ratio ≈ 8, since no normalization exists yet)

Run: `...python.exe -m pytest tests/test_forward_sampling_invariance.py -q -m slow` (check how slow tests are selected: `grep -n "slow" pyproject.toml tests/conftest.py` — follow the existing marker convention).

- [ ] **Step 3: Implement** — in `forward_from_static`, insert after the `im_1` allocation (line ~833):

```python
    # Sampling-invariance normalization (spec 2026-06-12 detector model,
    # §3.2(a)): each pixel sums Nsub^2 * NN3 voxel contributions and prob_z
    # is an unnormalized Gaussian whose column sum scales with NN3. Dividing
    # by Nsub^2 * (one column's prob_z sum) makes the absolute level
    # invariant under sampling-density changes (z-quadrature exact). The
    # grid is mgrid[xl, yl, zl] flattened C-order, so prob_z[:NN3] is one
    # column's z-weights.
    _sampling_norm = float(instr.Nsub) ** 2 * float(geom.prob_z[: instr.NN3].sum())
```

and divide before each of the two return clusters. Analytic branch (after `im_1 += contribution.reshape(im_1.shape)`):

```python
        im_1 /= _sampling_norm
        if qi_return:
            return im_1, qi.reshape(3, instr.NN1, instr.NN2, instr.NN3)
        return im_1
```

MC branch (after the `_mc_lut_forward(...)` call):

```python
    im_1 /= _sampling_norm
    if qi_return:
        assert qi is not None  # built above because qi_return is True
        return im_1, qi.reshape(3, instr.NN1, instr.NN2, instr.NN3)
    return im_1
```

- [ ] **Step 4: Run the invariance test — PASS.** Then run the FULL suite: `...python.exe -m pytest -q`. Expected fallout and triage:
  - Tests asserting *relative* structure (COM, parity analytic-vs-MC, shapes, frame counts): must still pass — both paths are normalized identically.
  - `tests/test_hdf5_bit_equiv.py` and `test_forward_output_matches_pickle_era_snapshot`: already xfail; if they switch to XPASS/different-fail, update their xfail reason strings to mention the sampling normalization.
  - Any test asserting an absolute image magnitude: scale its expectation by the norm — report it in the commit message. Do NOT silence value assertions by loosening tolerances.

- [ ] **Step 5: Commit** — `feat(forward): sampling-invariant image normalization (Nsub^2 x prob_z column sum)`

---

### Task 6: `DetectorConfig` + `[detector]` parsing + `[noise]` rejection + CLI

**Files:**
- Modify: `src/dfxm_geo/config.py` (delete `IdentificationNoiseConfig` lines ~689–701; parse sites ~651, ~843–848)
- Modify: `src/dfxm_geo/cli.py:82–98`
- Modify: `src/dfxm_geo/pipeline.py:37` (facade export)
- Test: `tests/test_detector_config.py`

⚠ After this task the suite is RED until Task 7 lands. Commit both together-adjacent; don't run full gates in between.

- [ ] **Step 1: Write the failing tests**

```python
"""[detector] config block parsing + [noise] rejection."""

import pytest

from dfxm_geo.config import DetectorConfig, load_identification_config
from dfxm_geo.pipeline import SimulationConfig

MINIMAL_IDENTIFY = """
mode = "single"
"""  # extend with whatever minimal keys load_identification_config requires
# (copy the smallest working TOML from tests/test_empty_toml_runs.py)


def test_detector_defaults_are_on_everywhere():
    cfg = DetectorConfig()
    assert cfg.model == "pco_edge_4.2_id03"
    assert cfg.exposure_time == 1.0
    assert cfg.rng_seed == 0
    assert cfg.counts_scale > 0


def test_detector_block_round_trips_from_toml(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(
        MINIMAL_IDENTIFY
        + '\n[detector]\nmodel = "ideal"\nexposure_time = 0.25\nrng_seed = 9\n'
    )
    cfg = load_identification_config(p)
    assert cfg.detector.model == "ideal"
    assert cfg.detector.exposure_time == 0.25
    assert cfg.detector.rng_seed == 9


def test_unknown_model_rejected(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(MINIMAL_IDENTIFY + '\n[detector]\nmodel = "bogus"\n')
    with pytest.raises(ValueError, match="unknown detector model"):
        load_identification_config(p)


def test_noise_block_rejected_with_pointer(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(MINIMAL_IDENTIFY + "\n[noise]\npoisson_noise = true\n")
    with pytest.raises(ValueError, match=r"\[noise\] was removed.*\[detector\]"):
        load_identification_config(p)


def test_simulation_config_also_parses_detector(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text('[detector]\nmodel = "ideal"\n')
    cfg = SimulationConfig.from_toml(p)
    assert cfg.detector.model == "ideal"
```

- [ ] **Step 2: Run — FAIL** (`ImportError: cannot import name 'DetectorConfig'`).

- [ ] **Step 3: Implement in `config.py`**

Add (near the other small blocks, importing at module top `from dfxm_geo.detector import resolve_model`):

```python
@dataclass(frozen=True, kw_only=True)
class DetectorConfig:
    """[detector] block: realistic detector model + absolute calibration.

    Replaces the pre-v3 [noise] block. ``rng_seed`` is the run's stochastic
    seed: identification parameter draws use spawn child [0], detector noise
    child [1], the synthetic sensor map child [2] (SeedSequence children are
    stable under spawn-count growth, so [0]/[1] are bit-identical to the old
    layout when seeds match). ``counts_scale`` is the data anchor in ADU/s
    per normalized intensity unit (docs/detector-noise-model.md).
    """

    model: str = "pco_edge_4.2_id03"
    exposure_time: float = 1.0
    counts_scale: float = 1.0e4  # provisional; Task 12 pins the derived value
    rng_seed: int = 0

    def __post_init__(self) -> None:
        resolve_model(self.model)  # raises ValueError on unknown names
        if self.exposure_time <= 0:
            raise ValueError(f"exposure_time must be > 0, got {self.exposure_time}")
        if self.counts_scale <= 0:
            raise ValueError(f"counts_scale must be > 0, got {self.counts_scale}")
```

In `load_identification_config` (~line 843), replace the noise line:

```python
    if "noise" in data:
        raise ValueError(
            "[noise] was removed; use [detector] instead "
            "(poisson_noise -> model, intensity_scale -> counts_scale, "
            "rng_seed -> [detector] rng_seed). See docs/detector-noise-model.md."
        )
    detector = DetectorConfig(**data.get("detector", {}))
```

…and store it on `IdentificationConfig` as field `detector: DetectorConfig` (delete the `noise` field). Apply the same two lines in `SimulationConfig.from_toml` (~line 651 — note the raw-TOML dict variable there is named `raw`, not `data`) and add `detector: DetectorConfig` to `SimulationConfig` (default `DetectorConfig()` so programmatic construction keeps working). Delete the `IdentificationNoiseConfig` class.

In `cli.py:96–98`:

```python
    if args.seed is not None:
        cfg = replace(cfg, detector=replace(cfg.detector, rng_seed=args.seed))
        cfg.__post_init__()  # re-run validation
```

and update the `--seed` help string: `"Override the config's [detector] rng_seed (integer). ..."`.

In `pipeline.py:37`: replace the `IdentificationNoiseConfig` re-export with `DetectorConfig`.

- [ ] **Step 4: Run ONLY the new test file — PASS.** (Full suite is red until Task 7.)

- [ ] **Step 5: Commit** — `feat(config)!: [detector] block replaces [noise] (breaking)`

---

### Task 7: mechanical sweep — `IdentificationNoiseConfig` → `DetectorConfig` across src + tests

**Files:** ~24 test files (list in the config-explorer report; re-derive with the greps below) + `orchestrator.py` seed-site renames.

- [ ] **Step 1: Inventory** (from repo root `Geometrical_Optics_master`):

```bash
grep -rln "IdentificationNoiseConfig\|config\.noise\|cfg\.noise\|noise_cfg\|\[noise\]\|poisson_noise" src/ tests/ configs/ | sort
```

- [ ] **Step 2: Apply the mapping table** — every occurrence, no exceptions:

| Old | New |
|---|---|
| `IdentificationNoiseConfig(poisson_noise=False, rng_seed=S)` | `DetectorConfig(model="ideal", rng_seed=S)` |
| `IdentificationNoiseConfig(poisson_noise=True, rng_seed=S, ...)` | `DetectorConfig(rng_seed=S)` (default model ON) |
| `noise=<expr>` kwarg in `IdentificationConfig(...)` | `detector=<expr>` |
| `config.noise.rng_seed` / `cfg.noise.rng_seed` | `config.detector.rng_seed` / `cfg.detector.rng_seed` |
| TOML `[noise]\npoisson_noise = false\n...` in test strings | `[detector]\nmodel = "ideal"\n` |
| TOML `[noise]\npoisson_noise = true\n...` in test strings | `[detector]\n` block with desired keys, or delete (defaults are ON) |
| `intensity_scale = 7.0` (TOML or kwarg) | delete (superseded by `counts_scale`; only keep if the test specifically tested scaling — then port to `counts_scale`) |

In `orchestrator.py`, rename the four seed-site reads (lines ~1065, ~1080, ~1306, ~1320): `noise_cfg = config.noise` → `detector_cfg = config.detector` (and `noise_cfg.rng_seed` → `detector_cfg.rng_seed`). Leave `_maybe_apply_poisson_noise` itself COMPILING but dead-simple for now: change its body's `config.noise` to `config.detector` references minimally (it is rewritten in Task 9; this task only needs the suite green).
Interim shim for Task 7 only: inside `_maybe_apply_poisson_noise`, map `scale = 1.0` and `poisson = detector_cfg.model != "ideal"`, keep the rest; Task 9 replaces the function wholesale.

Decision rule for tests currently relying on noiseless output but not explicitly setting `poisson_noise=False` (i.e. they relied on defaults): if they assert dtype/values/exact arrays → give them `model="ideal"`; if they only assert shapes/counts/attrs → leave defaults.

- [ ] **Step 3: Run the full suite**: `...python.exe -m pytest -q`. Triage to zero NEW failures (compare against the pre-existing xfails). Expected count baseline: 952 passing as of `8dbd2fe` + new tests from Tasks 1–6.

- [ ] **Step 4: `mypy src/dfxm_geo/`** → 0 errors.

- [ ] **Step 5: Commit** — `refactor!: sweep IdentificationNoiseConfig -> DetectorConfig across src+tests`

---

### Task 8: `replace_detector_image` helper in `io/hdf5.py`

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (near `_create_detector_skeleton`, ~line 136)
- Test: `tests/test_apply_detector_model.py` (new file, first test)

- [ ] **Step 1: Write the failing test**

```python
"""Detector-model seam: dataset replacement + orchestrator pass."""

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    _write_detector_file,
    replace_detector_image,
)


def test_replace_detector_image_swaps_dtype_and_keeps_attrs(tmp_path):
    p = tmp_path / "det_0000.h5"
    _write_detector_file(p, np.ones((3, 8, 8), dtype=np.float32))
    new = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    with h5py.File(p, "a") as f:
        old_attrs = dict(f[DETECTOR_INTERNAL_PATH].attrs)
        replace_detector_image(f, new, extra_attrs={"detector_model": "pco_edge_4.2_id03"})
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        assert np.array_equal(ds[...], new)
        assert ds.compression == "gzip"
        assert ds.chunks == (1, 8, 8)
        for k, v in old_attrs.items():
            assert ds.attrs[k] == v  # e.g. the NeXus 'interpretation' attr
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
    # softlinks (NXdata/measurement) must still resolve through the new dataset
    with h5py.File(p, "r") as f:
        assert f[DETECTOR_INTERNAL_PATH].shape == (3, 8, 8)
```

- [ ] **Step 2: Run — FAIL** (`ImportError: replace_detector_image`).

- [ ] **Step 3: Implement** in `io/hdf5.py`:

```python
def replace_detector_image(
    f: h5py.File,
    data: np.ndarray,
    *,
    extra_attrs: "dict[str, Any] | None" = None,
) -> None:
    """Replace the detector image dataset in-place, allowing a dtype change.

    Used by the post-write detector-model pass: the ideal float32 frames are
    recreated as uint16 ADU at the same internal path (softlinks resolve by
    name, so NXdata/measurement links survive). Original dataset attrs are
    preserved; ``extra_attrs`` adds [detector] provenance.
    """
    old = f[DETECTOR_INTERNAL_PATH]
    attrs = dict(old.attrs)
    n_frames, height, width = old.shape
    del f[DETECTOR_INTERNAL_PATH]
    new = f.create_dataset(
        DETECTOR_INTERNAL_PATH,
        data=data,
        dtype=data.dtype,
        chunks=(1, height, width),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    for k, v in attrs.items():
        new.attrs[k] = v
    for k, v in (extra_attrs or {}).items():
        new.attrs[k] = v
```

(Match the kwargs of `_create_detector_skeleton`'s create_dataset calls exactly — gzip-4 + shuffle + per-frame chunks.)

- [ ] **Step 4: Run — PASS. Commit** — `feat(io): replace_detector_image (dtype-swapping in-place dataset replacement)`

---

### Task 9: `_apply_detector_model` — the post-write seam (identification)

**Files:**
- Modify: `src/dfxm_geo/orchestrator.py` (rewrite `_maybe_apply_poisson_noise`, lines ~1225–1273; call sites 981, 1217, 1506)
- Modify: `src/dfxm_geo/pipeline.py:77` (facade re-export)
- Test: `tests/test_apply_detector_model.py` (append)

- [ ] **Step 1: Write the failing tests** (append; build the smallest real identification run the existing e2e tests use — copy the minimal config from `tests/test_pipeline_identification_hdf5.py::test_single_mode_writes_master_plus_scan_dirs` and add the `[detector]` variants)

```python
import pytest

from dfxm_geo.detector import PCO_EDGE_4P2_ID03


def _run_single(tmp_path, detector_toml: str):
    """Run the smallest identification single-mode e2e (copy the config from
    test_pipeline_identification_hdf5.test_single_mode_writes_master_plus_scan_dirs,
    swapping in `detector_toml` for its [detector] section), return the
    per-scan detector file path."""
    ...  # ~15 lines copied from the existing e2e test
    return tmp_path / "out" / "scan0001" / "dfxm_sim_detector_0000.h5"


@pytest.mark.slow
def test_identify_single_writes_uint16_with_measured_floor(tmp_path):
    det = _run_single(tmp_path, '[detector]\nexposure_time = 1.0\nrng_seed = 0\n')
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        vals = ds[...].astype(np.float64)
        m = PCO_EDGE_4P2_ID03
        # dark pixels sit at the measured floor: offset(1s) ~ 110 ADU
        dark = np.percentile(vals, 5)
        assert m.offset(1.0) - 10 < dark < m.offset(1.0) + 15
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
        assert ds.attrs["exposure_time"] == 1.0


@pytest.mark.slow
def test_identify_single_ideal_is_float32_passthrough(tmp_path):
    det = _run_single(tmp_path, '[detector]\nmodel = "ideal"\n')
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.float32
        assert "detector_model" not in ds.attrs


@pytest.mark.slow
def test_same_seed_reproduces_frames_and_seeds_differ(tmp_path):
    a = _run_single(tmp_path / "a", "[detector]\nrng_seed = 1\n")
    b = _run_single(tmp_path / "b", "[detector]\nrng_seed = 1\n")
    c = _run_single(tmp_path / "c", "[detector]\nrng_seed = 2\n")
    with h5py.File(a) as fa, h5py.File(b) as fb, h5py.File(c) as fc:
        ia, ib, ic = (
            f[DETECTOR_INTERNAL_PATH][...] for f in (fa, fb, fc)
        )
    assert np.array_equal(ia, ib)
    assert not np.array_equal(ia, ic)
```

Plus a labels-stay-noiseless test: run the zscan or multi `render_per_dislocation` config (copy from `tests/test_identification_zscan_per_dis.py`), assert `dfxm_sim_detector_primary_0000.h5` stays float32 while the combined file is uint16.

- [ ] **Step 2: Run — FAIL** (combined file still float32 / attrs missing).

- [ ] **Step 3: Implement** — replace `_maybe_apply_poisson_noise` wholesale:

```python
def _apply_detector_model(
    detector_cfg: DetectorConfig,
    output_dir: Path,
    n_scans: int,
    *,
    reflection_index: int = 0,
) -> None:
    """Convert combined-detector files to realistic uint16 ADU post-write.

    Only `dfxm_sim_detector_0000.h5` files are touched; per-dislocation
    files (`*_primary_*`, `*_secondary_*`) stay noiseless float32
    ground-truth labels (naming-based bypass, unchanged from the Poisson era).

    RNG layout (spec §3.1/§3.4): root = default_rng(rng_seed);
    spawn child [1] = noise stream (single-reflection; per-reflection runs
    use default_rng([rng_seed, reflection_index]) for independent noise),
    spawn child [2] = sensor map (SAME for every scan and reflection — the
    synthetic camera is one physical sensor). Children [0]/[1] keep the
    pre-v3 layout so parameter draws are bit-identical for equal seeds.
    """
    model = resolve_model(detector_cfg.model)
    if model is None:
        return
    if reflection_index > 0:
        noise_rng = np.random.default_rng([detector_cfg.rng_seed, reflection_index])
    else:
        noise_rng = np.random.default_rng(detector_cfg.rng_seed).spawn(2)[1]
    sensor_rng = np.random.default_rng(detector_cfg.rng_seed).spawn(3)[2]
    sensor: SensorMap | None = None
    extra_attrs = {
        "detector_model": model.name,
        "exposure_time": detector_cfg.exposure_time,
        "counts_scale": detector_cfg.counts_scale,
        "detector_gain": model.gain,
        "detector_offset": model.offset(detector_cfg.exposure_time),
        "detector_spec": "2026-06-12-detector-noise-model-design",
    }
    for k in range(1, n_scans + 1):
        det_file = (
            output_dir / SCAN_DIR_FMT.format(k) / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
        )
        if not det_file.is_file():
            continue
        with h5py.File(det_file, "a") as f:
            ideal = f[DETECTOR_INTERNAL_PATH][...].astype(np.float64)
            if sensor is None or sensor.fpn_offset.shape != ideal.shape[1:]:
                sensor = model.make_sensor_map(ideal.shape[1:], sensor_rng)
            adu = ideal * detector_cfg.counts_scale * detector_cfg.exposure_time
            noisy = model.apply(adu, detector_cfg.exposure_time, noise_rng, sensor)
            replace_detector_image(f, noisy, extra_attrs=extra_attrs)
```

Imports at orchestrator top: `from dfxm_geo.detector import SensorMap, resolve_model`, `from dfxm_geo.config import DetectorConfig`, `from dfxm_geo.io.hdf5 import replace_detector_image`. Update the three call sites (981/1217/1506): `_apply_detector_model(config.detector, output_dir, n_scans, reflection_index=reflection_index)`. Update `pipeline.py:77` re-export. Note in the docstring of the three `_run_identification_*` functions: replace mentions of "Poisson pass" with "detector-model pass" (5 docstring sites flagged by `grep -n "Poisson" src/dfxm_geo/orchestrator.py`).

- [ ] **Step 4: Run new tests + full suite — PASS.** Patch-convention check: anything monkeypatching `_maybe_apply_poisson_noise` (grep tests/) must be retargeted to `dfxm_geo.orchestrator._apply_detector_model` (internals → orchestrator, never the facade).

- [ ] **Step 5: Commit** — `feat(orchestrator)!: detector-model post-write pass replaces Poisson (uint16 ADU)`

---

### Task 10: forward-mode seam + `DarlingReader` cast

**Files:**
- Modify: `src/dfxm_geo/orchestrator.py` (the `write_simulation_h5` call site, ~line 441–464)
- Modify: `DarlingReader` source (find with `grep -rn "class DarlingReader" src/`)
- Test: `tests/test_apply_detector_model.py` (append), `tests/test_darling_reader.py` (extend)

- [ ] **Step 1: Failing test** — forward e2e (copy the minimal forward run from `tests/test_detector_dtype_float32.py`, default `[detector]`): assert `scan0001` (and `scan0002` when `include_perfect_crystal=true`) detector datasets are uint16 with floor ≈ `offset(t)`; with `model="ideal"` assert float32 (that keeps the original `test_detector_dtype_float32` intent alive — convert that file's test to the ideal variant rather than deleting it).

- [ ] **Step 2: Run — FAIL** (forward never applies the model).

- [ ] **Step 3: Implement** — in the forward writer function (after the `write_simulation_h5(...)` call at ~line 441):

```python
    n_det_scans = 2 if config.io.include_perfect_crystal else 1
    _apply_detector_model(
        config.detector,
        h5_path.parent,
        n_det_scans,
        reflection_index=reflection_index if reflection is not None else 0,
    )
```

(`reflection_index` is already in scope — it feeds `_reflection_attrs`. Multi-reflection forward thereby gets independent per-reflection noise, same convention as identify.)

`DarlingReader`: at the point the stack is loaded, add `data = data.astype(np.float32, copy=False)` so the reader contract (`dtype == float32`) holds for uint16 sources; extend `tests/test_darling_reader.py` with a uint16-source case using `replace_detector_image` on its existing fixture file.

- [ ] **Step 4: Run forward e2e + darling tests + full suite — PASS.**

- [ ] **Step 5: Commit** — `feat(forward)!: detector model applied to forward outputs (incl. perfect-crystal scan)`

---

### Task 11: shipped config templates + output-format docs

**Files:**
- Modify: `src/dfxm_geo/data/configs/identification_{single,multi,zscan}.toml`, `configs/profile_identify_{single,multi,zscan}.toml`
- Modify: `docs/output-format.md`
- Test: existing `tests/test_defaults_simple.py` / `tests/test_empty_toml_runs.py` already exercise template loading — extend `test_empty_toml_runs` with one assertion that the default-config run produces uint16 detector data.

- [ ] **Step 1:** In all six TOMLs replace the `[noise]` block with:

```toml
[detector]
# Realistic ID03 pco_ff model (uint16 ADU). Set model = "ideal" for raw
# float32 physics output. See docs/detector-noise-model.md.
model = "pco_edge_4.2_id03"
exposure_time = 1.0
rng_seed = 0
```

(carry over each file's old `rng_seed` value: profile configs used 1). The zscan file's old comment about Poisson living in rocking-curve calls is obsolete — drop it.

- [ ] **Step 2:** `docs/output-format.md`: document the uint16 dtype, the `[detector]` provenance attrs on the image dataset (`detector_model`, `exposure_time`, `counts_scale`, `detector_gain`, `detector_offset`, `detector_spec`), the ideal-mode float32 escape, and that per-dislocation label files stay float32 noiseless. State explicitly: no migration tool for pre-v3 files (no-backcompat rule).

- [ ] **Step 3:** Run full suite + mypy — green. **Commit** — `feat(config)!: shipped templates use [detector]; output-format docs updated`

---

### Task 12: `counts_scale` calibration (LOCAL — needs experimental data + kernel)

**Files:**
- Create: `docs/calibration/derive_counts_scale.py`
- Modify: `src/dfxm_geo/config.py` (pin the derived `counts_scale` default)
- Create: `docs/detector-noise-model.md` (started here, finished Task 13)

This task runs on Sina's machine only (needs `C:\Users\borgi\Documents\GM-reworked\experimental_data\` and a bootstrapped kernel). It is NOT a CI test.

- [ ] **Step 1:** Write `docs/calibration/derive_counts_scale.py`:
  1. Configure a forward scene approximating the dislocations dataset: Al, (1,-1,1)-family reflection at 17 keV, single dislocation, weak-beam offset (φ a few times the rocking width — reuse the M5 notebook-03 weak-beam settings), `model = "ideal"` so the output stays float (normalized intensity units).
  2. Measure experimental integrated feature signal: load `111_individual_dislocations_10x_focusing_2/scan0001/pco_ff_0000.h5` frame 30 (`import hdf5plugin` REQUIRED before h5py.File — LIMA bitshuffle), subtract the measured offset (110 ADU at 1 s), select the dislocation features by `subtract_background(frame, k=2.0) > 0`, and integrate the ADU above background over the largest connected feature (scipy.ndimage.label). Per-feature integral, in ADU.
  3. Measure the simulated counterpart: same integral over the simulated dislocation feature, in normalized-intensity units (sum of `subtract_background`-selected pixels — use k=2 on the noiseless image plus its own mean).
  4. `counts_scale = ADU_integral / (sim_integral * 1.0 s)`. Print it together with the secondary check: simulated core peak × counts_scale should land in 2 000–6 000 ADU.
  5. Physics-first cross-check printout: `expected_ADU_per_s ≈ flux_ID03 × FOV_fraction × run_exposure_simulation()[1] × 2.14`; pull `flux_ID03` from the ID03 beamline page (record number + URL in the docstring) and compare order of magnitude.
- [ ] **Step 2:** Run it; pin the resulting value as `DetectorConfig.counts_scale` default (one-line change in `config.py`) with a comment `# derived 2026-06-XX by docs/calibration/derive_counts_scale.py`. Record inputs/outputs in `docs/detector-noise-model.md`.
- [ ] **Step 3:** Full suite (counts_scale default change must not break tests — no test may pin the literal; if one does, fix it to reference `DetectorConfig().counts_scale`).
- [ ] **Step 4: Commit** — `feat(detector): pin data-anchored counts_scale default (+derivation script)`

---

### Task 13: docs + calibration-script archive

**Files:**
- Create: `docs/detector-noise-model.md` (complete), `docs/calibration/{analyze_darks.py,fit_gain.py,quick_floor_fit.py}` (copied from `C:\Users\borgi\Documents\GM-reworked\noise_scratch\`, paths parameterized via a `DATA_ROOT` constant at top, header noting the raw data is local-only)
- Modify: `README.md` only if it mentions `[noise]` (grep first).

- [ ] **Step 1:** `docs/detector-noise-model.md` content: the spec §2 fit table verbatim, the model equation, the photon-transfer method summary (adjacent-pair differences, drift normalization, boxcar scene mask), the two fit figures (copy `photon_transfer_fit.png` + `darks_characterization.png` into `docs/img/`), counts_scale derivation record (from Task 12), and the `subtract_background` display recipe with the strong-beam caveat.
- [ ] **Step 2:** Copy + clean the three fitting scripts into `docs/calibration/`; each gets a module docstring: data location expectation, `pip install hdf5plugin` note, "provenance, not CI" disclaimer.
- [ ] **Step 3:** Full suite + mypy green (docs-only, but the scripts must at least import-check: `python -m compileall docs/calibration`).
- [ ] **Step 4: Commit** — `docs: detector noise model — fit provenance, figures, calibration scripts`

---

### Task 14: notebook 03 weak-beam display section (deferrable)

**Files:**
- Modify: `examples/03_dislocations_and_contrast.ipynb` (via its builder in `C:\Users\borgi\Documents\GM-reworked\m5_scratch\` — follow the M5 builder pattern; do NOT hand-edit executed notebooks)

- [ ] Add a short section after the existing weak-beam cells: simulate with the default detector model, display raw uint16 frame vs `subtract_background(frame)` side by side, one sentence on the measured noise model with a link to `docs/detector-noise-model.md`.
- [ ] Production-image rule applies ([[feedback-production-nrays-for-images]]): regenerate with Nrays=1e8 / batch_size=2e7 for the shipped images; nbmake must stay green (`...python.exe -m pytest --nbmake examples/03_dislocations_and_contrast.ipynb`).
- [ ] If the M5 builder round-trip is heavier than expected, STOP and surface to Sina rather than shipping down-tuned images — this task can ship in a follow-up PR.
- [ ] **Commit** — `docs(examples): weak-beam reduction + realistic detector section in notebook 03`

---

### Task 15: final gates

- [ ] `...python.exe -m pytest -q` — full suite green (compare failure SET against the pre-existing xfails, not just counts — [[preexisting-test-failures-2026-05-28]] rule).
- [ ] `...python.exe -m pytest -q -m slow` — slow suite green (includes the invariance + e2e uint16 tests).
- [ ] `...python.exe -m mypy src/dfxm_geo/` — 0 errors.
- [ ] `grep -rn "noise" src/dfxm_geo/data/configs/ configs/` — no `[noise]` stragglers.
- [ ] Review `git log --oneline main..HEAD` — coherent breaking-change story; update CLAUDE.md working notes (release labeling is Sina's call; no tag, no push without confirmation).
- [ ] **Commit any final fixes; do NOT merge to main** — hand off for review per `superpowers:finishing-a-development-branch`.

---

## Self-review notes (kept for the executor)

- Spec coverage: §3.1→Tasks 2–4, §3.2(a)→Task 5, §3.2(b,c)→Task 12, §3.3→Tasks 6+7+11, §3.4→Tasks 9+10, §3.5→Tasks 8+11, §3.6→Task 1 (+14), §3.7→tests within each task + Task 15, §3.8→Tasks 13+14. Out-of-scope items (ghost, PRNU rings, strong-beam physics) have no tasks — correct per spec §4.
- The Task 5 normalization changes absolute image levels everywhere; Task 12's counts_scale is derived AFTER it, so ordering 5 → 12 is mandatory.
- Task 7's interim shim keeps the old Poisson body alive between Tasks 7 and 9 only so the suite can gate Task 7; Task 9 deletes it. Do not skip the Task 7 gate.
- RNG: `SeedSequence.spawn(n)` children are stable under growth of n (spawn_key = (i,)), so adding child [2] for the sensor map cannot disturb children [0]/[1]. Param draws stay bit-identical for equal seeds.
