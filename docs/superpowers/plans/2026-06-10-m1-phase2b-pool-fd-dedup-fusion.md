# M1 Phase 2b (v2.6.0) Implementation Plan — pool + Fd dedup + fused Hg kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ≥5× identify-sweep throughput via (W2) per-dislocation Fd dedup, (W3) routing identify Hg computation through the fused numba kernel, and (W1) a persistent worker pool in `scripts/fanout.py`.

**Architecture:** One new seam `find_hg_scene()` in `crystal/dislocations.py` replaces every `Fd_find_mixed`/`Fd_find_multi_dislocs_mixed` + `fast_inverse2` composition in the three identify orchestrators; it has a NumPy engine (bit-identical to today, the parity oracle) and a numba engine (reusing the shipped Phase-1 `_population_hg_kernel` plus a new per-dislocation variant). Independently, `fanout.py` gains a pool mode (default) where `ProcessPoolExecutor` workers call `cli_main`/`cli_main_identify` in-process via a new `dfxm_geo/fanout_worker.py`; `--isolate` keeps the subprocess path.

**Tech Stack:** Python 3.12, numba `@njit`, `concurrent.futures.ProcessPoolExecutor`, pytest, h5py.

**Spec:** `docs/superpowers/specs/2026-06-10-m1-phase2b-pool-and-fd-dedup-design.md`

**Conventions for every task:**
- Python = `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe` (NEVER bare `python` in bash — that's py2.7). Run from repo root `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`.
- After each task: `python -m pytest -q <task's test file>` then commit. mypy/ruff run in Task 13 and must be clean at the end, but fix obvious typing as you go (`mypy src/dfxm_geo/` should stay at 0 errors).
- Tests that need the resolution kernel npz use the existing skip-guard pattern (see `tests/test_fanout.py:226-228`):
  ```python
  kernel_dir = Path(fm.pkl_fpath)
  if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
      pytest.skip("No bootstrapped kernel npz found; skipping integration run.")
  ```
- Scan grids in new e2e tests stay tiny (phi steps ≤ 2, n_samples = 1) — `Npixels` is a module constant (px510, not TOML-tunable), so frame count is the only lever.

**File structure (locked in):**

| File | Role |
|---|---|
| `src/dfxm_geo/crystal/dislocations.py` | + `find_hg_scene`, `_find_hg_scene_numpy`, `_specs_to_population_arrays`, `_scene_perdis_hg_kernel`, `_write_hg_from_f` |
| `src/dfxm_geo/pipeline.py` | 3 orchestrators routed through `find_hg_scene`; nothing else |
| `src/dfxm_geo/fanout_worker.py` | NEW — in-process pool worker `run_one` (picklable, package-importable for Windows spawn) |
| `scripts/fanout.py` | + `run_pool`, `--isolate`, env-pinning context manager, `write_timing_json` isolate field |
| `tests/test_hg_scene.py` | NEW — W2/W3 unit + parity tests (no kernel needed) |
| `tests/test_identify_dedup.py` | NEW — orchestrator-level Fd-call-count tests (kernel-guarded) |
| `tests/test_fanout_worker.py` | NEW — `run_one` unit tests |
| `tests/test_fanout.py` | + pool-path tests + pool-vs-isolate bit-identity e2e (slow) |
| `lsf/fanout.bsub` | comment documenting `--isolate` |
| `docs/cluster-profiling.md` | Phase 2b post-optimization section |

---

## Task 1: `find_hg_scene` NumPy engine (W2 core)

The single seam every identify Hg computation will go through. The NumPy
engine must be **bit-identical** to the three compositions used today
(`pipeline.py:1673-1680`, `1872-1873` + `1888-1905`, `2117-2166`).

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (after `Fd_find_multi_dislocs_mixed`, ~line 398)
- Test: `tests/test_hg_scene.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hg_scene.py`:

```python
"""find_hg_scene — the single Hg seam for the identify orchestrators (v2.6.0).

NumPy-engine tests prove bit-identity with the legacy compositions; numba
engine tests (added in Tasks 5-6) prove parity at tight tolerance.
No resolution kernel needed: pure crystal/dislocations level.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import (
    Fd_find_mixed,
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
    find_hg_scene,
)
from dfxm_geo.crystal.rotations import fast_inverse2


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation via QR (det +1)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.fixture()
def scene():
    rng = np.random.default_rng(42)
    rl_um = rng.uniform(-50.0, 50.0, size=(3, 200))
    Us = _rand_rotation(rng)
    Theta = _rand_rotation(rng)
    specs = [
        MixedDislocSpec(
            Ud_mix=_rand_rotation(rng),
            rotation_deg=float(rng.uniform(0, 180)),
            position_lab_um=(1.5, -2.0, 0.5),
        ),
        MixedDislocSpec(
            Ud_mix=_rand_rotation(rng),
            rotation_deg=float(rng.uniform(0, 180)),
            position_lab_um=(-3.0, 1.0, 0.0),
        ),
    ]
    return rl_um, Us, Theta, specs


def _legacy_hg_single(rl_um, Us, spec, Theta):
    """Exactly the single-dislocation composition at pipeline.py:1673-1680."""
    Fg = Fd_find_mixed(
        rl_um,
        Us,
        Ud_mix=spec.Ud_mix,
        rotation_deg=spec.rotation_deg,
        Theta=Theta,
        position_lab_um=spec.position_lab_um,
    )
    return np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)


def test_single_spec_bit_identical_to_legacy(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numpy")
    expected = _legacy_hg_single(rl_um, Us, specs[0], Theta)
    assert solos is None
    np.testing.assert_array_equal(hg, expected)  # BIT-identical


def test_combined_bit_identical_to_multi_dislocs(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    Fg = Fd_find_multi_dislocs_mixed(rl_um, Us, specs, Theta)
    expected = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
    assert solos is None
    np.testing.assert_array_equal(hg, expected)


def test_per_dislocation_solos_bit_identical(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(
        rl_um, Us, specs, Theta, per_dislocation=True, engine="numpy"
    )
    assert solos is not None and len(solos) == 2
    for spec, solo in zip(specs, solos):
        np.testing.assert_array_equal(solo, _legacy_hg_single(rl_um, Us, spec, Theta))
    # combined unchanged by requesting components
    hg2, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    np.testing.assert_array_equal(hg, hg2)


def test_empty_specs_raises(scene):
    rl_um, Us, Theta, _ = scene
    with pytest.raises(ValueError):
        find_hg_scene(rl_um, Us, [], Theta, engine="numpy")


def test_unknown_engine_raises(scene):
    rl_um, Us, Theta, specs = scene
    with pytest.raises(ValueError):
        find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="fortran")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hg_scene.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_hg_scene'`

- [ ] **Step 3: Implement `find_hg_scene` + `_find_hg_scene_numpy`**

In `src/dfxm_geo/crystal/dislocations.py`, after `find_hg_population` (end of file). Note `fast_inverse2` must be imported at the top of the module (`from dfxm_geo.crystal.rotations import fast_inverse2` — check it isn't already; `rotations` imports nothing from `dislocations`, so no cycle):

```python
def find_hg_scene(
    rl_um: np.ndarray,
    Us: np.ndarray,
    specs: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    per_dislocation: bool = False,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    S: np.ndarray = _S_IDENTITY,
    engine: str = "numpy",  # flipped to "numba" in v2.6.0 W3 (Task 7)
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Hg for a scene of mixed dislocations, optionally with per-dislocation Hg.

    The single seam used by all three identify orchestrators (v2.6.0 W2/W3).
    Replaces the per-call composition
    ``transpose(fast_inverse2(Fd_find_*)) - I`` and, when
    ``per_dislocation=True``, computes each dislocation's field ONCE — the
    combined scene is derived as ``Σ(Fg_i − I) + I`` (the exact accumulation
    `Fd_find_multi_dislocs_mixed` performs), so the solo renders come for
    free instead of doubling the Fd work (spec §4).

    Args:
        rl_um: Lab-frame ray grid, shape (3, X), MICROMETRES.
        Us: Sample-to-grain rotation, (3, 3).
        specs: ≥1 dislocation specs (Ud_mix, rotation_deg, position_lab_um).
        Theta: Lab-to-sample rotation, (3, 3).
        per_dislocation: also return each dislocation's solo Hg.
        b, ny, S: as in `Fd_find_mixed`.
        engine: "numpy" (bit-identical legacy composition, the parity
            oracle) or "numba" (fused kernel, parity ≤1e-12).

    Returns:
        (Hg_combined, solos) — solos is a list of (X, 3, 3) arrays when
        ``per_dislocation`` else None.
    """
    if not specs:
        raise ValueError("find_hg_scene requires at least one dislocation spec")
    if engine == "numpy":
        return _find_hg_scene_numpy(
            rl_um, Us, specs, Theta, per_dislocation=per_dislocation, b=b, ny=ny, S=S
        )
    raise ValueError(f"unknown engine {engine!r}; expected 'numpy' or 'numba'")


def _find_hg_scene_numpy(
    rl_um: np.ndarray,
    Us: np.ndarray,
    specs: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    per_dislocation: bool,
    b: float,
    ny: float,
    S: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Bit-identical NumPy oracle: exactly the legacy call-site composition."""
    identity = np.identity(3)

    def _hg(Fg: np.ndarray) -> np.ndarray:
        return np.transpose(fast_inverse2(Fg), [0, 2, 1]) - identity

    if len(specs) == 1 and not per_dislocation:
        spec = specs[0]
        Fg = Fd_find_mixed(
            rl_um,
            Us,
            Ud_mix=spec.Ud_mix,
            rotation_deg=spec.rotation_deg,
            Theta=Theta,
            b=b,
            ny=ny,
            position_lab_um=spec.position_lab_um,
            S=S,
        )
        return _hg(Fg), None

    parts: list[np.ndarray] = [
        Fd_find_mixed(
            rl_um,
            Us,
            Ud_mix=spec.Ud_mix,
            rotation_deg=spec.rotation_deg,
            Theta=Theta,
            b=b,
            ny=ny,
            position_lab_um=spec.position_lab_um,
            S=S,
        )
        for spec in specs
    ]
    # Same accumulation order as Fd_find_multi_dislocs_mixed: zeros, then
    # += (Fg_i − I) in spec order, then + I — keeps the combined result
    # bit-identical to the legacy path.
    Fg_sum = np.zeros((rl_um.shape[1], 3, 3))
    for Fg_one in parts:
        Fg_sum += Fg_one - identity
    hg_combined = _hg(Fg_sum + identity)
    if not per_dislocation:
        return hg_combined, None
    # Convert each part to its solo Hg, releasing the (X,3,3) Fg as we go —
    # bounds the transient peak at px510 (~106 MB per array).
    solos: list[np.ndarray] = []
    for i in range(len(parts)):
        solos.append(_hg(parts[i]))
        parts[i] = None  # type: ignore[call-overload]
    return hg_combined, solos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hg_scene.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_hg_scene.py src/dfxm_geo/crystal/dislocations.py
git commit -m "feat(w2): find_hg_scene — single Hg seam, NumPy engine bit-identical to legacy"
```

---

## Task 2: Route `_iter_identification_multi` through `find_hg_scene`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:1872-1909` (and the import block at line 37-44)
- Test: `tests/test_identify_dedup.py` (new, kernel-guarded)

- [ ] **Step 1: Write the failing test**

Create `tests/test_identify_dedup.py`:

```python
"""W2 dedup contract at the orchestrator level.

The orchestrators must delegate each scene (combined + optional solos) to
exactly ONE `find_hg_scene` call with `per_dislocation` mirroring the
config flag — the pre-W2 code instead made one combined call plus one
recompute per solo render. The single-evaluation-per-dislocation property
*inside* the seam is proven engine-independently in tests/test_hg_scene.py;
spying on the seam (not on Fd_find_mixed) keeps this test valid after the
W3 engine flip routes the math through the numba kernel.

Needs the bootstrapped kernel npz (ctx construction loads the resolution LUT).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.dislocations import find_hg_scene as _real_scene

kernel_dir = Path(fm.pkl_fpath)
pytestmark = pytest.mark.skipif(
    not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")),
    reason="No bootstrapped kernel npz found.",
)

_MULTI_TOML = """\
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[noise]
poisson_noise = true
rng_seed = 7
intensity_scale = 7.0

[multi]
n_samples = 1
pos_std_um = 5.0
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""


def _spy_scene_calls(monkeypatch, toml_text: str, tmp_path: Path) -> list[dict]:
    """Run one identify config with find_hg_scene wrapped in a recording spy."""
    import dfxm_geo.pipeline as pipeline

    calls: list[dict] = []

    def spy(rl_um, Us, specs, Theta, **kwargs):
        calls.append(
            {
                "n_specs": len(specs),
                "per_dislocation": kwargs.get("per_dislocation", False),
            }
        )
        return _real_scene(rl_um, Us, specs, Theta, **kwargs)

    # The orchestrators call the name imported into the pipeline namespace.
    monkeypatch.setattr(pipeline, "find_hg_scene", spy)

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    cfg = pipeline.load_identification_config(cfg_path)
    pipeline.run_identification(cfg, tmp_path / "out")
    return calls


def test_multi_render_per_dislocation_is_one_scene_call(monkeypatch, tmp_path):
    # 1 sample × 2 dislocations, render_per_dislocation=True → exactly ONE
    # find_hg_scene call carrying both specs and per_dislocation=True.
    # Pre-W2 the orchestrator made 1 combined + 2 solo Fd computations and
    # never touched find_hg_scene (this test fails with AttributeError there).
    calls = _spy_scene_calls(monkeypatch, _MULTI_TOML, tmp_path)
    assert calls == [{"n_specs": 2, "per_dislocation": True}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_identify_dedup.py -v`
Expected: FAIL — `AttributeError: <module 'dfxm_geo.pipeline'> does not have the attribute 'find_hg_scene'` (the orchestrator doesn't use the seam yet). If it SKIPs instead, the kernel npz is missing — stop and check `reciprocal_space/pkl_files/` before proceeding; this task cannot be verified kernel-less.

- [ ] **Step 3: Rewire the multi orchestrator**

In `src/dfxm_geo/pipeline.py`, replace lines 1872-1909 (from `Fg_combined = Fd_find_multi_dislocs_mixed(...)` through `detectors["dfxm_sim_detector_dis1"] = dis1_args`) with:

```python
            # W2 dedup (v2.6.0): each dislocation's field is computed ONCE;
            # the combined scene is Σ(Fg_i − I) + I — bit-identical to the
            # old Fd_find_multi_dislocs_mixed + per-solo recompute path.
            Hg_combined, solo_hgs = find_hg_scene(
                rl_eff,
                Us_,
                specs,
                Theta_,
                per_dislocation=mc.render_per_dislocation,
            )

            combined_args, positioners = _scan_frames_args(
                Hg_combined, frames_at_z, config.scan, ctx
            )
            detectors: dict[
                str, list[tuple[int, np.ndarray, float, float, float, fm.ForwardContext]]
            ] = {
                "dfxm_sim_detector": combined_args,
            }

            if mc.render_per_dislocation:
                # Per-dislocation Hg: each rendered alone (other one absent), at
                # its own scene position so the renders overlay the combined
                # image as ground-truth instance labels. Noiseless by design.
                assert solo_hgs is not None
                dis0_args, _ = _scan_frames_args(solo_hgs[0], frames_at_z, config.scan, ctx)
                dis1_args, _ = _scan_frames_args(solo_hgs[1], frames_at_z, config.scan, ctx)
                detectors["dfxm_sim_detector_dis0"] = dis0_args
                detectors["dfxm_sim_detector_dis1"] = dis1_args
```

Update the import block at `pipeline.py:37-44`: add `find_hg_scene` to the
`from dfxm_geo.crystal.dislocations import (...)` tuple. Do NOT remove
`Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` / `fast_inverse2` yet —
single and zscan still use them (removed in Task 4).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_identify_dedup.py tests/test_hg_scene.py -v`
Expected: PASS.
Then the identify e2e safety net: `python -m pytest -q -k "identification or identify"`
Expected: same pass/skip set as before this task (run it on the base commit first if unsure).

- [ ] **Step 5: Commit**

```bash
git add tests/test_identify_dedup.py src/dfxm_geo/pipeline.py
git commit -m "feat(w2): multi orchestrator computes each per-dislocation Fd once (4->2 per sample)"
```

---

## Task 3: Route `_iter_identification_zscan` through `find_hg_scene`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:2110-2168`
- Test: `tests/test_identify_dedup.py` (extend)

- [ ] **Step 1: Write the failing test** (append to `tests/test_identify_dedup.py`)

```python
_ZSCAN_TOML = """\
mode = "z-scan"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[noise]
poisson_noise = true
rng_seed = 7
intensity_scale = 7.0

[crystal]
sweep_all_slip_planes = false
slip_plane_normal = [1, 1, 1]
b_vector_indices = [0]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 30.0

[zscan]
z_offsets_um = [0.0]
include_secondary = true
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""


def test_zscan_render_per_dislocation_is_one_scene_call(monkeypatch, tmp_path):
    # 1 z × 1 plane × 1 b × 1 angle, secondary on, render_per_dislocation on
    # → exactly ONE find_hg_scene call with both specs and per_dislocation=True.
    # Pre-W2: 1 combined + 2 solo Fd computations, zero seam calls.
    calls = _spy_scene_calls(monkeypatch, _ZSCAN_TOML, tmp_path)
    assert calls == [{"n_specs": 2, "per_dislocation": True}]
```

NOTE: if `load_identification_config` rejects any `[crystal]` key above
(schema drift), copy the exact key names from
`_identification_config_to_toml_str` (`pipeline.py:~1540-1562`) — the
round-trip serializer is the schema's source of truth.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_identify_dedup.py::test_zscan_render_per_dislocation_is_one_scene_call -v`
Expected: FAIL — the spy records `[]` (zscan still on the legacy composition), so `assert [] == [...]`

- [ ] **Step 3: Rewire the zscan orchestrator**

In `src/dfxm_geo/pipeline.py`, replace lines 2110-2168 (the whole
`if zscan.include_secondary:` / `else:` block down to and including
`detectors["dfxm_sim_detector"] = args_list`) with:

```python
                    if zscan.include_secondary:
                        sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)
                        secondary_spec = MixedDislocSpec(
                            Ud_mix=sec["Ud"],
                            rotation_deg=sec["alpha_deg"],
                            position_lab_um=sec["pos_um"],
                        )
                        sample["secondary"] = _build_dislocation_sample_entry(sec)
                        # W2 dedup: primary+secondary fields computed once;
                        # combined = Σ(Fg_i − I) + I. The solo secondary used
                        # to be rendered without its position offset — drawn
                        # at pos_std_um=0.0 the offset is (0,0,0), so passing
                        # it through the spec is bit-identical.
                        Hg, solo_hgs = find_hg_scene(
                            rl_shifted,
                            Us_,
                            [primary_spec, secondary_spec],
                            Theta_,
                            per_dislocation=zscan.render_per_dislocation,
                        )
                        if zscan.render_per_dislocation:
                            # Primary + secondary each rendered alone (noiseless
                            # ground-truth instance labels). Bypass the Poisson
                            # pass, which only touches `dfxm_sim_detector`.
                            assert solo_hgs is not None
                            prim_args, _ = _scan_frames_args(
                                solo_hgs[0], frames_at_z, config.scan, ctx
                            )
                            sec_args, _ = _scan_frames_args(
                                solo_hgs[1], frames_at_z, config.scan, ctx
                            )
                            detectors["dfxm_sim_detector_primary"] = prim_args
                            detectors["dfxm_sim_detector_secondary"] = sec_args
                    else:
                        Hg, _ = find_hg_scene(
                            rl_shifted,
                            Us_,
                            [primary_spec],
                            Theta_,
                        )

                    args_list, positioners = _scan_frames_args(Hg, frames_at_z, config.scan, ctx)
                    detectors["dfxm_sim_detector"] = args_list
```

(Note the `sample["secondary"] = ...` line moves up so the secondary entry
is recorded before the Hg computation — behaviour identical, RNG order
unchanged because `_draw_dislocation` still happens first.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_identify_dedup.py tests/test_hg_scene.py -v` → PASS
Run: `python -m pytest -q -k "zscan or z_scan"` → same pass/skip set as base.

- [ ] **Step 5: Commit**

```bash
git add tests/test_identify_dedup.py src/dfxm_geo/pipeline.py
git commit -m "feat(w2): zscan orchestrator dedups primary/secondary Fd via find_hg_scene"
```

---

## Task 4: Route `_iter_identification_single` + retire pipeline's direct Fd imports

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:1673-1680` and the import block (lines 37-44)

Single mode has no duplication (one Fd per (z, plane, b, α)) — this routing
is for uniformity so W3's fused engine accelerates all three modes from one
switch.

- [ ] **Step 1: Rewire the single orchestrator**

In `src/dfxm_geo/pipeline.py`, replace lines 1673-1680:

```python
                    Fg = Fd_find_mixed(
                        rl_eff,
                        Us_,
                        Ud_mix=Ud_mix,
                        rotation_deg=float(alpha),
                        Theta=Theta_,
                    )
                    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
```

with:

```python
                    Hg, _ = find_hg_scene(
                        rl_eff,
                        Us_,
                        [
                            MixedDislocSpec(
                                Ud_mix=Ud_mix,
                                rotation_deg=float(alpha),
                            )
                        ],
                        Theta_,
                    )
```

- [ ] **Step 2: Clean the pipeline import block**

`grep -n "Fd_find_mixed\|Fd_find_multi_dislocs_mixed\|fast_inverse2" src/dfxm_geo/pipeline.py`
— if (as expected) no call sites remain, remove `Fd_find_mixed` and
`Fd_find_multi_dislocs_mixed` from the dislocations import tuple and delete
the `from dfxm_geo.crystal.rotations import fast_inverse2` line
(`pipeline.py:44`). Keep `MixedDislocSpec` and `_draw_dislocation`-related
imports. If any other use of `fast_inverse2` remains in pipeline.py, keep
that import.

- [ ] **Step 3: Run the bit-identity safety net**

Run: `python -m pytest -q -k "identification or identify"` → same pass/skip
set as base (single-spec NumPy engine is the exact legacy composition, so
HDF5-golden tests must be untouched).
Run: `python -m mypy src/dfxm_geo/` → 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/dfxm_geo/pipeline.py
git commit -m "refactor(w2): single orchestrator through find_hg_scene; drop direct Fd imports"
```

---

## Task 5: W3 adapter — numba engine for the combined-only path

Reuses the shipped Phase-1 `_population_hg_kernel` via `find_hg_population`
(`dislocations.py:523`). No new kernel yet — only spec→arrays adaptation.

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py`
- Test: `tests/test_hg_scene.py` (extend)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hg_scene.py`)

```python
NUMBA_RTOL = 1e-12
NUMBA_ATOL = 1e-14
# Engines differ in FP op order (fast_inverse2 is fastmath=True; the fused
# kernel's inline inverse is fastmath=False) — parity, not bit-identity.
# Tolerances follow the Phase-1 population-kernel parity precedent.


def test_numba_combined_matches_numpy_two_specs(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    hg_nb, solos = find_hg_scene(rl_um, Us, specs, Theta, engine="numba")
    assert solos is None
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)


def test_numba_combined_matches_numpy_single_spec(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, _ = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numpy")
    hg_nb, _ = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numba")
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_hg_scene.py -v -k numba`
Expected: FAIL — `ValueError: unknown engine 'numba'`

- [ ] **Step 3: Implement the adapter + numba dispatch**

In `src/dfxm_geo/crystal/dislocations.py`:

```python
def _specs_to_population_arrays(
    specs: list[MixedDislocSpec],
    Us: np.ndarray,
    Theta: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack MixedDislocSpec list into the array layout the fused kernels take.

    M_d = Ud_d.T @ Us.T @ S.T @ Theta — the rl→rd transform of
    `Fd_find_mixed` (its offset subtraction happens in lab frame BEFORE
    Theta, matching the kernels' ``rl − offset`` then ``M @ ·``).
    """
    n = len(specs)
    M = np.empty((n, 3, 3))
    offset = np.empty((n, 3))
    Ud = np.empty((n, 3, 3))
    cos_rot = np.empty(n)
    sin_rot = np.empty(n)
    base = Us.T @ S.T @ Theta
    for d, spec in enumerate(specs):
        Ud[d] = spec.Ud_mix
        M[d] = spec.Ud_mix.T @ base
        offset[d] = spec.position_lab_um
        cos_rot[d] = np.cos(np.deg2rad(spec.rotation_deg))
        sin_rot[d] = np.sin(np.deg2rad(spec.rotation_deg))
    return M, offset, Ud, cos_rot, sin_rot
```

In `find_hg_scene`, replace the final `raise ValueError(...)` with:

```python
    if engine != "numba":
        raise ValueError(f"unknown engine {engine!r}; expected 'numpy' or 'numba'")
    M, offset, Ud, cos_rot, sin_rot = _specs_to_population_arrays(specs, Us, Theta, S)
    if not per_dislocation:
        return find_hg_population(rl_um, M, offset, Ud, cos_rot, sin_rot, b=b, ny=ny), None
    return _find_hg_scene_perdis_numba(rl_um, M, offset, Ud, cos_rot, sin_rot, b=b, ny=ny)
```

and add a stub so this task stays scoped to the combined path:

```python
def _find_hg_scene_perdis_numba(
    rl_um: np.ndarray,
    M: np.ndarray,
    offset: np.ndarray,
    Ud: np.ndarray,
    cos_rot: np.ndarray,
    sin_rot: np.ndarray,
    *,
    b: float,
    ny: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    raise NotImplementedError("per-dislocation numba path lands in the next task")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_hg_scene.py -v` → all PASS (perdis numba
not exercised yet).

- [ ] **Step 5: Commit**

```bash
git add tests/test_hg_scene.py src/dfxm_geo/crystal/dislocations.py
git commit -m "feat(w3): numba engine for find_hg_scene combined path via _population_hg_kernel"
```

---

## Task 6: W3 per-dislocation fused kernel

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py`
- Test: `tests/test_hg_scene.py` (extend)

- [ ] **Step 1: Write the failing test** (append to `tests/test_hg_scene.py`)

```python
def test_numba_per_dislocation_matches_numpy(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, solos_np = find_hg_scene(
        rl_um, Us, specs, Theta, per_dislocation=True, engine="numpy"
    )
    hg_nb, solos_nb = find_hg_scene(
        rl_um, Us, specs, Theta, per_dislocation=True, engine="numba"
    )
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)
    assert solos_nb is not None and len(solos_nb) == len(solos_np)
    for nb, np_ in zip(solos_nb, solos_np):
        np.testing.assert_allclose(nb, np_, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)


def test_numba_perdis_combined_equals_combined_only(scene):
    # Requesting components must not change the combined result (numba path).
    rl_um, Us, Theta, specs = scene
    hg_only, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numba")
    hg_with, _ = find_hg_scene(
        rl_um, Us, specs, Theta, per_dislocation=True, engine="numba"
    )
    np.testing.assert_array_equal(hg_only, hg_with)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_hg_scene.py -v -k perdis`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement the kernel**

In `src/dfxm_geo/crystal/dislocations.py`, directly after
`_population_hg_kernel`. The math is `_population_hg_kernel`
(`dislocations.py:400-520`) verbatim, refactored only to (a) keep each
dislocation's contribution `g` separate so the solo `Fg_d = I + g_d` can be
inverted in the same pass, and (b) share the inverse-transpose-minus-I tail
through an `inline="always"` helper:

```python
@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _write_hg_from_f(f00, f01, f02, f10, f11, f12, f20, f21, f22, out, x):
    """out[x] = inv(F).T − I for one ray's 3×3 F (mirrors fast_inverse2)."""
    c00 = f11 * f22 - f12 * f21
    c10 = -(f10 * f22 - f12 * f20)
    c20 = f10 * f21 - f11 * f20
    idet = 1.0 / (f00 * c00 + f01 * c10 + f02 * c20)
    out[x, 0, 0] = c00 * idet - 1.0
    out[x, 0, 1] = c10 * idet
    out[x, 0, 2] = c20 * idet
    out[x, 1, 0] = -idet * (f01 * f22 - f02 * f21)
    out[x, 1, 1] = idet * (f00 * f22 - f02 * f20) - 1.0
    out[x, 1, 2] = -idet * (f00 * f21 - f01 * f20)
    out[x, 2, 0] = idet * (f01 * f12 - f02 * f11)
    out[x, 2, 1] = -idet * (f00 * f12 - f02 * f10)
    out[x, 2, 2] = idet * (f00 * f11 - f01 * f10) - 1.0


@njit(cache=True, nogil=True, fastmath=False)
def _scene_perdis_hg_kernel(
    rl_um: np.ndarray,  # (3, X) float64, MICROMETRES
    M: np.ndarray,  # (N, 3, 3) = Ud.T @ Us.T @ S.T @ Theta
    offset: np.ndarray,  # (N, 3) micrometres
    Ud: np.ndarray,  # (N, 3, 3)
    cos_rot: np.ndarray,  # (N,)
    sin_rot: np.ndarray,  # (N,)
    b: float,
    ny: float,
    Hg_combined_out: np.ndarray,  # (X, 3, 3), written in place
    Hg_per_out: np.ndarray,  # (N, X, 3, 3), written in place
) -> None:
    """`_population_hg_kernel` + per-dislocation solo Hg in the same ray pass.

    Each dislocation's field is evaluated ONCE; its contribution
    g_d = Ud_d @ G_d @ Ud_d.T feeds both the combined Fg = I + Σ g_d and the
    solo Fg_d = I + g_d (the W2 dedup, fused). Math is identical to
    `_population_hg_kernel`; see that kernel's docstring for the reference.
    """
    X = rl_um.shape[1]
    N = M.shape[0]
    alpha = 1e-20
    bf = b / (4.0 * math.pi * (1.0 - ny))
    bf1 = b / (2.0 * math.pi)

    G = np.zeros((3, 3))
    Tmp = np.zeros((3, 3))
    g = np.zeros((3, 3))  # one dislocation's grain-frame contribution

    for x in range(X):
        rx = rl_um[0, x]
        ry = rl_um[1, x]
        rz = rl_um[2, x]

        f00 = 1.0
        f01 = 0.0
        f02 = 0.0
        f10 = 0.0
        f11 = 1.0
        f12 = 0.0
        f20 = 0.0
        f21 = 0.0
        f22 = 1.0

        for d in range(N):
            dx = rx - offset[d, 0]
            dy = ry - offset[d, 1]
            dz = rz - offset[d, 2]
            rd0 = M[d, 0, 0] * dx + M[d, 0, 1] * dy + M[d, 0, 2] * dz
            rd1 = M[d, 1, 0] * dx + M[d, 1, 1] * dy + M[d, 1, 2] * dz
            rd2 = M[d, 2, 0] * dx + M[d, 2, 1] * dy + M[d, 2, 2] * dz

            sqx = rd0 * rd0
            sqy = rd1 * rd1
            sqz = rd2 * rd2
            denom = (sqx + sqy) * (sqx + sqy) + alpha
            nyf = 2.0 * ny * (sqx + sqy)
            c = cos_rot[d]
            s = sin_rot[d]
            denom1 = sqz + sqy + alpha

            G[0, 0] = -rd1 * (3.0 * sqx + sqy - nyf) / denom * bf * c
            G[0, 1] = rd0 * (3.0 * sqx + sqy - nyf) / denom * bf * c + (-rd2 / denom1) * bf1 * s
            G[0, 2] = (rd1 / denom1) * bf1 * s
            G[1, 0] = -rd0 * (3.0 * sqy + sqx - nyf) / denom * bf * c
            G[1, 1] = rd1 * (sqx - sqy + nyf) / denom * bf * c
            G[1, 2] = 0.0
            G[2, 0] = 0.0
            G[2, 1] = 0.0
            G[2, 2] = 0.0

            for a in range(3):
                for col in range(3):
                    acc = 0.0
                    for j in range(3):
                        acc += G[a, j] * Ud[d, col, j]
                    Tmp[a, col] = acc

            for row in range(3):
                for col in range(3):
                    acc = 0.0
                    for j in range(3):
                        acc += Ud[d, row, j] * Tmp[j, col]
                    g[row, col] = acc

            f00 += g[0, 0]
            f01 += g[0, 1]
            f02 += g[0, 2]
            f10 += g[1, 0]
            f11 += g[1, 1]
            f12 += g[1, 2]
            f20 += g[2, 0]
            f21 += g[2, 1]
            f22 += g[2, 2]

            _write_hg_from_f(
                1.0 + g[0, 0],
                g[0, 1],
                g[0, 2],
                g[1, 0],
                1.0 + g[1, 1],
                g[1, 2],
                g[2, 0],
                g[2, 1],
                1.0 + g[2, 2],
                Hg_per_out[d],
                x,
            )

        _write_hg_from_f(
            f00, f01, f02, f10, f11, f12, f20, f21, f22, Hg_combined_out, x
        )
```

Replace the `_find_hg_scene_perdis_numba` stub body:

```python
    X = rl_um.shape[1]
    n = M.shape[0]
    Hg_combined = np.empty((X, 3, 3), dtype=np.float64)
    Hg_per = np.empty((n, X, 3, 3), dtype=np.float64)
    _scene_perdis_hg_kernel(
        np.ascontiguousarray(rl_um, dtype=np.float64),
        np.ascontiguousarray(M, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(Ud, dtype=np.float64),
        np.ascontiguousarray(cos_rot, dtype=np.float64),
        np.ascontiguousarray(sin_rot, dtype=np.float64),
        float(b),
        float(ny),
        Hg_combined,
        Hg_per,
    )
    return Hg_combined, [Hg_per[d] for d in range(n)]
```

CAUTION (parity risk to check if the perdis parity test fails on the
COMBINED array while solos pass): `_population_hg_kernel` accumulates the
combined F directly from `Ud @ Tmp` element sums, while this kernel goes
through the `g` scratch matrix. Same operations, same order — but if numba
fuses differently, make `_write_hg_from_f`-style accumulation match by
summing into `f..` from the same expressions the population kernel uses
(copy its nine `f.. +=` lines verbatim and read `g[row, col]` only for the
solo write). The committed version must pass BOTH parity tests at
rtol=1e-12.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_hg_scene.py -v`
Expected: all PASS (first run pays ~10 s numba JIT).

- [ ] **Step 5: Commit**

```bash
git add tests/test_hg_scene.py src/dfxm_geo/crystal/dislocations.py
git commit -m "feat(w3): fused per-dislocation scene kernel (combined + solo Hg, one ray pass)"
```

---

## Task 7: Flip `find_hg_scene` default engine to numba + golden inventory

This is the commit where identify outputs change at the ~1e-15 level
(spec §5). Keep it isolated so it's bisectable/revertible.

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (one default)
- Possibly regenerate: any identify golden the inventory flags

- [ ] **Step 1: Flip the default**

In `find_hg_scene`, change `engine: str = "numpy"` to
`engine: str = "numba"` and update the comment to
`# numpy = the bit-exact legacy oracle (parity tests)`.

- [ ] **Step 2: Inventory output-pinning tests**

Run the FULL kernel-present suite: `python -m pytest -q -m "slow or not slow"`
(expect ~10-25 min). Collect any new failures vs the Task 4 baseline.

For each failure, classify:
- asserts exact equality against stored identify output (golden npy/HDF5) →
  regenerate the golden **in this same commit**, with the regeneration
  command recorded in the commit message;
- asserts physics at tolerance → must pass; if it doesn't, the kernel has a
  real bug — stop and debug (do NOT loosen tolerances).

Forward-mode goldens (`Fd_find_smoke.npy`, pickle-era snapshots) go through
`Fd_find`/`find_hg_population`, NOT `Fd_find_mixed` — they must be
untouched. If a forward golden fails, something is wrong — stop.

- [ ] **Step 3: Quick perf sanity (informational, recorded in commit message)**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_identify.py
```
Compare the Hg-stage seconds against the baseline table in
`docs/cluster-profiling.md` (single 5.08 s / multi 7.50 s / zscan 5.54 s).
Expect a 2-5× drop on the Hg stage.

- [ ] **Step 4: Run the full suite again if goldens were regenerated** → green.

- [ ] **Step 5: Commit**

```bash
git add -A src/dfxm_geo tests
git commit -m "feat(w3): identify Hg default engine -> fused numba kernel"
```

---

## Task 8: `dfxm_geo/fanout_worker.py` — in-process pool worker (W1)

**Files:**
- Create: `src/dfxm_geo/fanout_worker.py`
- Test: `tests/test_fanout_worker.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fanout_worker.py`:

```python
"""dfxm_geo.fanout_worker.run_one — the pool-mode in-process config runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfxm_geo import fanout_worker


@pytest.fixture(autouse=True)
def _reset_import_timer(monkeypatch):
    # Each test sees a "fresh worker": first run_one reports a real import_s.
    monkeypatch.setattr(fanout_worker, "_import_timed", False)


def test_run_one_rejects_unknown_mode(tmp_path: Path):
    res = fanout_worker.run_one(
        "nonsense", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log")
    )
    assert res["returncode"] != 0
    assert "nonsense" in (tmp_path / "c.log").read_text(encoding="utf-8")


def test_run_one_calls_identify_cli_and_writes_timing(tmp_path: Path, monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_cli(argv):
        seen["argv"] = argv
        print("Wrote 4 samples to somewhere")
        return 0

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (fake_cli, []))
    log = tmp_path / "c.log"
    res = fanout_worker.run_one("identify", "cfg.toml", str(tmp_path / "out"), str(log))
    assert res["returncode"] == 0
    assert res["wall_s"] >= 0
    assert seen["argv"][:4] == ["--config", "cfg.toml", "--output", str(tmp_path / "out")]
    text = log.read_text(encoding="utf-8")
    assert "Wrote 4 samples" in text  # stdout redirected into the log
    timing_line = [ln for ln in text.splitlines() if ln.startswith("DFXM_TIMING ")][-1]
    payload = json.loads(timing_line[len("DFXM_TIMING ") :])
    assert payload["run_s"] >= 0
    assert payload["import_s"] >= 0


def test_run_one_first_call_reports_import_then_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (lambda argv: 0, []))

    def timing_from(log: Path) -> dict:
        line = [
            ln
            for ln in log.read_text(encoding="utf-8").splitlines()
            if ln.startswith("DFXM_TIMING ")
        ][-1]
        return json.loads(line[len("DFXM_TIMING ") :])

    log1, log2 = tmp_path / "a.log", tmp_path / "b.log"
    fanout_worker.run_one("identify", "a.toml", str(tmp_path / "o1"), str(log1))
    fanout_worker.run_one("identify", "b.toml", str(tmp_path / "o2"), str(log2))
    # pipeline is already imported in the test process, so the first measured
    # value may be ~0 — the contract is: second call reports exactly 0.0.
    assert timing_from(log2)["import_s"] == 0.0


def test_run_one_exception_is_rc_minus_one_with_traceback(tmp_path: Path, monkeypatch):
    def boom(argv):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (boom, []))
    log = tmp_path / "c.log"
    res = fanout_worker.run_one("identify", "c.toml", str(tmp_path / "out"), str(log))
    assert res["returncode"] == -1
    assert "kaboom" in log.read_text(encoding="utf-8")


def test_run_one_systemexit_code_is_propagated(tmp_path: Path, monkeypatch):
    def argparse_style_exit(argv):
        raise SystemExit(2)

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (argparse_style_exit, []))
    res = fanout_worker.run_one(
        "identify", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log")
    )
    assert res["returncode"] == 2


def test_forward_mode_appends_no_postprocess(tmp_path: Path, monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_cli(argv):
        seen["argv"] = argv
        return 0

    # forward mode resolves with the --no-postprocess extra
    monkeypatch.setattr(
        fanout_worker, "_resolve_cli", lambda mode: (fake_cli, ["--no-postprocess"])
    )
    fanout_worker.run_one("forward", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log"))
    assert seen["argv"][-1] == "--no-postprocess"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_fanout_worker.py -v`
Expected: FAIL — `ImportError: cannot import name 'fanout_worker'`

- [ ] **Step 3: Implement**

Create `src/dfxm_geo/fanout_worker.py`:

```python
"""In-process config runner for `scripts/fanout.py` pool mode (v2.6.0 W1).

Lives inside the package (not in scripts/) so `ProcessPoolExecutor` can
pickle `run_one` by module path under Windows spawn. Keep this module's
import light: `dfxm_geo.pipeline` (the heavy import) is loaded lazily
inside `run_one` so its cost is measured and attributed, exactly like the
subprocess path's DFXM_TIMING contract.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

_import_timed = False


def _timed_pipeline_import() -> float:
    """Import dfxm_geo.pipeline; return the seconds it took.

    Returns the measured time on this process's first call and exactly 0.0
    afterwards — so per-config DFXM_TIMING rows show the worker's one-time
    import cost on its first config and 0.0 on the warm ones.
    """
    global _import_timed
    if _import_timed:
        return 0.0
    t0 = time.perf_counter()
    import dfxm_geo.pipeline  # noqa: F401  (heavy: numpy/numba/h5py chain)

    _import_timed = True
    return time.perf_counter() - t0


def _resolve_cli(mode: str) -> tuple[Callable[[list[str]], int], list[str]]:
    """Map fanout mode -> (CLI callable, extra argv). Mirrors fanout.build_cmd."""
    from dfxm_geo.pipeline import cli_main, cli_main_identify

    if mode == "forward":
        return cli_main, ["--no-postprocess"]
    if mode == "identify":
        return cli_main_identify, []
    raise ValueError(f"mode must be 'forward' or 'identify', got {mode!r}")


def run_one(mode: str, config: str, output_dir: str, log_path: str) -> dict[str, Any]:
    """Run one config in-process; the pool-mode analogue of one subprocess.

    Contract (mirrors `scripts/fanout.py` `_CHILD_SNIPPET` + `_default_runner`):
    - stdout AND stderr (tqdm writes to stderr) go to `log_path`;
    - a `DFXM_TIMING {json}` line with import_s/run_s is printed into the log
      so `fanout.parse_timing_log` works unchanged in pool mode;
    - every exception is contained: returns `{"returncode": -1}` with the
      traceback in the log (batch resilience — one bad config never kills
      the sweep). SystemExit codes (argparse errors) are propagated as rc.

    Args are strings (not Path) so the cross-process pickle stays trivial.
    Returns `{"returncode": int, "wall_s": float}`.
    """
    t_wall = time.perf_counter()
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh, redirect_stdout(fh), redirect_stderr(fh):
        try:
            import_s = _timed_pipeline_import()
            cli, extra = _resolve_cli(mode)
            argv = ["--config", config, "--output", output_dir] + extra
            t0 = time.perf_counter()
            rc = int(cli(argv))
            run_s = time.perf_counter() - t0
            print(
                "DFXM_TIMING "
                + json.dumps({"import_s": round(import_s, 3), "run_s": round(run_s, 3)})
            )
        except SystemExit as exc:  # argparse exits with code 2 on bad args
            code = exc.code
            rc = code if isinstance(code, int) else (0 if code is None else 1)
            if rc != 0:
                traceback.print_exc()
        except Exception:
            traceback.print_exc()
            rc = -1
    return {"returncode": rc, "wall_s": time.perf_counter() - t_wall}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fanout_worker.py -v` → all PASS.
Run: `python -m mypy src/dfxm_geo/` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/fanout_worker.py tests/test_fanout_worker.py
git commit -m "feat(w1): dfxm_geo.fanout_worker.run_one — in-process pool worker"
```

---

## Task 9: Pool path in `scripts/fanout.py` (`run_pool` + `--isolate`)

**Files:**
- Modify: `scripts/fanout.py`
- Test: `tests/test_fanout.py` (extend)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fanout.py`)

```python
# ---------------------------------------------------------------------------
# Pool mode (v2.6.0 W1). Tests inject worker_fn + executor_factory so no real
# process pool is spawned: a ThreadPoolExecutor exercises the same
# submit/as_completed/recovery code paths and imposes no picklability
# constraint on the fakes.
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool


def _ok_worker(mode, config, output_dir, log_path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(
        'DFXM_TIMING {"import_s": 1.5, "run_s": 2.5}\nWrote 4 images to x\n',
        encoding="utf-8",
    )
    return {"returncode": 0, "wall_s": 0.01}


def test_run_pool_basic(tmp_path: Path) -> None:
    configs = [tmp_path / f"c{i}.toml" for i in range(3)]
    for c in configs:
        c.write_text("")
    results = fanout.run_pool(
        configs,
        tmp_path / "out",
        n_workers=2,
        mode="identify",
        worker_fn=_ok_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    assert len(results) == 3
    assert all(r.returncode == 0 for r in results)
    assert {r.config for r in results} == set(configs)
    # Results come back in input order regardless of completion order.
    assert [r.config for r in results] == configs
    timing = fanout.parse_timing_log(results[0].log_path)
    assert timing["import_s"] == 1.5 and timing["images"] == 4


def test_run_pool_worker_dict_failure_is_contained(tmp_path: Path) -> None:
    c = tmp_path / "bad.toml"
    c.write_text("")

    def failing_worker(mode, config, output_dir, log_path):
        Path(log_path).write_text("boom\n", encoding="utf-8")
        return {"returncode": -1, "wall_s": 0.0}

    results = fanout.run_pool(
        [c],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=failing_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    assert results[0].returncode == -1


def test_run_pool_raising_worker_is_contained(tmp_path: Path) -> None:
    """A worker_fn that raises (not via run_one's guard) is rc=-1 + log."""
    good, bad = tmp_path / "good.toml", tmp_path / "boom.toml"
    good.write_text("")
    bad.write_text("")

    def worker(mode, config, output_dir, log_path):
        if "boom" in config:
            raise RuntimeError("worker exploded")
        return _ok_worker(mode, config, output_dir, log_path)

    results = fanout.run_pool(
        [good, bad],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    by_name = {r.config.name: r for r in results}
    assert by_name["good.toml"].returncode == 0
    assert by_name["boom.toml"].returncode == -1
    assert "worker exploded" in by_name["boom.toml"].log_path.read_text(encoding="utf-8")


def test_run_pool_recovers_from_broken_pool(tmp_path: Path) -> None:
    """First executor 'hard-crashes' on one config; run_pool rebuilds the
    executor and retries; second attempt succeeds. The other config's result
    survives. A config that breaks the pool twice is marked rc=-2."""
    ok = tmp_path / "ok.toml"
    flaky = tmp_path / "flaky.toml"
    fatal = tmp_path / "fatal.toml"
    for c in (ok, flaky, fatal):
        c.write_text("")
    attempts: dict[str, int] = {}

    def worker(mode, config, output_dir, log_path):
        name = Path(config).name
        attempts[name] = attempts.get(name, 0) + 1
        if name == "flaky.toml" and attempts[name] == 1:
            raise BrokenProcessPool("simulated worker death")
        if name == "fatal.toml":
            raise BrokenProcessPool("simulated worker death")
        return _ok_worker(mode, config, output_dir, log_path)

    results = fanout.run_pool(
        [ok, flaky, fatal],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    by_name = {r.config.name: r for r in results}
    assert by_name["ok.toml"].returncode == 0
    assert by_name["flaky.toml"].returncode == 0  # retried, then fine
    assert by_name["fatal.toml"].returncode == -2  # gave up after 2 attempts
    assert attempts["fatal.toml"] == 2


def test_run_manifest_dispatches_pool_by_default(tmp_path: Path, monkeypatch) -> None:
    c = tmp_path / "c.toml"
    c.write_text("")
    called = {}

    def fake_run_pool(configs, output_root, **kwargs):
        called["pool"] = True
        return []

    monkeypatch.setattr(fanout, "run_pool", fake_run_pool)
    fanout.run_manifest([c], tmp_path / "out", n_workers=1, mode="identify")
    assert called.get("pool") is True


def test_run_manifest_isolate_uses_subprocess_path(tmp_path: Path, monkeypatch) -> None:
    c = tmp_path / "c.toml"
    c.write_text("")

    def fake_run_pool(*a, **k):  # must NOT be called
        raise AssertionError("pool path used despite isolate=True")

    monkeypatch.setattr(fanout, "run_pool", fake_run_pool)

    def fake_runner(config, output_dir, env, log_path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest(
        [c], tmp_path / "out", n_workers=1, runner=fake_runner, isolate=True
    )
    assert results[0].returncode == 0


def test_run_manifest_runner_seam_implies_isolate(tmp_path: Path, monkeypatch) -> None:
    """Back-compat: every existing test that injects `runner` keeps testing
    the subprocess orchestration without passing isolate."""
    c = tmp_path / "c.toml"
    c.write_text("")
    monkeypatch.setattr(
        fanout, "run_pool", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    def fake_runner(config, output_dir, env, log_path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest([c], tmp_path / "out", n_workers=1, runner=fake_runner)
    assert results[0].returncode == 0


def test_run_pool_sizes_executor_to_n_workers(tmp_path: Path) -> None:
    """Spec §7.5 concurrency cap: the executor is created with n_workers —
    the pool's concurrency limit is delegated to the executor itself."""
    configs = [tmp_path / f"c{i}.toml" for i in range(4)]
    for c in configs:
        c.write_text("")
    sizes: list[int] = []

    def recording_factory(n: int):
        sizes.append(n)
        return ThreadPoolExecutor(max_workers=n)

    fanout.run_pool(
        configs,
        tmp_path / "out",
        n_workers=3,
        mode="identify",
        worker_fn=_ok_worker,
        executor_factory=recording_factory,
    )
    assert sizes == [3]


def test_pool_env_pinned_during_run_and_restored_after(tmp_path: Path) -> None:
    import os

    c = tmp_path / "c.toml"
    c.write_text("")
    seen_env: dict[str, str | None] = {}

    def env_spy_worker(mode, config, output_dir, log_path):
        seen_env["DFXM_MAX_WORKERS"] = os.environ.get("DFXM_MAX_WORKERS")
        seen_env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
        return _ok_worker(mode, config, output_dir, log_path)

    before = os.environ.get("DFXM_MAX_WORKERS")
    fanout.run_pool(
        [c],
        tmp_path / "out",
        n_workers=1,
        threads_per_worker=3,
        mode="identify",
        worker_fn=env_spy_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    # ThreadPoolExecutor shares the parent env, so the spy observes exactly
    # what spawned children would inherit at submit time.
    assert seen_env["DFXM_MAX_WORKERS"] == "3"
    assert seen_env["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("DFXM_MAX_WORKERS") == before  # restored
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_fanout.py -v -k "pool or isolate"`
Expected: FAIL — `AttributeError: module 'fanout' has no attribute 'run_pool'`

- [ ] **Step 3: Implement the pool path**

In `scripts/fanout.py`:

(a) extend imports:

```python
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor
```

(b) env-pinning context manager (after `worker_env`):

```python
@contextlib.contextmanager
def _pinned_environ(threads_per_worker: int):
    """Pin worker env keys in os.environ for the pool's lifetime, restore after.

    Pool workers inherit the env at process creation (fork on Linux/LSF,
    spawn on Windows), so the keys must be set in the PARENT before the
    executor spawns its first worker — and restored afterwards so the
    calling process (tests, notebooks) is not permanently mutated.
    """
    target = worker_env(threads_per_worker)
    keys = ["DFXM_MAX_WORKERS", *_PINNED_SINGLE]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = target[k]
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
```

(c) the pool runner:

```python
def run_pool(
    configs: list[Path],
    output_root: Path,
    *,
    n_workers: int = 8,
    threads_per_worker: int = 16,
    mode: str = "forward",
    worker_fn: Callable[[str, str, str, str], dict] | None = None,
    executor_factory: Callable[[int], "Executor"] | None = None,
    max_attempts: int = 2,
) -> list[ConfigResult]:
    """Run `configs` on a persistent worker pool (v2.6.0 W1, the default mode).

    Workers import dfxm_geo once, JIT once, and retain the kernel LUT
    (`pipeline._KERNEL_CTX_CACHE`) across configs — amortizing the fixed
    ~47%-of-wall per-subprocess cost the old isolate mode pays every config.

    Failure containment:
    - worker_fn exceptions and rc!=0 dicts -> per-config failed result;
    - a hard worker death (`BrokenProcessPool`) aborts the current executor:
      finished results are kept, undone configs are resubmitted on a fresh
      executor; a config still broken after `max_attempts` is rc=-2.

    `worker_fn` / `executor_factory` are test seams; production uses
    `dfxm_geo.fanout_worker.run_one` on a `ProcessPoolExecutor`.
    """
    if worker_fn is None:
        from dfxm_geo import fanout_worker

        worker_fn = fanout_worker.run_one
    if executor_factory is None:
        executor_factory = lambda n: ProcessPoolExecutor(max_workers=n)  # noqa: E731

    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[Path, ConfigResult] = {}
    attempts: dict[Path, int] = dict.fromkeys(configs, 0)
    pending = list(configs)

    def _record(config: Path, rc: int, wall_s: float) -> None:
        results[config] = ConfigResult(
            config,
            output_root / config.stem,
            rc,
            wall_s,
            output_root / f"{config.stem}.log",
        )

    with _pinned_environ(threads_per_worker):
        while pending:
            batch, pending = pending, []
            broken: list[Path] = []
            with executor_factory(n_workers) as ex:
                futs: dict[Future, Path] = {}
                for c in batch:
                    attempts[c] += 1
                    out_dir = output_root / c.stem
                    log_path = output_root / f"{c.stem}.log"
                    futs[ex.submit(worker_fn, mode, str(c), str(out_dir), str(log_path))] = c
                not_done = set(futs)
                while not_done:
                    done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
                    for fut in done:
                        c = futs[fut]
                        try:
                            payload = fut.result()
                            _record(
                                c,
                                int(payload["returncode"]),
                                float(payload.get("wall_s", 0.0)),
                            )
                        except BrokenProcessPool:
                            broken.append(c)
                        except Exception:
                            # worker_fn raised (run_one's guard normally
                            # prevents this) — failed config, not a dead pool.
                            log_path = output_root / f"{c.stem}.log"
                            log_path.write_text(
                                f"fanout: pool worker raised for {c}\n\n"
                                f"{traceback.format_exc()}",
                                encoding="utf-8",
                            )
                            _record(c, -1, 0.0)
                # A broken executor poisons every future submitted to it:
                # anything not individually recorded above is broken too.
            for c in broken:
                if attempts[c] < max_attempts:
                    pending.append(c)
                else:
                    log_path = output_root / f"{c.stem}.log"
                    log_path.write_text(
                        f"fanout: worker pool died twice running {c}; giving up "
                        f"(run with --isolate to debug this config)\n",
                        encoding="utf-8",
                    )
                    _record(c, -2, 0.0)

    return [results[c] for c in configs]
```

Implementation note: with `ProcessPoolExecutor`, ONE dead worker breaks the
whole executor — every outstanding `fut.result()` raises
`BrokenProcessPool`, so all of the batch's unfinished configs land in
`broken` and get one retry as a group. That is intended: only the config
that kills the pool twice is condemned.

(d) `run_manifest` dispatch — add the `isolate` parameter and route:

```python
def run_manifest(
    configs: list[Path],
    output_root: Path,
    *,
    n_workers: int = 8,
    threads_per_worker: int = 16,
    runner: Runner | None = None,
    base_env: Mapping[str, str] | None = None,
    mode: str = "forward",
    isolate: bool = False,
) -> list[ConfigResult]:
```

and as the FIRST statement of the body:

```python
    # Pool is the default (v2.6.0). The subprocess-per-config path remains
    # behind --isolate; an injected `runner` implies it (the runner seam IS
    # the subprocess abstraction, and all pre-v2.6.0 tests use it).
    if runner is None and not isolate:
        return run_pool(
            configs,
            output_root,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            mode=mode,
        )
```

(the rest of the existing body is unchanged).

(e) CLI — in `main()` add:

```python
    ap.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Run each config in a fresh subprocess (pre-v2.6.0 behavior) "
            "instead of the persistent worker pool. Slower (re-pays import/"
            "JIT/kernel-load per config) but gives hard crash isolation — "
            "use it to debug a config that kills pool workers."
        ),
    )
```

pass `isolate=args.isolate` in the `run_manifest(...)` call, and include
the mode in the startup print:
`f"... [{args.mode}{', isolate' if args.isolate else ', pool'}] -> {args.output}"`.

Also update the module docstring usage section to mention pool default +
`--isolate`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fanout.py tests/test_fanout_worker.py -v`
Expected: ALL pass — new pool tests AND every pre-existing fanout test
(they inject `runner` → isolate path → unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add scripts/fanout.py tests/test_fanout.py
git commit -m "feat(w1): persistent worker pool is fanout's default; --isolate keeps subprocess path"
```

---

## Task 10: Timing-json continuity in pool mode

**Files:**
- Modify: `scripts/fanout.py` (`write_timing_json`, `main`)
- Test: `tests/test_fanout.py` (extend)

- [ ] **Step 1: Write the failing test** (append to `tests/test_fanout.py`)

```python
def test_write_timing_json_records_isolate_flag(tmp_path: Path) -> None:
    import json as _json

    c = tmp_path / "c.toml"
    c.write_text("")
    results = fanout.run_pool(
        [c],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=_ok_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    out = tmp_path / "timing.json"
    fanout.write_timing_json(
        out,
        results,
        total_wall_s=1.0,
        n_workers=1,
        threads_per_worker=1,
        mode="identify",
        isolate=False,
    )
    payload = _json.loads(out.read_text(encoding="utf-8"))
    assert payload["sweep"]["isolate"] is False
    # Pool-mode logs carry the same DFXM_TIMING contract -> rows unchanged.
    assert payload["configs"][0]["import_s"] == 1.5
    assert payload["configs"][0]["run_s"] == 2.5
    assert payload["configs"][0]["images"] == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_fanout.py::test_write_timing_json_records_isolate_flag -v`
Expected: FAIL — `write_timing_json() got an unexpected keyword argument 'isolate'`

- [ ] **Step 3: Implement**

`write_timing_json` signature gains `isolate: bool = False` (keyword-only,
after `mode`); add `"isolate": isolate,` to the `sweep` dict right after
`"mode": mode,`. Docstring: note that in pool mode `import_s` is per-worker
amortized (real on each worker's first config, 0.0 on warm ones) and
`wall_s` excludes queue wait. In `main()`, pass `isolate=args.isolate`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fanout.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/fanout.py tests/test_fanout.py
git commit -m "feat(w1): timing.json records isolate/pool mode; schema otherwise unchanged"
```

---

## Task 11: Pool-vs-isolate bit-identity gate (the headline determinism test)

**Files:**
- Test: `tests/test_fanout.py` (extend)

- [ ] **Step 1: Write the test** (append to `tests/test_fanout.py`)

```python
@pytest.mark.slow
def test_pool_and_isolate_produce_bit_identical_identify_output(tmp_path: Path) -> None:
    """Spec §7.1 gate: same seeded identify-multi configs through the pool
    path and the subprocess (--isolate) path must yield byte-identical
    detector datasets + positioners. Compares datasets, not raw files —
    master files embed timestamps."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping integration run.")

    import h5py
    import numpy as np

    cfg_text = """\
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[noise]
poisson_noise = true
rng_seed = {seed}
intensity_scale = 7.0

[multi]
n_samples = 1
pos_std_um = 5.0
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""
    manifest_dir = tmp_path / "configs"
    manifest_dir.mkdir()
    for seed in (11, 12):
        (manifest_dir / f"seed{seed}.toml").write_text(
            cfg_text.format(seed=seed), encoding="utf-8"
        )
    configs = fanout.discover_configs(manifest_dir)

    out_iso = tmp_path / "out_isolate"
    out_pool = tmp_path / "out_pool"
    res_iso = fanout.run_manifest(
        configs, out_iso, n_workers=1, threads_per_worker=2, mode="identify", isolate=True
    )
    res_pool = fanout.run_manifest(
        configs, out_pool, n_workers=1, threads_per_worker=2, mode="identify"
    )
    assert all(r.returncode == 0 for r in res_iso), [
        r.log_path.read_text() for r in res_iso if r.returncode
    ]
    assert all(r.returncode == 0 for r in res_pool), [
        r.log_path.read_text() for r in res_pool if r.returncode
    ]

    def _datasets(root: Path) -> dict[str, np.ndarray]:
        """name -> array for every dataset in every per-scan HDF5 under root."""
        out: dict[str, np.ndarray] = {}
        for h5path in sorted(root.rglob("*.h5")):
            if h5path.name == "dfxm_identify.h5":
                continue  # master: external links + timestamped metadata
            with h5py.File(h5path, "r") as f:

                def _grab(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        out[f"{h5path.relative_to(root)}::{name}"] = obj[()]

                f.visititems(_grab)
        return out

    for stem in ("seed11", "seed12"):
        d_iso = _datasets(out_iso / stem)
        d_pool = _datasets(out_pool / stem)
        assert d_iso.keys() == d_pool.keys()
        assert d_iso, f"no datasets found for {stem}"
        for key in d_iso:
            a, b_ = d_iso[key], d_pool[key]
            if isinstance(a, np.ndarray) and a.dtype.kind == "f":
                np.testing.assert_array_equal(a, b_, err_msg=key)  # BIT-identical
            else:
                assert np.array_equal(a, b_), key
```

NOTE: per-scan files may store string/provenance datasets that embed the
config PATH (different tmp dirs) — if the test fails only on such a
dataset, exclude provenance-string keys by name (e.g. skip keys containing
`"config"` or `"cli"`) and assert the rest. Detector images and positioners
must always be compared.

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_fanout.py::test_pool_and_isolate_produce_bit_identical_identify_output -v -m slow`
Expected: PASS (≈1-3 min: 4 identify runs at px510 × 2 frames, the pool run
pays one JIT). It runs the REAL `ProcessPoolExecutor` + `fanout_worker.run_one`
— this is also the Windows-spawn smoke test for the worker.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fanout.py
git commit -m "test(w1): pool-vs-isolate bit-identity gate on seeded identify-multi configs"
```

---

## Task 12: e2e pool smoke for forward mode + bsub note

**Files:**
- Modify: `tests/test_fanout.py` (the existing e2e at line 220 keeps covering isolate; add a pool twin)
- Modify: `lsf/fanout.bsub`

- [ ] **Step 1: Add the pool e2e twin** (append to `tests/test_fanout.py`)

```python
def test_fanout_pool_end_to_end_forward(tmp_path: Path) -> None:
    """Pool-mode twin of test_fanout_end_to_end_runs_two_configs: real
    ProcessPoolExecutor + run_one, two tiny forward configs, n_workers=1."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping integration run.")

    import h5py
    import numpy as np

    cfg_text = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n"
        "[scan.phi]\nrange = 0.001\nsteps = 2\n\n"
        "[io]\ninclude_perfect_crystal = false\n\n"
        "[postprocess]\nenabled = false\n"
    )
    manifest_dir = tmp_path / "configs"
    manifest_dir.mkdir()
    for name in ("alpha.toml", "beta.toml"):
        (manifest_dir / name).write_text(cfg_text)

    out_root = tmp_path / "out"
    results = fanout.run_manifest(
        fanout.discover_configs(manifest_dir),
        out_root,
        n_workers=1,
        threads_per_worker=2,
        mode="forward",
    )
    assert all(r.returncode == 0 for r in results), [r.log_path.read_text() for r in results]
    for stem in ("alpha", "beta"):
        det = out_root / stem / "scan0001" / "dfxm_sim_detector_0000.h5"
        assert det.is_file()
        with h5py.File(det, "r") as f:
            img = f["/entry_0000/dfxm_sim_detector/image"]
            assert img.dtype == np.float32
            assert img.shape[0] == 2
        timing = fanout.parse_timing_log(out_root / f"{stem}.log")
        assert timing.get("run_s", 0) > 0
    # One worker, two configs: exactly one config paid the import.
    imports = [
        fanout.parse_timing_log(out_root / f"{s}.log").get("import_s")
        for s in ("alpha", "beta")
    ]
    assert sorted(imports) == sorted([imports[0], imports[1]])  # both present
    assert min(imports) == 0.0  # the warm config reports 0.0
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_fanout.py::test_fanout_pool_end_to_end_forward -v`
Expected: PASS.

- [ ] **Step 3: Document `--isolate` in the LSF template**

In `lsf/fanout.bsub`, add next to the `python scripts/fanout.py` invocation
(do NOT change the command itself — pool is the new default and the
template stays valid):

```bash
# v2.6.0: fanout runs a persistent worker pool by default (import/JIT/kernel
# loaded once per worker). Append --isolate to fall back to the pre-v2.6.0
# subprocess-per-config mode when debugging a config that kills workers.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_fanout.py lsf/fanout.bsub
git commit -m "test(w1): pool-mode forward e2e + lsf template --isolate note"
```

---

## Task 13: Full gates (spec §7.4)

- [ ] **Step 1: Full suite, kernel present**

Run: `python -m pytest -q -m "slow or not slow"`
Expected: green (same skip set as the v2.5.0 baseline + the new tests
passing; compare the failure SET to the baseline, not the green count —
see [[preexisting-test-failures-2026-05-28]] for the principle).

- [ ] **Step 2: Kernel-less run (the v2.5.0 CI lesson)**

```powershell
Rename-Item "C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\reciprocal_space\pkl_files" pkl_files_hidden
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
Rename-Item "C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\reciprocal_space\pkl_files_hidden" pkl_files
```
Expected: green (kernel-dependent tests SKIP — notably the new
`test_identify_dedup.py`, the bit-identity gate, and both e2e tests; no
ERRORS, no import-time kernel loads).

- [ ] **Step 3: Static gates**

Run: `python -m mypy src/dfxm_geo/` → `Success: no issues found`
Run: `python -m ruff check . && python -m ruff format --check .` → clean
(pre-commit also enforces these at each commit).

- [ ] **Step 4: Fix anything found, re-run, commit fixes**

```bash
git add -A
git commit -m "chore: gate fixes (full suite + kernel-less + mypy + ruff)"
```
(skip the commit if nothing needed fixing)

---

## Task 14: Benchmark + `docs/cluster-profiling.md`

- [ ] **Step 1: Quick 16-config comparison (same shape as the baseline row)**

```powershell
$py = "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe"
& $py scripts/gen_identify_sweep_configs.py --n-configs 16 --n-samples 2 --out-dir tmp/sweep16
& $py scripts/fanout.py --manifest tmp/sweep16 --output tmp/bench16_pool --mode identify --n-workers 8 --threads-per-worker 1 --timing-json tmp/bench16_pool/timing.json
& $py scripts/fanout.py --manifest tmp/sweep16 --output tmp/bench16_iso --mode identify --n-workers 8 --threads-per-worker 1 --isolate --timing-json tmp/bench16_iso/timing.json
```
Baseline to beat: 703 configs/hour / 4.3 images/s (isolate path,
pre-W2/W3). Read `configs_per_hour` from both timing.json files. The
`--isolate` run shows W2+W3's contribution alone; the pool run shows the
full stack.

- [ ] **Step 2: The DoD measurement — 500-config sweep**

```powershell
& $py scripts/gen_identify_sweep_configs.py --n-configs 500 --n-samples 2 --render-per-dislocation --out-dir tmp/sweep500
& $py scripts/fanout.py --manifest tmp/sweep500 --output tmp/bench500_pool --mode identify --n-workers 8 --threads-per-worker 1 --timing-json tmp/bench500_pool/timing.json
```
(run in background; expect well under an hour post-optimization).

**How to score the ≥5× honestly.** The recorded 2026-06-10 baseline
(703 configs/hour) was the SAME 16-config shape but WITHOUT
`--render-per-dislocation`, so it cannot be compared to a flag-on run.
Report two numbers:
(a) **the DoD comparison** — Step 1's flag-off 16-config pool run vs the
recorded 703 configs/hour baseline (identical shape, apples-to-apples);
(b) **the production-shape datapoint** — the 500-config flag-on pool run's
absolute configs/hour and images/s, with its own flag-on `--isolate`
16-config control from Step-1-style commands so the flag-on speedup is
also measured against THIS code's isolate path.
If (a) < 5×, stop and escalate to Sina with the numbers before reaching for
further optimization — the roadmap P2 items (Z_shift/Ud precompute) are the
next lever and need a decision.

- [ ] **Step 3: Record in `docs/cluster-profiling.md`**

Append a `## Phase 2b post-optimization (v2.6.0, 2026-06-XX)` section after
the baseline section: the 16-config before/after table (configs/hour,
images/s, mean import_s/run_s), the 500-config row, the speedup factors
attributable to pool vs W2+W3 (from the isolate-vs-pool comparison), and a
sentence that the LSF node row follows the same procedure via
`lsf/fanout.bsub` (still pending from the cluster side at the time of
writing). Keep the existing "LSF node baseline: TODO" block updated if the
parallel session has landed the cluster row by now.

- [ ] **Step 4: Clean up benchmark intermediates**

Delete `tmp/bench*`, `tmp/sweep*` output HDF5 trees (regenerable; CLAUDE.md
wrap-up rule for >10 MB intermediates). Keep the two timing.json files —
copy them to `docs/data/` if referenced, else inline their numbers in the
doc and delete.

- [ ] **Step 5: Commit**

```bash
git add docs/cluster-profiling.md
git commit -m "docs: Phase 2b post-optimization benchmark (pool + dedup + fused kernel)"
```

---

## Task 15: Release prep (v2.6.0) — STOP before push

- [ ] **Step 1: Version bump**

- `pyproject.toml`: `version = "2.5.0"` → `"2.6.0"`.
- `git mv tests/test_version_is_2_5_0.py tests/test_version_is_2_6_0.py`
  and update the asserted string inside to `"2.6.0"` (keep the file's
  existing structure).
- Reinstall so importlib metadata matches: `python -m pip install -e ".[dev]" --no-deps`
- Run: `python -m pytest tests/test_version_is_2_6_0.py -v` → PASS.

- [ ] **Step 2: Release notes**

Follow the repo's existing release-notes convention (check
`docs/release-notes/` or where v2.5.0's notes live — `git show v2.5.0
--stat` reveals the file). Content: the three workstreams, the
`--isolate` flag, the engine flip + ~1e-15 output delta note + any golden
regeneration, the timing.json `isolate` field, benchmark numbers.

- [ ] **Step 3: Commit, then STOP**

```bash
git add -A
git commit -m "Release prep v2.6.0: pool + Fd dedup + fused Hg kernel"
```

**Do NOT merge to main, tag, or push.** Per CLAUDE.md: `git fetch` + check
`origin/main` first, merge `--no-ff`, tag v2.6.0 — all of that happens only
after presenting the gate results + benchmark numbers to Sina and getting
explicit approval. CLAUDE.md + auto-memory updates happen at ship time.

---

## Self-review checklist (done at plan-writing time)

- Spec coverage: §3 pool → Tasks 8-12; §4 dedup → Tasks 1-4; §5 fusion →
  Tasks 5-7; §6 timing → Tasks 8 (DFXM_TIMING line), 10; §7 gates → Tasks
  11, 13 (+ per-task suites); §8 DoD → Task 14 (LSF row remains external);
  §10 release → Task 15. Deferred §9 items: no tasks, by design.
- Known judgment calls encoded: runner-seam implies isolate (back-compat);
  BrokenProcessPool batch-retry semantics; engine flip isolated in Task 7;
  zscan secondary position note (pos_std=0 → bit-identical).
- The W3 combined-accumulation parity caution (Task 6 Step 3) is the one
  place implementation may legitimately deviate — the parity tests are the
  arbiter.
