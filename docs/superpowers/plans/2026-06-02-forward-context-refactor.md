# ForwardContext refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `dfxm_geo.direct_space.forward_model`'s module-level mutable globals with an explicit, immutable `ForwardContext` threaded through the forward/postprocess/identify paths, eliminating the persistent-worker-pool cross-config leakage hazard and the simplified-reflection `theta_0` staleness bug.

**Architecture:** Three frozen dataclasses split state by lifetime — `InstrumentContext` (per-process), `GeometryContext` (per-reflection Bragg geometry), `ResolutionContext` (per-reflection kernel/analytic backend) — composed into `ForwardContext`. The state migrates in 5 bit-exact slices behind a PEP-562 `__getattr__` shim so every consumer converts independently. The numba kernel `_mc_lut_forward` already takes all args explicitly; its signature never changes, so there is no recompile and no perf change. Strain (`Hg`/`q_hkl`) stays threaded explicitly (already is) — the refactor just stops the module-global *writes/reads* of it.

**Tech Stack:** Python 3.11+, NumPy, numba (`@njit`), dataclasses, h5py, pytest, mypy. Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Source spec:** `docs/superpowers/specs/2026-06-02-forward-context-refactor.md`

---

## Conventions for every task

- **Venv pytest:** `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q <args>`. Bash's bare `python` is Python 2.7 — never use it.
- **mypy gate:** `& "...\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/` must report 0 errors after every task that touches `src/`.
- **Kernel-present suite:** Several forward/identify tests SKIP unless a bootstrapped kernel sits on disk. Before the slice gates, bootstrap a seed-0 canonical kernel (see "Canonical kernel" below) so those tests actually run — CI alone is blind to kernel paths (this is the bug that bit v2.3.0). Run with the kernel present at every slice gate.
- **Bit-exactness is the contract.** The float ops in `precompute_forward_static`/`forward_from_static`/`_mc_lut_forward` are unchanged — only where their *inputs* come from changes. Every numerical verification uses `np.array_equal` / elementwise `==`, never `allclose`. **Never regenerate or touch** any `*bit_equiv*`, `*snapshot*`, `*pickle_era*`, or `tests/data/golden/` artifact. If a slice would alter one, the slice is wrong — revert and diagnose.
- **Commits:** one per task (or per step where noted). Co-author trailer per repo convention. Do not push; the user confirms pushes.

### Canonical kernel (needed for kernel-present gates)

Bootstrap a simplified Al-111 17 keV seed-0 kernel from a config with a `[reciprocal]` block + a mount `[crystal]` (lattice/a/mount_x/y/z) + `[geometry] mode="simplified"`:

```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.reciprocal_space.kernel --seed 0 --config <that-config.toml>
```
(`dfxm-bootstrap --config configs/default.toml` also works as of v2.3.1.) The kernel `.npz` lands under `reciprocal_space/pkl_files/`. This is a regenerable scratch artifact — delete it at wrap-up (it is >10 MB).

---

## File structure

| File | Responsibility | Slices |
|---|---|---|
| `src/dfxm_geo/direct_space/forward_model.py` | Owns the globals today; gains the dataclasses, `build_*_context` constructors, `__getattr__` shim, `ctx`-threaded function signatures; loses the globals + CM in Slice 5. | all |
| `src/dfxm_geo/pipeline.py` | `run_simulation`/`run_postprocess`/`run_identification` build `ctx`; the 3 identify generators read `ctx.geometry.*`; stop `fm.Hg`/`fm.q_hkl` writes; `_forward_state_guard`. | 1, 4, 5 |
| `src/dfxm_geo/io/hdf5.py` | `_compute_frame` + writers carry `ctx`; `run_postprocess` Hg-from-file read. | 1, 3, 4 |
| `src/dfxm_geo/io/migrate.py` | Source provenance from `ctx`/HDF5 once the shim is gone. | 5 |
| `tests/test_forward_context.py` (new) | Parity oracle (context vs legacy globals), dual-path bit-identity, no-recompile assertion. | 2, 3 |
| `tests/test_forward_state_guard.py` (new) | Cross-config leakage / generator-write-race regression tests. | 1 |

---

## SLICE 1 — `#10` hardening (minimal, bit-exact, shippable on its own)

Removes the persistent-worker-pool hazard *without* introducing `ForwardContext`. No hot-path edit, no numba recompile. This slice is independently releasable (the spec calls it patch/minor).

### Task 1: Stop the generator-body global write (`fm.q_hkl`)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:1674-1675` (`_iter_identification_multi`)
- Test: `tests/test_forward_state_guard.py` (create)

- [ ] **Step 1: Write the failing test**

`_iter_identification_multi` currently does `q_hkl = np.asarray(fm.q_hkl, dtype=float); fm.q_hkl = q_hkl` — the write is a no-op leftover (it assigns back the same float64 object) that mutates a process global from inside a generator body. **A purely behavioral test cannot distinguish pre/post deletion** (the written value equals the read value), and driving the generator to that line requires a full kernel + population stand-up. So the honest TDD-shaped guard is a static source assertion that the reassignment is gone — red before the edit, green after, and a permanent guard against reintroduction. The existing multi-identify integration tests provide the behavioral regression safety net.

```python
# tests/test_forward_state_guard.py
import inspect
import numpy as np
import pytest
from dfxm_geo.direct_space import forward_model as fm
from dfxm_geo import pipeline


def test_multi_identify_generator_does_not_assign_fm_q_hkl():
    """_iter_identification_multi must not assign the fm.q_hkl global.

    Regression for #10: a generator writing a process global is a data race
    on the planned persistent-worker pool. The original write was a no-op
    (assigned back the value just read), so this guards the source directly.
    """
    src = inspect.getsource(pipeline._iter_identification_multi)
    # No assignment to fm.q_hkl in any spacing (reads `... = np.asarray(fm.q_hkl...)`
    # are fine; only `fm.q_hkl = ...` is the forbidden write).
    compact = src.replace(" ", "")
    assert "fm.q_hkl=" not in compact, "generator still assigns fm.q_hkl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_forward_state_guard.py -v`
Expected: FAIL — `assert "fm.q_hkl=" not in compact` fails because the generator still contains the assignment at line 1675.

- [ ] **Step 3: Delete the write**

In `src/dfxm_geo/pipeline.py`, change lines 1674-1675 from:

```python
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    fm.q_hkl = q_hkl
```
to:
```python
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_forward_state_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add tests/test_forward_state_guard.py src/dfxm_geo/pipeline.py
git commit -m "fix(#10): stop _iter_identification_multi writing fm.q_hkl global"
```

### Task 2: Pass `Hg`/`q_hkl` explicitly into `run_postprocess`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` `run_postprocess` (signature + lines 1125-1127)
- Test: `tests/test_pipeline.py` (add a `--postprocess-only`-style test, or new `tests/test_postprocess_strain_source.py`)

`run_postprocess` reads `fm.Hg` and raises `RuntimeError("fm.Hg is not set. Call run_simulation() first.")` (lines 1125-1127). The writer already persists `Hg` at `/1.1/dfxm_geo/Hg` when `write_strain_provenance=True` (hdf5.py:851-852). Recover strain from the file instead of a stale global.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_postprocess_strain_source.py
import numpy as np
from dfxm_geo.direct_space import forward_model as fm


def test_run_postprocess_reads_Hg_from_file_when_global_absent(tmp_path, monkeypatch):
    """run_postprocess must recover Hg from /1.1/dfxm_geo/Hg, not a stale global."""
    from dfxm_geo import pipeline

    # 1) Run a tiny forward sim that persists Hg provenance.
    config = pipeline._minimal_forward_config_for_test(tmp_path)  # see note below
    pipeline.run_simulation(config, tmp_path)
    # 2) Clear the global to simulate a fresh process / --postprocess-only.
    monkeypatch.setattr(fm, "Hg", None, raising=False)
    # 3) Postprocess must NOT raise and must produce qi_field.
    pipeline.run_postprocess(tmp_path, config)
    import h5py
    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
```

For `_minimal_forward_config_for_test`, reuse the smallest existing forward fixture (grep `run_simulation` in `tests/test_hdf5_*`); ensure `io.write_strain_provenance` is True (the default) and the scan is centered/single so postprocess applies. Keep it 5×5, Npixels 64.

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_postprocess_strain_source.py -v`
Expected: FAIL — `RuntimeError: fm.Hg is not set` after the global is cleared.

- [ ] **Step 3: Add optional `Hg`/`q_hkl` params + file fallback**

Change `run_postprocess`'s signature to accept optional strain, and replace lines 1125-1127:

```python
def run_postprocess(
    output_dir: Path,
    config: "SimulationConfig",
    *,
    Hg: np.ndarray | None = None,
    q_hkl: np.ndarray | None = None,
) -> None:
    ...
```

Replace the `fm.Hg` read (lines 1125-1127) with a resolution helper:

```python
    Hg_pp = _resolve_postprocess_Hg(h5_path, Hg)
    _, qi_field = fm.forward(Hg_pp, phi=0, qi_return=True)
```

Add the helper near the other pipeline module-level helpers:

```python
def _resolve_postprocess_Hg(h5_path: Path, Hg: np.ndarray | None) -> np.ndarray:
    """Strain source for postprocess, in priority order.

    1. explicit ``Hg`` argument;
    2. persisted ``/1.1/dfxm_geo/Hg`` from the run's HDF5 (written when
       ``io.write_strain_provenance`` is True);
    3. the legacy ``fm.Hg`` module global (deprecated shim; one release only).
    """
    if Hg is not None:
        return np.asarray(Hg, dtype=float)
    with h5py.File(h5_path, "r") as f:
        if "/1.1/dfxm_geo/Hg" in f:
            return np.asarray(f["/1.1/dfxm_geo/Hg"][()], dtype=float)
    if fm.Hg is not None:
        return np.asarray(fm.Hg, dtype=float)
    raise RuntimeError(
        "Cannot postprocess: no Hg passed, none persisted at "
        "/1.1/dfxm_geo/Hg (run with io.write_strain_provenance=true), and "
        "fm.Hg is unset."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_postprocess_strain_source.py -v`
Expected: PASS.

- [ ] **Step 5: mypy + commit**

Run: `... -m mypy src/dfxm_geo/` → 0 errors.
```
git add tests/test_postprocess_strain_source.py src/dfxm_geo/pipeline.py
git commit -m "feat(#10): run_postprocess recovers Hg from HDF5, not a stale global"
```

### Task 3: Snapshot/restore the full forward state around each run

**Files:**
- Create: `_forward_state_guard` in `src/dfxm_geo/direct_space/forward_model.py`
- Modify: `src/dfxm_geo/pipeline.py` `run_simulation` + `run_identification` bodies (wrap in the guard)
- Test: `tests/test_forward_state_guard.py` (extend)

- [ ] **Step 1: Write the failing test**

A run that loads an oblique/analytic config must not leave mutated globals visible to the next run on the same process.

```python
# add to tests/test_forward_state_guard.py
import contextlib
import numpy as np
from dfxm_geo.direct_space import forward_model as fm


def test_forward_state_guard_restores_all_mutable_globals():
    names = [
        "theta_0", "theta", "Theta", "xl_start", "xl_range", "rl", "prob_z",
        "Resq_i", "qi1_start", "qi1_step", "qi2_start", "qi2_step",
        "qi3_start", "qi3_step", "npoints1", "npoints2", "npoints3",
        "qi_starts", "qi_steps", "_analytic_eval", "Hg", "q_hkl",
        "_loaded_kernel_path",
    ]
    before = {n: getattr(fm, n, None) for n in names}
    with fm._forward_state_guard():
        fm.theta_0 = 1.2345
        fm.Hg = np.array([[9.0]])
        fm._analytic_eval = object()
    after = {n: getattr(fm, n, None) for n in names}
    for n in names:
        b, a = before[n], after[n]
        if isinstance(b, np.ndarray) or isinstance(a, np.ndarray):
            assert np.array_equal(np.asarray(b), np.asarray(a)), n
        else:
            assert b is a or b == a, n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_forward_state_guard.py::test_forward_state_guard_restores_all_mutable_globals -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_forward_state_guard'`.

- [ ] **Step 3: Implement the guard**

Add to `forward_model.py` (near `reflection_theta_if_oblique`, ~line 152):

```python
_GUARDED_GLOBALS = (
    "theta_0", "theta", "Theta", "xl_start", "xl_range", "rl", "prob_z",
    "Resq_i", "qi1_start", "qi1_step", "qi2_start", "qi2_step",
    "qi3_start", "qi3_step", "npoints1", "npoints2", "npoints3",
    "qi_starts", "qi_steps", "_analytic_eval", "Hg", "q_hkl",
    "_loaded_kernel_path",
)


@contextlib.contextmanager
def _forward_state_guard() -> Iterator[None]:
    """Snapshot every mutable forward-model global on entry; restore on exit.

    Makes cross-config leakage on a persistent worker impossible: whatever a
    run mutates (kernel load, oblique theta rebuild, strain field) is rolled
    back when the run completes or raises. Subsumes
    ``reflection_theta_if_oblique``'s restore responsibility.

    Usable as a context manager OR a decorator (``@contextmanager`` produces a
    ``ContextDecorator``): ``@_forward_state_guard()`` re-snapshots on each call.
    ``globals()`` here is forward_model's module dict regardless of where the
    decorator is applied, so it guards exactly these names.
    """
    g = globals()
    saved = {n: g.get(n) for n in _GUARDED_GLOBALS}
    try:
        yield
    finally:
        g.update(saved)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_forward_state_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Make `run_postprocess` self-load its kernel (decouple from `run_simulation`'s globals)**

`run_postprocess` currently calls `fm.forward(...)` relying on `fm.Resq_i` being loaded by a prior `run_simulation` *in the same process* (`cli_main` line 1232-1234). It loads no kernel itself, so `dfxm-forward --postprocess-only` (cli_main line 1230) is already a latent crash on a fresh process (`fm.forward` → "forward_model state is not initialized"). Once `run_simulation` is guarded (Step 6), it restores `fm.Resq_i=None` on exit, breaking the *normal* flow too. Fix: make `run_postprocess` load its own resolution backend, exactly as `run_simulation` does at pipeline.py:819.

Add near the top of `run_postprocess`, right after the `h5_path` existence check:
```python
    _load_resolution(config.reciprocal, config.geometry)
```
This is the same call `run_simulation` makes (handles both MC and analytic backends via `config.backend`). For the normal flow it reloads the identical kernel the config selects → the `qi_field` is **bit-identical** (same `Resq_i`, same default `theta_0` for simplified; oblique `qi_field` theta behavior is unchanged — `run_postprocess` did not wrap `reflection_theta_if_oblique` before and still doesn't, a pre-existing detail handled in Slice 4). It also fixes `--postprocess-only`.

A `_run_simulation_inner`-style `_load_resolution` call is cheap relative to the postprocess work, and idempotent.

- [ ] **Step 6: Apply the guard to all three run functions**

In `pipeline.py`, decorate `run_simulation`, `run_identification`, AND `run_postprocess` with `@fm._forward_state_guard()` (decorator form — no body re-indentation; each call gets a fresh snapshot restored on return/raise, including any kernel load done inside the run). Guarding `run_postprocess` too serves the #10 no-leakage goal directly (a worker that postprocesses shouldn't leak globals either) and is safe because Step 5 made it self-loading. Keep `reflection_theta_if_oblique` where it is — the guard owns the *restore*, the CM still owns the geometry *rebuild* during the run.

```python
@fm._forward_state_guard()
def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    ...

@fm._forward_state_guard()
def run_identification(config: SimulationConfig, output_dir: Path) -> ...:
    ...

@fm._forward_state_guard()
def run_postprocess(output_dir: Path, config: SimulationConfig, *, Hg=None, q_hkl=None) -> dict[str, Any]:
    ...
```

- [ ] **Step 6b: Self-load regression test**

Add to `tests/test_forward_state_guard.py` a test proving `run_postprocess` no longer free-rides on a prior `run_simulation`'s globals. Reuse the kernel-on-disk gate + the wall-mode config from `tests/test_postprocess_strain_source.py`. After `run_simulation` (now guarded → it restores `fm.Resq_i=None` on exit), assert `fm.Resq_i is None` *before* `run_postprocess`, then assert `run_postprocess` still writes `/1.1/dfxm_geo/analysis/qi_field` (proving self-load):

```python
def test_run_postprocess_self_loads_kernel_after_guarded_run_simulation(
    tmp_path, _kernel_on_disk
):
    from dfxm_geo.pipeline import (AxisScanConfig, CrystalConfig, IOConfig,
        ReciprocalConfig, ScanConfig, SimulationConfig, WallCrystalConfig,
        run_postprocess, run_simulation)
    cfg = SimulationConfig(
        crystal=CrystalConfig(mode="wall", wall=WallCrystalConfig(dis=4.0, ndis=10, sample_remount="S1")),
        scan=ScanConfig(phi=AxisScanConfig(range=0.0006*180/np.pi, steps=3),
                        chi=AxisScanConfig(range=0.002*180/np.pi, steps=3)),
        io=IOConfig(include_perfect_crystal=True, max_workers=1),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    assert fm.Resq_i is None, "guard should have restored Resq_i to None on run_simulation exit"
    run_postprocess(out, cfg)   # must self-load the kernel, not crash
    import h5py
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
```
(Define a `_kernel_on_disk` fixture in this file identical to the one in `test_postprocess_strain_source.py`.)

- [ ] **Step 7: Slice-1 gate — full kernel-present verification**

Bootstrap the canonical kernel, then:
```
... -m pytest -q tests/test_forward_state_guard.py tests/test_postprocess_strain_source.py
... -m pytest -q -k "forward or reflection or oblique or identification or postprocess or pipeline"
... -m mypy src/dfxm_geo/
```
Expected: all green, mypy 0 errors. The Fd_find golden (`tests/data/golden/Fd_find_smoke.npy`) reproduces. `_mc_lut_forward` not recompiled (no hot-path edit this slice).

- [ ] **Step 8: Commit**

```
git add tests/test_forward_state_guard.py src/dfxm_geo/direct_space/forward_model.py src/dfxm_geo/pipeline.py
git commit -m "feat(#10): self-loading run_postprocess + snapshot/restore globals around each run"
```

---

## SLICE 2 — introduce the contexts as return values (additive, still shimmed)

Pure-additive: define the dataclasses + `build_*_context` constructors + make the loaders *also* return a `ResolutionContext`. No caller changes; globals still set. The Slice-2 parity oracle is the foundation for all later bit-identity gates.

### Task 4: Define the context dataclasses

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (add dataclasses near top, after the imports / `_S_IDENTITY` ~line 36)

- [ ] **Step 1: Add the dataclasses**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class InstrumentContext:
    """Per-process, reflection-independent ray grid + detector geometry."""
    psize: float
    zl_rms: float
    Npixels: int
    Nsub: int
    NN1: int
    NN2: int
    NN3: int
    Ud: np.ndarray            # (3, 3)
    Us: np.ndarray            # (3, 3)
    flat_indices: np.ndarray  # (N,) int64, C-order scatter map
    yl_start: float
    xl_steps: int
    yl_steps: int
    zl_steps: int


@dataclass(frozen=True)
class GeometryContext:
    """Per-reflection Bragg geometry + the ray grid it drives."""
    theta_0: float
    Theta: np.ndarray   # (3, 3)
    xl_start: float
    xl_range: float
    rl: np.ndarray      # (3, N) metres
    prob_z: np.ndarray  # (N,) beam profile weight


@dataclass(frozen=True)
class ResolutionContext:
    """Per-reflection resolution backend (exactly one of the two is set)."""
    Resq_i: "np.ndarray | None"
    qi1_start: float
    qi1_step: float
    qi2_start: float
    qi2_step: float
    qi3_start: float
    qi3_step: float
    npoints1: "int | None"
    npoints2: "int | None"
    npoints3: "int | None"
    analytic_eval: "AnalyticResolution | None"
    loaded_kernel_path: "Path | None"


@dataclass(frozen=True)
class ForwardContext:
    """Everything forward_from_static needs, bundled. Immutable + thread-safe."""
    instrument: InstrumentContext
    geometry: GeometryContext
    resolution: ResolutionContext
```

- [ ] **Step 2: mypy + commit**

Run: `... -m mypy src/dfxm_geo/` → 0 errors. (`AnalyticResolution` is imported lazily today; use the string annotation as shown so no import cycle.)
```
git add src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): add ForwardContext dataclasses (additive, unused)"
```

### Task 5: `build_instrument_context` + `build_geometry_context`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_forward_context.py` (create)

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_forward_context.py
import numpy as np
from dfxm_geo.direct_space import forward_model as fm


def test_build_instrument_context_matches_globals():
    instr = fm.build_instrument_context()
    assert instr.psize == fm.psize
    assert instr.zl_rms == fm.zl_rms
    assert instr.Npixels == fm.Npixels
    assert instr.Nsub == fm.Nsub
    assert (instr.NN1, instr.NN2, instr.NN3) == (fm.NN1, fm.NN2, fm.NN3)
    assert np.array_equal(instr.Ud, fm.Ud)
    assert np.array_equal(instr.Us, fm.Us)
    assert np.array_equal(instr.flat_indices, fm._flat_indices)
    assert instr.yl_start == fm.yl_start
    assert (instr.xl_steps, instr.yl_steps, instr.zl_steps) == (
        fm.xl_steps, fm.yl_steps, fm.zl_steps,
    )


def test_build_geometry_context_matches_default_globals():
    instr = fm.build_instrument_context()
    geom = fm.build_geometry_context(fm.theta_0, instr)
    assert geom.theta_0 == fm.theta_0
    assert np.array_equal(geom.Theta, fm.Theta)
    assert geom.xl_start == fm.xl_start
    assert geom.xl_range == fm.xl_range
    assert np.array_equal(geom.rl, fm.rl)
    assert np.array_equal(geom.prob_z, fm.prob_z)
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_forward_context.py -v`
Expected: FAIL — `build_instrument_context` not defined.

- [ ] **Step 3: Implement the constructors**

`build_instrument_context` reads the import-time instrument constants verbatim; `build_geometry_context` reuses the *exact* expressions from `reflection_theta_if_oblique` (so the oblique path is reproduced bit-for-bit) but applies them unconditionally at the passed `theta_0` — this is what fixes the simplified-staleness bug.

```python
def build_instrument_context() -> InstrumentContext:
    return InstrumentContext(
        psize=psize, zl_rms=zl_rms, Npixels=Npixels, Nsub=Nsub,
        NN1=NN1, NN2=NN2, NN3=NN3, Ud=Ud, Us=Us,
        flat_indices=_flat_indices, yl_start=yl_start,
        xl_steps=xl_steps, yl_steps=yl_steps, zl_steps=zl_steps,
    )


def build_geometry_context(theta_0_: float, instrument: InstrumentContext) -> GeometryContext:
    th = float(theta_0_)
    Theta_ = np.array(
        [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    )
    xl_start_ = instrument.yl_start / np.tan(2 * th) / 3
    xl_range_ = -xl_start_
    rl_ = np.vstack(  # type: ignore[call-overload]
        np.mgrid[
            -xl_range_ : xl_range_ : complex(instrument.xl_steps),
            -yl_range : yl_range : complex(instrument.yl_steps),
            -zl_range : zl_range : complex(instrument.zl_steps),
        ]
    ).reshape(3, -1)
    prob_z_ = np.exp(-0.5 * (rl_[2] / instrument.zl_rms) ** 2)
    return GeometryContext(
        theta_0=th, Theta=Theta_, xl_start=xl_start_, xl_range=xl_range_,
        rl=rl_, prob_z=prob_z_,
    )
```
(`yl_range`, `zl_range` remain module constants — they are reflection-independent.)

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/test_forward_context.py -v`
Expected: PASS — geometry built at the default `theta_0` reproduces the default globals bit-for-bit.

- [ ] **Step 5: mypy + commit**

```
git add tests/test_forward_context.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): build_instrument_context + build_geometry_context with parity test"
```

### Task 6: Loaders also return a `ResolutionContext`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` `_load_default_kernel` (330-441), `_load_analytic_resolution` (548-577)
- Test: `tests/test_forward_context.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_forward_context.py — requires a bootstrapped kernel
import pytest
from pathlib import Path
from dfxm_geo.direct_space import forward_model as fm

_KERNELS = list((Path(fm.pkl_fpath)).glob("*.npz"))


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_load_default_kernel_returns_matching_resolution_context():
    res = fm._load_default_kernel(_KERNELS[0], compute_Hg=False)
    assert res is not None
    assert np.array_equal(res.Resq_i, fm.Resq_i)
    assert res.qi1_start == fm.qi1_start and res.qi1_step == fm.qi1_step
    assert res.qi2_start == fm.qi2_start and res.qi2_step == fm.qi2_step
    assert res.qi3_start == fm.qi3_start and res.qi3_step == fm.qi3_step
    assert (res.npoints1, res.npoints2, res.npoints3) == (
        fm.npoints1, fm.npoints2, fm.npoints3,
    )
    assert res.analytic_eval is None
    assert res.loaded_kernel_path == fm._loaded_kernel_path
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_forward_context.py::test_load_default_kernel_returns_matching_resolution_context -v`
Expected: FAIL — `_load_default_kernel` returns `None` today (its annotated return type is `-> None`).

- [ ] **Step 3: Make the loaders return a `ResolutionContext`**

Keep the global assignments (shim) and *additionally* build + return a `ResolutionContext` at the end of each loader. Change `_load_default_kernel`'s return annotation `-> None` → `-> ResolutionContext` and append:

```python
    # ... existing global assignments stay (shim) ...
    res = ResolutionContext(
        Resq_i=Resq_i, qi1_start=qi1_start, qi1_step=qi1_step,
        qi2_start=qi2_start, qi2_step=qi2_step, qi3_start=qi3_start,
        qi3_step=qi3_step, npoints1=npoints1, npoints2=npoints2,
        npoints3=npoints3, analytic_eval=None,
        loaded_kernel_path=_loaded_kernel_path,
    )
    return res
```

For `_load_analytic_resolution`, return:
```python
    res = ResolutionContext(
        Resq_i=None, qi1_start=0.0, qi1_step=0.0, qi2_start=0.0, qi2_step=0.0,
        qi3_start=0.0, qi3_step=0.0, npoints1=None, npoints2=None,
        npoints3=None, analytic_eval=_analytic_eval, loaded_kernel_path=None,
    )
    return res
```
Update both annotations and any caller that relied on `-> None` (none assign the result yet; mypy will confirm).

- [ ] **Step 4: Run to verify it passes (kernel present)**

Bootstrap the canonical kernel, then:
Run: `... -m pytest tests/test_forward_context.py -v`
Expected: PASS.

- [ ] **Step 5: mypy + Slice-2 gate + commit**

```
... -m mypy src/dfxm_geo/
... -m pytest -q -k "forward or reflection or oblique or identification or pipeline or context"
git add tests/test_forward_context.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): kernel/analytic loaders return ResolutionContext (parity-checked)"
```

---

## SLICE 3 — thread `ctx` through the hot path (bit-identity gated)

Add `ctx: ForwardContext | None = None` to the three forward functions with a `_context_from_globals()` fallback so existing callers keep working unchanged. Verify bit-identity with `np.array_equal` and assert no numba recompile.

### Task 7: `_context_from_globals` + `build_forward_context`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_forward_context.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_forward_context.py — kernel present
@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_context_from_globals_roundtrips(monkeypatch):
    fm._load_default_kernel(_KERNELS[0], compute_Hg=False)
    ctx = fm._context_from_globals()
    assert np.array_equal(ctx.geometry.Theta, fm.Theta)
    assert np.array_equal(ctx.geometry.rl, fm.rl)
    assert np.array_equal(ctx.resolution.Resq_i, fm.Resq_i)
    assert ctx.geometry.theta_0 == fm.theta_0
    assert ctx.instrument.Nsub == fm.Nsub
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `_context_from_globals` not defined.

- [ ] **Step 3: Implement**

```python
def _context_from_globals() -> ForwardContext:
    """Build a ForwardContext snapshotting the current module globals.

    The migration fallback: lets ctx-threaded functions run for callers that
    have not yet been converted. Deleted in Slice 5.
    """
    instr = build_instrument_context()
    geom = GeometryContext(
        theta_0=theta_0, Theta=Theta, xl_start=xl_start, xl_range=xl_range,
        rl=rl, prob_z=prob_z,
    )
    res = ResolutionContext(
        Resq_i=Resq_i, qi1_start=qi1_start, qi1_step=qi1_step,
        qi2_start=qi2_start, qi2_step=qi2_step, qi3_start=qi3_start,
        qi3_step=qi3_step, npoints1=npoints1, npoints2=npoints2,
        npoints3=npoints3, analytic_eval=_analytic_eval,
        loaded_kernel_path=_loaded_kernel_path,
    )
    return ForwardContext(instrument=instr, geometry=geom, resolution=res)


def build_forward_context(theta_0_: float | None = None) -> ForwardContext:
    """Compose a context for a run. theta_0_ defaults to the current global."""
    instr = build_instrument_context()
    geom = build_geometry_context(theta_0 if theta_0_ is None else theta_0_, instr)
    res = ResolutionContext(
        Resq_i=Resq_i, qi1_start=qi1_start, qi1_step=qi1_step,
        qi2_start=qi2_start, qi2_step=qi2_step, qi3_start=qi3_start,
        qi3_step=qi3_step, npoints1=npoints1, npoints2=npoints2,
        npoints3=npoints3, analytic_eval=_analytic_eval,
        loaded_kernel_path=_loaded_kernel_path,
    )
    return ForwardContext(instrument=instr, geometry=geom, resolution=res)
```

- [ ] **Step 4: Run + commit**

```
... -m pytest tests/test_forward_context.py -v   # kernel present
... -m mypy src/dfxm_geo/
git add tests/test_forward_context.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): _context_from_globals + build_forward_context"
```

### Task 8: Add `ctx` param to `precompute_forward_static`, `forward_from_static`, `forward`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (580-591, 664-766, 769-825)
- Test: `tests/test_forward_context.py` (dual-path bit-identity)

- [ ] **Step 1: Write the failing dual-path bit-identity test**

```python
# add to tests/test_forward_context.py — kernel present
@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_forward_ctx_path_bit_identical_to_globals_path():
    fm._load_default_kernel(_KERNELS[0], compute_Hg=True)
    Hg = fm.Hg
    n_sig_before = len(fm._mc_lut_forward.signatures)
    img_globals = fm.forward(Hg, phi=1e-4, chi=2e-4)
    ctx = fm._context_from_globals()
    img_ctx = fm.forward(Hg, ctx=ctx, phi=1e-4, chi=2e-4)
    assert np.array_equal(img_ctx, img_globals)   # bit-exact, not allclose
    assert len(fm._mc_lut_forward.signatures) == n_sig_before  # no recompile
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `forward()` does not accept `ctx`.

- [ ] **Step 3: Thread `ctx` with a globals fallback**

`precompute_forward_static`:
```python
def precompute_forward_static(Hg: np.ndarray, ctx: "ForwardContext | None" = None) -> np.ndarray:
    Us_ = Us if ctx is None else ctx.instrument.Us
    q_hkl_ = q_hkl if ctx is None else q_hkl  # q_hkl stays a module/strain value
    qs = Us_ @ Hg @ q_hkl_
    return qs.squeeze().T
```
(Note: `q_hkl` is strain, not in `ForwardContext` by design — keep reading the global here; it is threaded explicitly elsewhere. The only ctx-sourced field is `Us`.)

`forward_from_static`: introduce a local `ctx = ctx or _context_from_globals()` at the top, then source every former-global read from `ctx`:
```python
def forward_from_static(
    base_qc: np.ndarray,
    ctx: "ForwardContext | None" = None,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    ctx = ctx if ctx is not None else _context_from_globals()
    res = ctx.resolution
    geom = ctx.geometry
    instr = ctx.instrument
    if res.Resq_i is None and res.analytic_eval is None:
        raise RuntimeError(...)  # unchanged message
    if TwoDeltaTheta != 0:
        th = geom.theta_0 + TwoDeltaTheta
        Theta_ = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    else:
        Theta_ = geom.Theta
    im_1 = np.zeros([(instr.NN2 // instr.Nsub), instr.NN1 // instr.Nsub])
    ang0 = phi - TwoDeltaTheta / 2
    ang1 = float(chi)
    ang2 = (TwoDeltaTheta / 2) / np.tan(geom.theta_0)
    qi = None
    if res.analytic_eval is not None or qi_return:
        qc = base_qc + np.asarray([[ang0],[ang1],[ang2]])
        qi = Theta_ @ qc
    if res.analytic_eval is not None:
        prob = (res.analytic_eval(qi) * geom.prob_z).astype(np.float32)
        contribution = np.bincount(instr.flat_indices, weights=prob, minlength=im_1.size)
        im_1 += contribution.reshape(im_1.shape)
        if qi_return:
            return im_1, qi.reshape(3, instr.NN1, instr.NN2, instr.NN3)
        return im_1
    assert res.Resq_i is not None
    assert res.npoints1 is not None and res.npoints2 is not None and res.npoints3 is not None
    _mc_lut_forward(
        base_qc, ang0, ang1, ang2, Theta_,
        geom.prob_z, instr.flat_indices,
        res.Resq_i.reshape(-1), im_1.reshape(-1),
        res.qi1_start, res.qi1_step, res.qi2_start, res.qi2_step,
        res.qi3_start, res.qi3_step,
        res.npoints1, res.npoints2, res.npoints3,
        res.npoints2 * res.npoints3, res.npoints3,
    )
    if qi_return:
        assert qi is not None
        return im_1, qi.reshape(3, instr.NN1, instr.NN2, instr.NN3)
    return im_1
```
**Critical:** the `qi.reshape(3, NN1, NN2, NN3)` uses `instr.NN1/2/3` — verify these equal the former globals (they do, via `build_instrument_context`). Float ops are byte-for-byte the originals.

`forward`:
```python
def forward(
    Hg: np.ndarray,
    ctx: "ForwardContext | None" = None,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    ctx = ctx if ctx is not None else _context_from_globals()
    # keep the existing init guard, sourcing from ctx.resolution
    base_qc = precompute_forward_static(Hg, ctx)
    return forward_from_static(base_qc, ctx, phi, chi, TwoDeltaTheta, qi_return)
```

**Caller-signature note:** `forward`/`forward_from_static` gained a *second positional* param (`ctx`). Audit existing positional callers: `io/hdf5.py:_compute_frame` calls `forward_from_static(base_qc, phi=phi, chi=chi, ...)` (keyword — safe). `run_postprocess` calls `fm.forward(Hg_pp, phi=0, qi_return=True)` (keyword — safe). Any positional `forward(Hg, phi)` caller would now pass `phi` as `ctx` — grep `forward_from_static(` and `\.forward(` across `src/` and `tests/` and fix any positional second-arg call to use `ctx=None` or keywords.

- [ ] **Step 4: Run dual-path test (kernel present)**

Run: `... -m pytest tests/test_forward_context.py::test_forward_ctx_path_bit_identical_to_globals_path -v`
Expected: PASS — `np.array_equal` True, signature count unchanged.

- [ ] **Step 5: Slice-3 gate + commit**

```
... -m pytest -q -k "forward or reflection or oblique or context"   # kernel present
... -m mypy src/dfxm_geo/
git add tests/test_forward_context.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): thread ForwardContext through hot path (bit-identical, no recompile)"
```

### Task 9: `io/hdf5.py` carries and passes `ctx`

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` `_compute_frame` (120-130) + `write_simulation_h5`/`write_identification_h5` (pass `ctx` into precompute/frame compute)
- Test: existing `tests/test_hdf5_*` (must stay green)

- [ ] **Step 1: Add `ctx` to `_FrameArgs` + `_compute_frame`**

`ctx` is shared read-only across all frames of a scan, so it rides on the writer call, not duplicated per frame. Add a writer-level `ctx` and include it in `_FrameArgs`:
```python
def _compute_frame(args: _FrameArgs) -> tuple[int, np.ndarray]:
    frame_idx, base_qc, phi, chi, two_dtheta, ctx = args
    im = cast(np.ndarray, _fm.forward_from_static(
        base_qc, ctx, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta))
    return frame_idx, im
```
Update `_FrameArgs` (the type alias / namedtuple) to a 6-tuple and every site that builds it to append `ctx`. `precompute_forward_static(Hg_in)` → `precompute_forward_static(Hg_in, ctx)` at hdf5.py:737 and :816.

- [ ] **Step 2: Thread `ctx` into the writers**

Add `ctx: "_fm.ForwardContext | None" = None` to `write_simulation_h5` / `write_identification_h5`; default `None` → `ctx = _fm._context_from_globals()` at the top of each (so current callers unchanged). Build `_FrameArgs` with this `ctx`.

- [ ] **Step 3: Verify (kernel present) + commit**

```
... -m pytest -q tests/test_hdf5_pipeline.py tests/test_hdf5_run_simulation_end_to_end.py tests/test_master_writer.py tests/test_detector_dtype_float32.py
... -m mypy src/dfxm_geo/
git add src/dfxm_geo/io/hdf5.py
git commit -m "feat(#16): hdf5 writers thread ForwardContext (default builds from globals)"
```

---

## SLICE 4 — thread `ctx` through pipeline + postprocess + generators

### Task 10: `run_simulation`/`run_postprocess` build and pass `ctx`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` `run_simulation` (build ctx, pass to writer + Hg_provider), `run_postprocess` (build ctx, pass Hg + ctx to `forward`)

- [ ] **Step 1: `run_simulation`**

After kernel load + oblique geometry setup, build the run's `ctx` and pass it to `write_simulation_h5(..., ctx=ctx)`. Remove the `fm.Hg = Hg_base` / `fm.q_hkl = q_hkl` writes (lines 907-908) — the writer no longer reads them off the module; `Hg` is passed explicitly (already is, line 917). Behind the shim the globals still resolve via `__getattr__` for any straggler reader; the explicit writes go away.

```python
    Hg_base, q_hkl = Hg_provider(0.0)
    ctx = fm.build_forward_context()   # built at the run's current theta_0
    ...
    write_simulation_h5(h5_path, Hg=Hg_base, q_hkl=q_hkl, ctx=ctx, ...)
```

- [ ] **Step 2: `run_postprocess`**

```python
    ctx = fm.build_forward_context()
    Hg_pp = _resolve_postprocess_Hg(h5_path, Hg)
    _, qi_field = fm.forward(Hg_pp, ctx=ctx, phi=0, qi_return=True)
```

- [ ] **Step 3: Verify + commit**

```
... -m pytest -q -k "pipeline or postprocess or hdf5"   # kernel present
... -m mypy src/dfxm_geo/
git add src/dfxm_geo/pipeline.py
git commit -m "feat(#16): run_simulation/run_postprocess thread ForwardContext"
```

### Task 11: Identify generators + `Find_Hg*` take `ctx`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` `_iter_identification_single/multi/zscan` (read `ctx.geometry.*`/`ctx.instrument.*` instead of `fm.theta`/`fm.Us`/`fm.Theta`/`fm.rl`/`fm.psize`/`fm.zl_rms`); `run_identification` builds + passes `ctx`
- Modify: `Find_Hg` / `Find_Hg_from_population` signatures (they read `Theta`/`rl`) — add `ctx` param with globals fallback

- [ ] **Step 1: Thread `ctx` into `run_identification` + the generators**

`run_identification` builds `ctx = fm.build_forward_context()` (inside the `_forward_state_guard` from Slice 1) and passes it to whichever `_iter_identification_*` it dispatches. Each generator takes `ctx` and replaces:
- `fm.theta` → `ctx.geometry.theta_0` (note: `fm.theta == fm.theta_0` always; confirm at the call sites in 1562-1564 / 1763-1765)
- `fm.Us` → `ctx.instrument.Us`
- `fm.Theta` → `ctx.geometry.Theta`
- `fm.rl` → `ctx.geometry.rl`
- `fm.psize` → `ctx.instrument.psize`
- `fm.zl_rms` → `ctx.instrument.zl_rms`
- `fm.Z_shift(...)` / `fm.build_scan_grid(...)` stay as function calls (not state) — but `Z_shift` reads `rl`; give it a `ctx`/`rl` argument too if it reads the global (grep `def Z_shift`).

- [ ] **Step 2: `Find_Hg` / `Find_Hg_from_population` take `ctx`**

Add `ctx: ForwardContext | None = None`; inside, `rl_ = ctx.geometry.rl if ctx else rl`, `Theta_ = ctx.geometry.Theta if ctx else Theta`. Update the generator + writer callers to pass `ctx`.

- [ ] **Step 3: Slice-4 geometry-regression gate (the real watch-list)**

```
... -m pytest -q tests/test_identification_oblique_wiring.py tests/test_forward_reflection_theta.py
... -m pytest -q -k "oblique or identification or provenance or contrast"
... -m mypy src/dfxm_geo/
```
These exercise the one place behavior *should* be unchanged but the code path moved. All must stay green with the existing goldens (no regeneration).

- [ ] **Step 4: Commit**

```
git add src/dfxm_geo/pipeline.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "feat(#16): identify generators + Find_Hg* thread ForwardContext"
```

---

## SLICE 5 — delete the shim

> **SUPERSEDED 2026-06-03.** Scoping found Slice 5 is not a mechanical "delete + grep": the loader→context flow round-trips through the globals, so deletion requires inverting the loaders to return their `ResolutionContext` and threading it. The full staged re-plan (sub-tasks S0–S5: `run_theta` resolver + simplified-θ fix → loaders return ctx → break the round-trip → retire the oblique CM → convert migrate/images/numpy-oracle readers → delete globals + guard) lives in **`docs/superpowers/plans/2026-06-03-forward-context-slice5.md`**. Execute that; the Task-12 sketch below is retained only for history.

### Task 12 (HISTORICAL SKETCH — see the Slice 5 doc above): PEP-562 `__getattr__` shim (transitional) then deletion

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Modify: `src/dfxm_geo/io/migrate.py` (source provenance from HDF5/ctx)

- [ ] **Step 1: Add the transitional shim (if any non-converted reader remains)**

If, after Slices 1-4, a repo-wide grep still shows production readers of `fm.theta`/`fm.Theta`/`fm.rl`/`fm.prob_z`/`fm.Resq_i`/`fm._analytic_eval`/`fm.Hg`/`fm.q_hkl`/`fm._loaded_kernel_path`, add a `__getattr__` that delegates to a process-global `_CURRENT_CTX` for the transition. If the grep already shows zero non-test production readers, skip straight to Step 3.

```python
_CURRENT_CTX: "ForwardContext | None" = None

def __getattr__(name: str):  # PEP 562
    ctx = _CURRENT_CTX
    if ctx is not None:
        if name in ("theta", "theta_0"):
            return ctx.geometry.theta_0
        if name == "Theta":
            return ctx.geometry.Theta
        if name in ("rl", "prob_z", "xl_start", "xl_range"):
            return getattr(ctx.geometry, name)
        if name in ("Resq_i",):
            return ctx.resolution.Resq_i
        if name == "_analytic_eval":
            return ctx.resolution.analytic_eval
        if name == "_loaded_kernel_path":
            return ctx.resolution.loaded_kernel_path
    raise AttributeError(name)
```

- [ ] **Step 2: Verify the shim keeps stragglers green**

Run the full kernel-present suite; fix any remaining reader to thread `ctx` instead, until the grep in Step 3 is clean.

- [ ] **Step 3: Delete the globals + CM + fallback (gated on a clean grep)**

Repo-wide grep gate (must return only deleted-test diffs / comments):
```
rg "fm\.(Hg|q_hkl|theta|theta_0|Theta|rl|prob_z|Resq_i|_analytic_eval|_loaded_kernel_path)\b" src/
rg "reflection_theta_if_oblique" src/
rg "_context_from_globals" src/
```
**CRITICAL — the `fm.` grep does NOT catch BARE-NAME readers inside `forward_model.py` itself** (it reads its own globals as `Theta`, `rl`, `theta_0`, … not `fm.Theta`). Slice 4's sweep already found the live intra-module readers of the to-be-deleted globals — handle each before deleting:
  - `_find_hg_from_population_numpy` (~line 1414) reads `Theta` (and `Us`, which is kept) as bare globals — it's the test-only NumPy parity oracle for `find_hg_population`. Thread `ctx` into it (and update the parity test that calls it to pass `ctx=_context_from_globals()` *before* `_context_from_globals` is deleted, OR pass an explicitly-built `ForwardContext`), OR keep it reading via a small shim. Do NOT let it `NameError` after the deletion.
  - `reflection_theta_if_oblique` reads/writes the deleted globals — it is itself deleted in this slice.
  - `_context_from_globals` / `build_forward_context` read them intentionally (the snapshot source) — deleted with the shim here.
  - `Find_Hg`, `Find_Hg_from_population`, `precompute_forward_static`, `forward_from_static`, `forward` already take `ctx` (Slices 3-4) — verify their callers always pass it once the fallback is removed.
Re-grep `forward_model.py` for bare `\b(Theta|rl|prob_z|theta_0|Resq_i|qi1_start|_analytic_eval|Hg|q_hkl|_loaded_kernel_path)\b` reads in any function body before deleting, to be sure no other intra-module reader remains.

Then delete: the per-reflection/per-resolution/strain module globals (`theta_0`, `theta`, `Theta`, `xl_start`, `xl_range`, `rl`, `prob_z`, `Resq_i`, `qi*`, `npoints*`, `qi_starts/steps`, `_analytic_eval`, `Hg`, `q_hkl`, `_loaded_kernel_path`), `reflection_theta_if_oblique`, `_context_from_globals`, and the `ctx=None` fallbacks in the three forward functions (make `ctx` a required positional). **Keep** the instrument constants (`psize`, `zl_rms`, `Npixels`, `Nsub`, `NN1/2/3`, `Us`, `Ud`, `_flat_indices`, `yl_start`, `xl/yl/zl_steps/range`) — `InstrumentContext` is built from them and they are reflection-independent (spec "Out of scope").

- [ ] **Step 4: Update `io/migrate.py`**

Repoint any `fm.theta`/`fm._loaded_kernel_path` reads in `migrate.py` to the HDF5 provenance group (`/dfxm_geo/...`) or a built `ctx`.

- [ ] **Step 5: Final gate**

```
... -m pytest -q                      # full suite, kernel present
... -m mypy src/dfxm_geo/
```
Expected: full suite green (same pass count as pre-refactor minus any intentionally-deleted shim tests), mypy 0 errors. Fd_find golden + all bit-equiv/snapshot goldens unchanged and reproducing.

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/direct_space/forward_model.py src/dfxm_geo/io/migrate.py
git commit -m "refactor(#16): delete forward_model globals + reflection_theta_if_oblique shim"
```

---

## Post-refactor: release notes + behavior-change test

### Task 13: Document the simplified-reflection theta fix + add its test

**Files:**
- Create/modify: `tests/test_forward_reflection_theta.py` (add the behavior-change assertion)
- Modify: release notes / `docs/` changelog for v2.4.0

- [ ] **Step 1: Behavior-change test**

The simplified path now builds geometry at the run's actual Bragg angle, not the import-time Al(-1,1,-1) default. Add a test asserting a non-default *simplified* reflection uses its own theta:

```python
def test_simplified_nondefault_reflection_uses_own_theta():
    """Pre-refactor, a simplified run at a non-default hkl used the stale
    import-time Theta. Now build_geometry_context(theta_run) is unconditional."""
    instr = fm.build_instrument_context()
    theta_run = fm.theta_0 * 1.05  # any non-default Bragg angle
    geom = fm.build_geometry_context(theta_run, instr)
    assert geom.theta_0 == theta_run
    assert not np.array_equal(geom.Theta, fm.build_geometry_context(fm.theta_0, instr).Theta)
```

- [ ] **Step 2: Release notes**

Note in the v2.4.0 notes: "Simplified-mode forward/identify at a non-default reflection now computes geometry at that reflection's Bragg angle (previously used the import-time Al(-1,1,-1)@17keV default). Confirmed OK by Sina — no golden was made at a non-default simplified hkl." (per spec Risks & mitigations).

- [ ] **Step 3: Commit**

```
git add tests/test_forward_reflection_theta.py docs/
git commit -m "docs(#16): document simplified-reflection theta fix + add behavior test"
```

---

## Self-review checklist (run before handing off to execution)

1. **Spec coverage:** Slice 1 ↔ #10 hardening (3 tasks); Slice 2 ↔ dataclasses + loaders return ctx (parity oracle); Slice 3 ↔ hot-path threading + dual-path bit-identity + no-recompile; Slice 4 ↔ pipeline/postprocess/generators/Find_Hg; Slice 5 ↔ shim + deletion + migrate.py; Task 13 ↔ simplified-theta behavior-change test + release notes. All spec sections mapped.
2. **Bit-exactness:** every numeric gate uses `np.array_equal`; goldens never regenerated.
3. **No-recompile:** asserted via `len(_mc_lut_forward.signatures)` in Task 8.
4. **Positional-arg hazard:** Task 8 Step 3 explicitly audits second-positional `forward(...)`/`forward_from_static(...)` callers after inserting `ctx`.
5. **Out-of-scope respected:** instrument constants stay module-level; no worker-pool build; no multi-reflection iteration API; no `.cif` work.

---

## Execution note for the orchestrator

Slices are independent merge/review checkpoints. Slice 1 is independently shippable (bit-exact, no API churn) — consider it the minimum viable landing if the arc is interrupted. Slices 2-5 are the full #16 migration. After Slice 5, dispatch a spec-review + code-quality-review subagent pair (parallel, per the repo's parallel-subagents rule) before merging the refactor to main.
