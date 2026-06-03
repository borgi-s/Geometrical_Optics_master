# ForwardContext refactor — Slice 5 (delete the globals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. This SUPERSEDES "Task 12" of `2026-06-02-forward-context-refactor.md`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete `forward_model`'s per-reflection geometry/resolution/strain module globals, the `reflection_theta_if_oblique` context manager, `_context_from_globals`, the `ctx=None` fallbacks, and the now-obsolete `_forward_state_guard` — by inverting the loader→context flow so state is *returned and threaded*, never round-tripped through globals. Keep the instrument constants. Also lands the intentional **simplified-reflection θ fix** (a simplified run at a non-default reflection builds geometry at *its* Bragg angle).

**Why this is its own doc:** the original Task 12 assumed "delete + grep." The real blocker (mapped 2026-06-03) is that `_load_resolution` discards the `ResolutionContext` the loaders already build and `build_forward_context()` recovers state by *reading the globals*. Breaking that round-trip touches the loaders, the 3 run functions, `io/hdf5.py`, `io/migrate.py`, `io/images.py`, and the NumPy parity oracle. Staged so the suite stays green until the final deletion.

**Tech stack / conventions:** identical to the parent plan. Venv python `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`; bit-exact gates use `np.array_equal`; NEVER touch goldens; commit per sub-task (Co-Authored-By trailer); don't push. Canonical `(-1,1,-1)` 17 keV seed-0 kernel is on disk under `reciprocal_space/pkl_files/`.

**Key facts (verified 2026-06-03):**
- `_validate_reflection(hkl, keV, a) -> float` (`reciprocal_space/kernel.py:121`) computes the Bragg θ. `ReciprocalConfig` carries `hkl`, `keV`, `lattice_a` (default Al `4.0495e-10`). `_load_analytic_resolution` already calls it (`forward_model.py:~698`).
- `GeometryConfig.theta_validated` (`pipeline.py:446`) is set ONLY for oblique mode (`None` for simplified). So the run θ is: oblique → `theta_validated`; simplified → `_validate_reflection(hkl, keV, lattice_a)`.
- Both loaders (`_load_default_kernel`, `_load_analytic_resolution`) ALREADY return a `ResolutionContext` (Slice 2) — but `_load_resolution`/`_lookup_and_load_kernel` discard it (`pipeline.py:803-808`, `:775`).
- `build_forward_context(theta_0_=None)` reads the resolution globals + the `theta_0` global. All 3 callers pass no arg (`pipeline.py:868`, `:1145`, `:2167`).
- Readers of to-be-deleted globals: `io/hdf5.py` (718/719/784/865, has `_ctx` in scope), `write_identification_h5` (613/614), `io/migrate.py` (130/165/196, NO ctx — standalone CLI), `io/images.py` (82/126, **dead path** — no production caller), `_find_hg_from_population_numpy` (`forward_model.py:~1374`, reads `Theta` bare; called by 2 tests), `pipeline._load_resolution` (writes `fm._analytic_eval=None` at 804/807; reads `fm._loaded_kernel_path` at 773 idempotent-check + 856 logging).
- `q_hkl`/`Hg` globals are set by the loaders (`Hg, q_hkl = Find_Hg(...)`). `q_hkl` is read by the 3 identify generators (`pipeline.py:1570/1761/1977`) + `precompute_forward_static`. `Hg` global is read only by the `_resolve_postprocess_Hg` priority-3 legacy fallback.

---

## θ decision (Sina, 2026-06-03): **true Bragg everywhere; regenerate the default goldens**

`run_theta` uses `_validate_reflection(hkl, keV, lattice_a)` for ALL simplified reflections (no default special-casing) — making the forward geometry θ consistent with the kernel (which is already bootstrapped at the true Bragg angle) and fixing non-default staleness. This shifts the DEFAULT `(-1,1,-1)@17keV` geometry by **5.8e-5 rad** (8.97650°→8.97317°), so the canonical default-reflection goldens **are regenerated** at S2 as a deliberate, documented physics correction. S0/S1 are additive (no behavior change → no golden change); the shift manifests ONLY at S2 when `build_forward_context` starts using `run_theta`. At S2: first enumerate every golden that depends on the default geometry θ (`tests/data/golden/Fd_find_smoke.npy` + any forward/identification golden), regenerate them with the venv python, and commit them in a SEPARATE, clearly-messaged commit ("regenerate goldens: true-Bragg θ-fix, Sina-authorized, +5.8e-5 rad"). The `*bit_equivalent_to_legacy*` golden (already a documented pre-existing failure) tests equivalence to the PICKLE-ERA legacy output — the θ-fix intentionally diverges from it, so convert it to xfail-with-reason or retire it (it no longer has a valid premise), don't regenerate it to pass.

## Design decisions for this slice

1. **`q_hkl` joins `ForwardContext`, computed from `hkl`.** It is a per-reflection constant (`[h,k,l]/√(h²+k²+l²)`, the `B_0=I` form `Find_Hg` uses at `forward_model.py:379-380`), same lifetime as geometry/resolution. Add `q_hkl: np.ndarray` to `ForwardContext` (top-level). **`build_forward_context` takes the reflection `hkl` and computes `q_hkl` internally** (NOT threaded from the `Hg_provider`) — bit-identical to the old global, and every caller already has `config.reciprocal.hkl`, so `run_postprocess` (which has no provider) is covered too. This lets `precompute_forward_static` and the generators read `ctx.q_hkl` and lets the `q_hkl` global be deleted.
2. **The `Hg` loader side-effect goes away.** Production already passes `Hg` explicitly. Drop `compute_Hg`'s global write; the loaders return resolution only. Remove the `_resolve_postprocess_Hg` priority-3 `fm.Hg` fallback (it was "one release only" — Tasks 1-2 already made file-recovery the real path).
3. **`_forward_state_guard` is deleted.** Once no per-run mutable globals exist (loaders return contexts; everything threads `ctx`), cross-config leakage is impossible by construction — the guard guards nothing. Remove it, its `_GUARDED_GLOBALS`, the 3 decorators, and its tests. (The Task-1 `fm.q_hkl`-write guard test also goes, since the `q_hkl` global is gone.)
4. **`migrate.py` builds its own one-off `ForwardContext`** for the legacy IUCrJ-2024 Al `(-1,1,-1)`@17 keV reflection (it already loads/needs a kernel for its `Find_Hg` call); source θ + kernel-path from that ctx.
5. **`images.py` dead `.npy` forward path: delete it** (no production caller; confirmed). If a test references `save_image`/`save_images_parallel`, delete/adjust that test too.

---

## Sub-task S0 — `run_theta(config)` resolver + simplified-θ behavior test (pure addition)

**Files:** `src/dfxm_geo/pipeline.py`; test `tests/test_run_theta.py` (new). Also folds in the parent plan's **Task 13** behavior test.

- [ ] **Step 1: failing test** — assert the resolver yields the reflection's own Bragg angle for a non-default *simplified* reflection (not the import default `theta_0`):
```python
# tests/test_run_theta.py
import numpy as np
from dfxm_geo.pipeline import run_theta, SimulationConfig, ReciprocalConfig
from dfxm_geo.reciprocal_space.kernel import _validate_reflection

def test_run_theta_simplified_uses_reflection_bragg_angle():
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(2, 2, 0), keV=17.0))
    expected = _validate_reflection((2, 2, 0), 17.0, cfg.reciprocal.lattice_a)
    assert run_theta(cfg) == expected

def test_run_theta_oblique_uses_theta_validated():
    # build a minimal oblique config whose geometry.theta_validated is set;
    # reuse the smallest oblique fixture in tests/ (grep theta_validated).
    ...  # assert run_theta(cfg) == cfg.geometry.theta_validated
```
- [ ] **Step 2:** run → FAIL (`run_theta` undefined).
- [ ] **Step 3: implement** in `pipeline.py`:
```python
def run_theta(config: "SimulationConfig") -> float:
    """The run's Bragg angle (rad): oblique → solver theta_validated;
    simplified → computed from the reflection (hkl, keV, lattice_a)."""
    geom = config.geometry
    if geom.mode == "oblique" and geom.theta_validated is not None:
        return float(geom.theta_validated)
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection
    r = config.reciprocal
    return float(_validate_reflection(r.hkl, r.keV, r.lattice_a))
```
- [ ] **Step 4:** run → PASS. mypy. **S0 is purely additive** — `run_theta` is defined + unit-tested but NOT yet wired into the geometry build (that's S2), so NO golden changes here and the full suite stays green. (The 5.8e-5 rad default-reflection shift + golden regeneration happen at S2, per the θ decision above.)
- [ ] **Step 5:** commit `feat(#16): run_theta resolver (simplified reflections use their own Bragg angle)`.

## Sub-task S1 — loaders RETURN the ResolutionContext (globals still set)

**Files:** `pipeline.py` (`_load_resolution`, `_lookup_and_load_kernel`).

- [ ] Change `_load_resolution(config, geometry=None) -> "fm.ResolutionContext"` to RETURN the context: capture `res = fm._load_analytic_resolution(config)` / `res = _lookup_and_load_kernel(...)` and return it. Change `_lookup_and_load_kernel(...) -> "fm.ResolutionContext"` to return `fm._load_default_kernel(...)`'s result.
- [ ] **Re-home the idempotent check** (`:773 if fm._loaded_kernel_path == target: return`): cache the last `(target_path → ResolutionContext)` in a module-level `dict` keyed by resolved path so a repeat load returns the cached context without a disk reload (preserves the test/REPL fast-path without reading a global). On cache hit, return the cached context.
- [ ] **Re-home the logging** (`:856` reads `fm._loaded_kernel_path`): log from the returned context's `loaded_kernel_path` (move the log after the load, or pass it through).
- [ ] Keep the global assignments inside the loaders for now (belt-and-suspenders until S5).
- [ ] **Verify:** new test asserting `_load_resolution(reciprocal, geometry)` returns a `ResolutionContext` whose fields equal what `build_forward_context()` reads from globals (parity). Full pipeline/forward/identification suite green; mypy. Commit `feat(#16): _load_resolution returns the ResolutionContext (idempotency re-homed)`.

## Sub-task S2 — `build_forward_context` takes explicit θ + resolution + q_hkl (break the round-trip)

**Files:** `forward_model.py` (`ForwardContext`, `build_forward_context`, `precompute_forward_static`), `pipeline.py` (3 call sites + the Hg_provider/generator q_hkl threading), `io/hdf5.py`.

- [ ] **Add `q_hkl` to `ForwardContext`:** `q_hkl: np.ndarray` (top-level field). Update `_context_from_globals` (set `q_hkl=q_hkl` from the global) and the new `build_forward_context` (compute it from `hkl`) to populate it.
- [ ] **New `build_forward_context` signature** (takes `hkl`, computes `q_hkl`):
```python
def build_forward_context(
    theta_run: float,
    resolution: "ResolutionContext",
    hkl: "tuple[int, int, int]",
    instrument: "InstrumentContext | None" = None,
) -> ForwardContext:
    instr = instrument if instrument is not None else build_instrument_context()
    geom = build_geometry_context(theta_run, instr)
    q = np.asarray(hkl, dtype=float)
    q_hkl = q / np.sqrt(float(q @ q))   # B_0=I form, matches Find_Hg:379-380 bit-for-bit
    return ForwardContext(instrument=instr, geometry=geom, resolution=resolution, q_hkl=q_hkl)
```
Verify `q / np.sqrt(q @ q)` is bit-identical to `Find_Hg`'s `np.asarray([h,k,l]) / np.sqrt(h*h+k*k+l*l)` (it is, for integer hkl). If any float discrepancy, MATCH Find_Hg's exact expression.
- [ ] **`precompute_forward_static`** reads `ctx.q_hkl` instead of the `q_hkl` global: `qs = ctx.instrument.Us @ Hg @ ctx.q_hkl`. (When `ctx is None` it still falls back to `_context_from_globals()` for this slice.)
- [ ] **The 3 call sites** (`pipeline.py:868`, `:1145` run_postprocess, `:2167` run_identification): build `ctx = fm.build_forward_context(run_theta(config), res, config.reciprocal.hkl, instr)` where `res` is the S1 `_load_resolution` return value (capture it now). **`run_postprocess` now passes `run_theta(config)`** — this *fixes* its previously-silent default-θ behavior (the default-reflection qi_field golden shifts by the 5.8e-5 rad correction → regenerate per the θ decision).
- [ ] **Thread `q_hkl` to the generators:** the 3 `_iter_identification_*` read `ctx.q_hkl` instead of `fm.q_hkl`.
- [ ] **`io/hdf5.py`:** `write_simulation_h5` builds `_ctx = ctx` (require ctx; drop the `_context_from_globals()` default once callers always pass it — they do); re-source `_fm.theta`→`_ctx.geometry.theta_0` (784/865), `_fm._loaded_kernel_path`→`_ctx.resolution.loaded_kernel_path` (718), `_fm._analytic_eval`→`_ctx.resolution.analytic_eval` (719). `write_identification_h5`: source 613/614 from `ctx.resolution`.
- [ ] **Verify (bit-identity gate):** dual-path — `build_forward_context(run_theta, res, q_hkl)` for the DEFAULT reflection vs the legacy globals path must be `np.array_equal` EXCEPT the known ~3e-4 θ shift (if S0 chose to regenerate). Run forward + identification + hdf5 + oblique gates. mypy. Commit `feat(#16): build_forward_context takes explicit theta/resolution/q_hkl; break the globals round-trip`.

## Sub-task S3 — retire `reflection_theta_if_oblique`

**Files:** `pipeline.py` (remove the 2 `with fm.reflection_theta_if_oblique(...)` wrappers at `:824`, `:2163`), `forward_model.py` (the CM stays defined until S5).

- [ ] With S2, geometry is built from `run_theta` directly, so the oblique CM that mutated `theta_0/Theta/rl/prob_z` is redundant for the threaded path. Remove the two `with` wrappers; the run bodies execute directly.
- [ ] **HAZARD — trace first:** before removing, confirm EVERY reader inside those `with` blocks (the population build, `Find_Hg`/`Find_Hg_from_population`, the generators) now gets the oblique `rl`/`Theta` via the explicit `ctx` (built from `run_theta`), NOT via the mutated globals. Grep the block bodies for bare `rl`/`Theta`/`globals()["rl"]` reads and for `Find_Hg*(... )` calls missing `ctx=ctx`.
- [ ] **Verify:** the oblique gate (`test_oblique_forward_contrast`, `test_identification_oblique_wiring`, `test_pipeline_writes_oblique_provenance`, `test_forward_reflection_theta`) RUN (use `-m "slow or not slow"`) + pass — this is the regression surface for removing the CM. mypy. Commit `refactor(#16): retire reflection_theta_if_oblique (geometry built from run_theta)`.

## Sub-task S4 — convert the remaining global readers

**Files:** `forward_model.py` (`_find_hg_from_population_numpy`), `tests/test_find_hg_kernel_parity.py` + `tests/test_forward_model_backlog.py`, `io/migrate.py`, `io/images.py`.

- [ ] **`_find_hg_from_population_numpy`:** add a `ctx` param (mirror `Find_Hg_from_population`); source `Theta`/`Us` from `ctx` (Us via `ctx.instrument.Us`, Theta via `ctx.geometry.Theta`) with globals fallback. Update the 2 test callers to pass an explicit `ForwardContext` (a default-reflection ctx built via `build_forward_context(run_theta(default_cfg), res, q_hkl)` or a small fixture). Both sides use the same Theta → parity holds.
- [ ] **`migrate.py`:** build a one-off `ForwardContext` for the IUCrJ-2024 Al `(-1,1,-1)`@17 keV default — load the kernel (`_load_resolution`/`_lookup_and_load_kernel`, which it needs anyway for `Find_Hg`), compute `run_theta` for that reflection, build `q_hkl`, compose the ctx. Re-source `_fm.theta`→`ctx.geometry.theta_0` (165/196), `_fm._loaded_kernel_path`→`ctx.resolution.loaded_kernel_path` (130), and pass `ctx` to its `_fm.Find_Hg(...)` call (98). Verify the migrate CLI tests pass.
- [ ] **`images.py`:** delete the dead `.npy` forward path (`save_image`, `save_images_parallel`, and the `forward_from_static`/`precompute_forward_static` calls). Confirm zero `src/` callers first (already verified). Delete/adjust any test that referenced them.
- [ ] **Verify:** `tests/test_find_hg_kernel_parity.py`, `tests/test_forward_model_backlog.py`, the migrate tests, and any images test green; mypy. Commit `refactor(#16): thread ctx into numpy oracle + migrate; delete dead images forward path`.

## Sub-task S5 — DELETE the globals + CM + `_context_from_globals` + fallbacks + the guard

**Files:** `forward_model.py`, `pipeline.py`, `io/hdf5.py`, `tests/test_forward_state_guard.py`.

- [ ] **Pre-deletion gate — clean grep (both forms):**
```
rg "fm\.(Hg|q_hkl|theta|theta_0|Theta|rl|prob_z|Resq_i|_analytic_eval|_loaded_kernel_path)\b" src/   # external readers → must be 0 (except intra-file none)
rg "_context_from_globals|reflection_theta_if_oblique" src/
# intra-module bare-name readers (the grep above misses these):
rg "\b(Theta|rl|prob_z|theta_0|theta|Resq_i|qi1_start|qi1_step|qi2_start|qi2_step|qi3_start|qi3_step|npoints1|npoints2|npoints3|qi_starts|qi_steps|_analytic_eval|Hg|q_hkl|_loaded_kernel_path)\b\s*[^=]" src/dfxm_geo/direct_space/forward_model.py
```
Resolve every remaining reader before deleting. Expected legitimate remaining readers: only `_context_from_globals`/`build_forward_context`'s OWN reads (deleted here) and the loaders' own `global ...; X = ...` assignments (removed here).
- [ ] **Loaders stop setting globals:** remove the `global ...` declarations + assignments in `_load_default_kernel`/`_load_analytic_resolution` (they return the `ResolutionContext` only). Remove `compute_Hg`'s global write (and the param if now unused) — production passes `Hg` explicitly; the loader no longer computes/stores it. Keep `q_hkl` computation only insofar as it flows into the returned context / the run's `build_forward_context` call.
- [ ] **Delete the globals + import-time geometry derivations:** `theta_0`, `theta`, `Theta`, `xl_start`, `xl_range`, `rl`, `prob_z` (and their import-time construction lines), `Resq_i`, `qi1_range`/`qi2_range`/`qi3_range`, `qi1_start/step`…`qi3_start/step`, `npoints1/2/3`, `qi_starts`, `qi_steps`, `_analytic_eval`, `Hg`, `q_hkl`, `_loaded_kernel_path`. **KEEP** instrument constants (`psize`, `zl_rms`, `Npixels`, `Nsub`, `NN1/2/3`, `Us`, `Ud`, `_flat_indices`, `yl_start`, `xl/yl/zl_steps`, `xl/yl/zl_range`).
- [ ] **Delete** `reflection_theta_if_oblique`, `_context_from_globals`, and the `ctx = ctx if ctx is not None else _context_from_globals()` fallbacks in `forward`/`forward_from_static`/`precompute_forward_static` (+ `hdf5.py:735`) — make `ctx` a REQUIRED positional in those functions.
- [ ] **Delete `_forward_state_guard` + `_GUARDED_GLOBALS` + the 3 decorators** (`run_simulation`/`run_identification`/`run_postprocess`) — obsolete now that no per-run mutable global exists. Delete `tests/test_forward_state_guard.py`'s guard-restore + self-load + `fm.q_hkl`-write tests (they assert behavior of deleted machinery); keep nothing that references the deleted globals. (The Task-2 `tests/test_postprocess_strain_source.py` file-recovery test stays — it exercises the surviving file-read path.)
- [ ] **`_resolve_postprocess_Hg`:** drop the priority-3 `fm.Hg` fallback (raise if no explicit Hg and no `/1.1/dfxm_geo/Hg`).
- [ ] **Verify — FINAL FULL GATE (kernel present):**
```
... -m pytest -q -m "slow or not slow"          # FULL suite
... -m mypy src/dfxm_geo/
rg "fm\.(theta|Theta|rl|Resq_i|_analytic_eval|_loaded_kernel_path|Hg|q_hkl)\b|_context_from_globals|reflection_theta_if_oblique|_forward_state_guard" src/   # → 0
```
Expected: full suite green (minus intentionally-deleted shim tests + the documented pre-existing `test_hdf5_writer_bit_equivalent_to_legacy_npy_golden`). Fd_find golden + all non-deleted bit-equiv/snapshot goldens reproduce. The grep returns zero. mypy 0 errors.
- [ ] **Commit** `refactor(#16): delete forward_model per-reflection globals + CM + state guard (fully threaded)`.

---

## Final review (after S5)
Dispatch a spec-review + code-quality-review subagent pair (parallel) over the whole Slice 5 range, then the parent plan's Task 13 (release notes; the simplified-θ behavior test is already added in S0). Then `superpowers:finishing-a-development-branch`.

## Hazards recap (from the 2026-06-03 mapping)
1. **Default-reflection θ shifts ~3e-4 rad** (`17.953/2`° → `_validate_reflection`≈8.9732°) at S0 — the one non-bit-identical change; audit goldens, decide regenerate-vs-pin, never silently regenerate.
2. **`migrate.py`** is the only reader with no natural ctx — it must build its own (S4).
3. **`run_postprocess`** flips default→run θ (S2) — intended, but check its qi_field golden.
4. **All 3 `build_forward_context` call sites + 2 CM sites** migrate together — a partial change leaves wrong arity.
5. **`_forward_state_guard` deletion** removes the cross-config safety net — only safe because S1-S4 leave zero per-run mutable globals; the final grep is the proof.
