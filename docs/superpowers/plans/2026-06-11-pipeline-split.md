# pipeline.py Split (Refactor Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 2760-line `src/dfxm_geo/pipeline.py` into `config.py` (dataclasses + TOML), `orchestrator.py` (run logic), and `cli.py` (argparse entry points), with `pipeline.py` remaining a FULL facade so every existing import path, pyproject entry point, and external script keeps working unchanged.

**Architecture:** Three additive extraction stages, each ending with the full suite green. `dfxm_geo.pipeline` becomes a pure re-export shell (`from dfxm_geo.config import *` etc. with explicit `__all__`-free name re-exports). Monkeypatch strings in tests are retargeted in the SAME commit as the code move they depend on, following one principle: **patch the module where the bare-name call executes** (cli callers → `dfxm_geo.cli.X`; orchestrator-internal callees → `dfxm_geo.orchestrator.X`). pyproject entry points stay `dfxm_geo.pipeline:cli_main[_identify]` (facade) — zero pyproject change, zero conda-forge implications.

**Tech Stack:** Python 3.12, pytest, mypy (0-error policy), pre-commit (ruff + ruff-format).

**Execution context:**
- Worktree: `C:\Users\borgi\Documents\GM-reworked\wt-identify-fanout`, branch `refactor/pipeline-split` (off main `8a9e8fe`).
- Python: ALWAYS `C:\Users\borgi\Documents\GM-reworked\wt-identify-fanout\.venv\Scripts\python.exe` (editable install of this tree; 122 MB kernel npz present so kernel-gated tests run).
- Branch-start gates (verified on main `8a9e8fe`): full suite **851 passed / 1 skipped / 22 deselected / 1 xfailed, 0 failures**; mypy 0 errors / 36 files. Every task must end at this failure set (= empty) with the passed count unchanged.
- All line numbers below refer to `src/dfxm_geo/pipeline.py` at `8a9e8fe` (recon 2026-06-11). After Task 1 the later ranges shift — locate by NAME, the ranges are orientation only.

**Non-negotiable invariants (from recon):**
1. `dfxm_geo.pipeline` re-exports EVERYTHING it exports today — including underscore names (`_load_resolution`, `_lookup_and_load_kernel`, `_KERNEL_CTX_CACHE`, `_CANONICAL_AXES`, `_build_scan_frames`, `_dataclass_to_toml_str`, `_run_identification_*`, `_iter_identification_*`, `_passes_invisibility`, `_draw_dislocation`, `_scan_frames_args`, `_context_for_run`, `_resolution_for_run`, `_build_scan_frames_at_z`, `_iterate_simulation_frames`, `_identification_config_to_toml_str`) and the module alias `fm`. 62 test files + `scripts/scaling_sweep.py`, `scripts/render_rocking_gif.py`, `forward_model.py:1029` (runtime `_CANONICAL_AXES` import) depend on it.
2. `config.py` must import NOTHING from `orchestrator.py`/`cli.py` (preserves today's no-runtime-cycle property; `forward_model.py` and `io/hdf5.py` back-import from pipeline only under TYPE_CHECKING + the one deferred `_CANONICAL_AXES` function-local import).
3. `_KERNEL_CTX_CACHE` stays a module-global dict in `orchestrator.py` that is never REASSIGNED by library code (only `.get`/`[...]=`/`.clear()`), so the facade's re-exported binding stays the same object (conftest.py's `.clear()` keeps working without retarget).
4. `pipeline.fm` must be the `dfxm_geo.direct_space.forward_model` MODULE object (test_pipeline.py patches `dfxm_geo.pipeline.fm.forward` etc. — module-attr mutation, works through any alias as long as the facade exports `fm`).
5. pyproject `[project.scripts]` untouched.

---

### Task 1: Extract `config.py` (dataclasses, loaders, serializers, run_theta)

**Files:**
- Create: `src/dfxm_geo/config.py`
- Modify: `src/dfxm_geo/pipeline.py` (remove moved code; add facade import block at top)
- Test: existing suite is the test (move-only task; no new tests)

- [ ] **Step 1: Create `src/dfxm_geo/config.py`** with a module docstring ("Config dataclasses, TOML loaders and serializers for dfxm_geo — extracted from pipeline.py (refactor gate, 2026-06-11). Import via `dfxm_geo.pipeline` (the stable facade) or directly.") and MOVE these from pipeline.py, verbatim, in this order (names + 8a9e8fe line ranges):
  - `AxisScanConfig` (74-104), `_CANONICAL_AXES` (106), `_AXIS_TO_LABEL` (114-129), `ScanConfig` (130-180)
  - `CenteredCrystalConfig` (181-227), `WallCrystalConfig` (228-250), `RandomDislocationsConfig` (251-275), `_CRYSTAL_MODE_NAMES` (276), `CrystalConfig` (279-363)
  - `IOConfig` (364-378), `PostprocessConfig` (379-398), `ReciprocalConfig` (399-467), `GeometryConfig` (468-490)
  - `_build_geometry_config` (491-545), `_parse_reflections_tables` (546-594)
  - `SimulationConfig` (595-637)
  - `IdentificationCrystalConfig` (638-651), `IdentificationNoiseConfig` (652-665), `IdentificationMonteCarloConfig` (666-679), `IdentificationZScanConfig` (680-708), `IdentificationConfig` (709-767), `load_identification_config` (768-833)
  - `run_theta` (904-924) — pure config-shaped dispatch (SimulationConfig|IdentificationConfig → Bragg angle); its `_validate_reflection` / geometry imports come with it
  - `ScanFrames` (1568-1582) — frozen data carrier, no behavior
  - `_dataclass_to_toml_str` (1245-1371) and `_identification_config_to_toml_str` (1737-1837) — the TOML serializers; they read only dataclass fields
  Bring along exactly the imports each moved item needs (numpy, dataclasses, tomllib, Path, typing, the `crystal.oblique` / `crystal.reflections` / `reciprocal_space.kernel` imports used by `_build_geometry_config` / `_parse_reflections_tables` / `ReciprocalConfig.__post_init__` / `run_theta`, `SAMPLE_REMOUNT_OPTIONS` for WallCrystalConfig). `config.py` MUST NOT import `dfxm_geo.io.hdf5`, `forward_model`, or anything that will later live in orchestrator/cli (invariant 2). If a moved item turns out to need one of those (check before moving!), STOP and report — that item stays in pipeline.py for now and is flagged.

- [ ] **Step 2: Shrink pipeline.py.** Delete the moved code; at the import section add:

```python
# --- refactor gate (2026-06-11): config layer extracted to dfxm_geo.config.
# pipeline remains the stable facade: import config names from here as before.
from dfxm_geo.config import (  # noqa: F401
    _AXIS_TO_LABEL,
    _CANONICAL_AXES,
    _CRYSTAL_MODE_NAMES,
    AxisScanConfig,
    CenteredCrystalConfig,
    CrystalConfig,
    GeometryConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    IdentificationZScanConfig,
    IOConfig,
    PostprocessConfig,
    RandomDislocationsConfig,
    ReciprocalConfig,
    ScanConfig,
    ScanFrames,
    SimulationConfig,
    WallCrystalConfig,
    _build_geometry_config,
    _dataclass_to_toml_str,
    _identification_config_to_toml_str,
    _parse_reflections_tables,
    load_identification_config,
    run_theta,
)
```

(ruff will re-sort; explicit names — NOT `import *` — so mypy and ruff can verify every re-export. The remaining pipeline.py code keeps using these names unchanged: they resolve via this import.)

- [ ] **Step 3: Verify the three external coupling points still work**
  - `forward_model.py:1029` (`from dfxm_geo.pipeline import _CANONICAL_AXES`) — covered by the facade import; no edit needed.
  - `io/hdf5.py:31` TYPE_CHECKING `ScanFrames` — covered; no edit.
  - `forward_model.py:29` TYPE_CHECKING config imports — covered; no edit.

Run: `& "...\.venv\Scripts\python.exe" -c "import dfxm_geo.pipeline as p; import dfxm_geo.config; print(p.SimulationConfig is dfxm_geo.config.SimulationConfig, p.run_theta is dfxm_geo.config.run_theta)"`
Expected: `True True`

- [ ] **Step 4: Full suite + mypy**

Run: `& "...\.venv\Scripts\python.exe" -m pytest -q` → 851 passed / 0 failures (identical counts).
Run: `& "...\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/config.py src/dfxm_geo/pipeline.py
git commit -m "refactor: extract config layer to dfxm_geo.config (pipeline stays the facade)"
```

---

### Task 2: Extract `cli.py` (+ retarget the 8 CLI patch strings)

**Files:**
- Create: `src/dfxm_geo/cli.py`
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline.py` (8 patch strings)

- [ ] **Step 1: Create `src/dfxm_geo/cli.py`** — move `cli_main` (1524-1567) and `cli_main_identify` (2709-2760) verbatim, plus their imports (`argparse`, `sys`, `Path`, `replace` from dataclasses). The functions call `run_simulation`, `run_postprocess`, `run_identification`, `load_identification_config`, `SimulationConfig` — import these at module top of cli.py:

```python
from dfxm_geo.config import IdentificationConfig, SimulationConfig, load_identification_config
from dfxm_geo.pipeline import run_identification, run_postprocess, run_simulation  # noqa: F401
```

WAIT — importing from `dfxm_geo.pipeline` here creates a cycle once pipeline re-imports cli (Step 2). Instead: at Task 2 time the orchestrators still LIVE in pipeline.py, and at Task 3 they move to orchestrator.py. To avoid touching cli.py twice AND avoid the cycle, cli.py imports the orchestration functions LAZILY inside each function body:

```python
def cli_main(argv: list[str] | None = None) -> int:
    ...
    # Lazy import: pipeline (the facade) re-imports this module, so a
    # module-top import would be circular. Resolving at call time also
    # keeps `monkeypatch.setattr("dfxm_geo.cli.run_simulation", ...)`...
```

NO — lazy `from dfxm_geo.pipeline import run_simulation` inside the function would make patches of `dfxm_geo.cli.run_simulation` ineffective (the name is re-imported fresh each call) and patches of `dfxm_geo.pipeline.run_simulation` effective. DECISION (keep it simple and patch-friendly): cli.py does `import dfxm_geo.pipeline as _pipeline` LAZILY inside each function and calls `_pipeline.run_simulation(...)`, `_pipeline.run_postprocess(...)`, `_pipeline.run_identification(...)`, `_pipeline.load_identification_config(...)`. Consequence: the EXISTING patch strings `dfxm_geo.pipeline.run_simulation` / `run_postprocess` in `tests/test_pipeline.py:411-508` KEEP WORKING UNCHANGED (they patch the facade attribute, and cli resolves through the facade at call time). So:
  - cli.py top-level imports: `argparse`, `sys`, `from dataclasses import replace`, `from pathlib import Path` ONLY.
  - Each function body starts with `import dfxm_geo.pipeline as _pipeline` and uses `_pipeline.X` for: `run_simulation`, `run_postprocess`, `run_identification`, `load_identification_config`, `SimulationConfig` (cli_main builds/loads configs — check the actual body at 1524-1567 for every pipeline name it touches and route ALL of them through `_pipeline.`).
  - **No test patch strings change in this task after all** — the `_pipeline.`-routing preserves them. (The task title's "retarget" is then a no-op; verify instead.)

- [ ] **Step 2: pipeline.py facade for the CLI**: delete the moved functions, add

```python
from dfxm_geo.cli import cli_main, cli_main_identify  # noqa: F401  (refactor gate)
```

This is safe (no cycle): cli.py has no module-top import of pipeline.

- [ ] **Step 3: Verify entry-point + patch resolution**

Run: `& "...\.venv\Scripts\python.exe" -c "from dfxm_geo.pipeline import cli_main, cli_main_identify; print('entry points ok')"`
Run: `& "...\.venv\Scripts\python.exe" -m pytest tests/test_pipeline.py -q` → all pass (the 8 `dfxm_geo.pipeline.run_simulation`/`run_postprocess` patch tests at lines 411-508 prove the lazy facade routing works).

- [ ] **Step 4: Full suite + mypy** (same bar: 851/0, mypy 0).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/cli.py src/dfxm_geo/pipeline.py
git commit -m "refactor: extract CLI entry points to dfxm_geo.cli (facade routing keeps patch targets stable)"
```

---

### Task 3: Extract `orchestrator.py` + retarget the orchestrator-internal patch strings (ONE commit)

> **Patch-surface rule (from Task 2's quality review):** the CLI's patch
> surface is PERMANENTLY the facade — `dfxm_geo.pipeline.run_simulation`
> / `run_postprocess` / `run_identification` (cli.py resolves names off the
> facade at call time). Any NEW CLI test must patch the facade, never
> `dfxm_geo.orchestrator.*` (an orchestrator patch would silently no-op for
> CLI calls). Orchestrator-INTERNAL callees follow the table below instead.

**Files:**
- Create: `src/dfxm_geo/orchestrator.py`
- Modify: `src/dfxm_geo/pipeline.py` (becomes pure facade)
- Modify (patch-string retargets, same commit): `tests/conftest.py` is NOT changed (`.clear()` mutates the shared object); `tests/test_pipeline.py`, `tests/test_pipeline_identification.py`, `tests/test_identification_oblique_wiring.py`, `tests/test_pipeline_multi_reflection.py`, `tests/test_identification_multi_per_dis.py`, `tests/test_identify_dedup.py`, `tests/test_identify_threads_eta_to_analytic.py`

- [ ] **Step 1: Move EVERYTHING remaining** (except the two facade import blocks) from pipeline.py into `src/dfxm_geo/orchestrator.py`: the `fm` alias import, `find_hg_scene` import, the io.hdf5 writer imports, `_KERNEL_CTX_CACHE`, `_lookup_and_load_kernel`, `_load_resolution`, `_resolution_for_run`, `_context_for_run`, `run_simulation`, `_run_simulation_inner`, `_resolve_postprocess_Hg`, `run_postprocess`, `_build_scan_frames`, `_build_scan_frames_at_z`, `_iterate_simulation_frames`, `_scan_frames_args`, `_positioners_for_scan_frames`, `_identify_title`, `_q_unit`, `_passes_invisibility`, `_iter_identification_single`, `_run_identification_single`, `_ALL_111_PLANES`, `_draw_dislocation`, `_build_dislocation_sample_entry`, `_iter_identification_multi`, `_run_identification_multi`, `_maybe_apply_poisson_noise`, `_iter_identification_zscan`, `_run_identification_zscan`, `_identify_geometry_attrs`, `_dispatch_identification`, `run_identification` — plus all their remaining imports. orchestrator.py imports config names DIRECTLY from `dfxm_geo.config` (NOT from pipeline — invariant 2 inverted: orchestrator may import config, never the reverse). Module docstring: "Forward + identification run orchestration — extracted from pipeline.py (refactor gate, 2026-06-11)."

- [ ] **Step 2: pipeline.py becomes the pure facade** — final content shape (exact, modulo the alphabetized config block from Task 1):

```python
"""dfxm_geo.pipeline — stable facade over the split modules (refactor gate, 2026-06-11).

The 2700-line pipeline module was split into:
  dfxm_geo.config        — config dataclasses, TOML loaders/serializers, run_theta
  dfxm_geo.orchestrator  — run_simulation / run_identification + helpers
  dfxm_geo.cli           — dfxm-forward / dfxm-identify argparse entry points

This module re-exports the complete historical surface (public AND underscore
names, plus the `fm` module alias) so every existing import path, pyproject
entry point, fanout template, and notebook keeps working. New code should
import from the specific module; tests that monkeypatch orchestration
internals patch `dfxm_geo.orchestrator.*` (the module where the bare-name
call sites execute) — patching the facade binding does NOT reach them.
"""

from dfxm_geo.config import (...)          # the Task-1 block, unchanged
from dfxm_geo.cli import cli_main, cli_main_identify  # noqa: F401
from dfxm_geo.orchestrator import (  # noqa: F401
    _ALL_111_PLANES,
    _KERNEL_CTX_CACHE,
    _build_dislocation_sample_entry,
    _build_scan_frames,
    _build_scan_frames_at_z,
    _context_for_run,
    _dispatch_identification,
    _draw_dislocation,
    _identify_geometry_attrs,
    _identify_title,
    _iter_identification_multi,
    _iter_identification_single,
    _iter_identification_zscan,
    _iterate_simulation_frames,
    _load_resolution,
    _lookup_and_load_kernel,
    _maybe_apply_poisson_noise,
    _passes_invisibility,
    _positioners_for_scan_frames,
    _q_unit,
    _resolution_for_run,
    _resolve_postprocess_Hg,
    _run_identification_multi,
    _run_identification_single,
    _run_identification_zscan,
    _run_simulation_inner,
    _scan_frames_args,
    find_hg_scene,
    fm,
    run_identification,
    run_postprocess,
    run_simulation,
)
```

(`fm` re-exported as the module object — invariant 4. `find_hg_scene` and the writers re-exported because tests patched/imported them via pipeline.)

- [ ] **Step 3: Retarget the orchestrator-internal patch strings** (these patch a callee whose callers now execute in orchestrator.py; the facade binding no longer reaches them). Exact retarget table — change ONLY the module path in the patch target, nothing else:

| Old target | New target | Files (8a9e8fe lines) |
|---|---|---|
| `dfxm_geo.pipeline._load_resolution` / `setattr(pipeline, "_load_resolution", ...)` | `dfxm_geo.orchestrator._load_resolution` | test_pipeline.py:313,392; test_pipeline_identification.py:404,418,433,825,840; test_identification_oblique_wiring.py:156,217 |
| `pipeline.find_hg_scene` | `orchestrator.find_hg_scene` | test_identification_multi_per_dis.py:98; test_identify_dedup.py:78 |
| `p.write_simulation_h5` (pipeline ns) | orchestrator ns | test_pipeline_multi_reflection.py:90 |
| `pipeline._run_identification_single` | `orchestrator._run_identification_single` | test_identification_oblique_wiring.py:157 |
| runner_map `dfxm_geo.pipeline._run_identification_{single,multi,zscan}` | `dfxm_geo.orchestrator._run_identification_*` | test_identify_threads_eta_to_analytic.py:116-118 |
| `setattr(p, "_KERNEL_CTX_CACHE", {})` | orchestrator ns | test_pipeline_multi_reflection.py:66,188 |

NOT retargeted (verify, don't change): conftest.py:32 `.clear()` (same-object mutation through the facade — works); `dfxm_geo.pipeline.fm.forward` / `fm.yl_start` / `fm.xl_steps` / `fm.yl_steps` / `fm.zl_steps` in test_pipeline.py:304,318-321 (module-attr mutation through the alias — works); `dfxm_geo.pipeline.run_simulation` / `run_postprocess` CLI patches (Task 2's `_pipeline.`-routing — works). Each retargeted test file needs its import line adjusted if it does `from dfxm_geo import pipeline as p` style aliasing — import `dfxm_geo.orchestrator` alongside, patch through that.

- [ ] **Step 4: Sweep for stragglers**

Run: `& "...\.venv\Scripts\python.exe" -m pytest -q` → 851 passed / 0 failures. ANY failure here is almost certainly a missed patch retarget or a missed re-export — diagnose by the failing test's patch target.
Grep: `grep -rn "dfxm_geo\.pipeline\._" tests/ src/ scripts/` — every remaining hit must be either an IMPORT (fine — facade) or one of the deliberately-unretargeted patches above; no orchestrator-internal patch may remain on the pipeline namespace.
Run: `& "...\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/` → 0 errors.
Run: `& "...\.venv\Scripts\python.exe" -m pytest -m slow tests/test_identification_oblique_e2e.py tests/test_oblique_remount_wall.py -q` → 4 passed (real-compute paths through the new module layout).

- [ ] **Step 5: Commit (move + retargets together)**

```bash
git add src/dfxm_geo/orchestrator.py src/dfxm_geo/pipeline.py tests/
git commit -m "refactor: extract orchestration to dfxm_geo.orchestrator; pipeline is now a pure facade (test patch targets follow the call sites)"
```

---

### Task 4: Line-budget check, docs, and final gates

**Files:**
- Modify: `docs/architecture.md` (module-layout section)
- No src changes expected

- [ ] **Step 1: Sanity-check the split sizes**

Run: `Get-Content src/dfxm_geo/{pipeline,config,orchestrator,cli}.py | Measure-Object -Line` (per file). Expected ballpark: pipeline ≈ 60-90 (facade only), config ≈ 1000-1100, orchestrator ≈ 1500-1700, cli ≈ 110-130. If pipeline.py still holds ANY logic beyond the three import blocks + docstring, something was missed — go back.

- [ ] **Step 2: Update `docs/architecture.md`** — find the section describing pipeline.py (grep "pipeline.py") and rewrite it to describe the four-module layout (facade + config/orchestrator/cli, the import-direction rule config ← orchestrator ← cli, and the test-patching convention: orchestration internals are patched on `dfxm_geo.orchestrator`). Keep it to one short subsection in the file's existing voice.

- [ ] **Step 3: Final gates (the full M2-style battery)**

Run: full suite (`-q`) → 851/0; kernel-less check is CI's job but run `& "...\.venv\Scripts\python.exe" -m pytest tests/test_pipeline.py tests/test_pipeline_identification.py tests/test_fanout.py -q` as a focused smoke; mypy → 0; `pre_commit run --all-files` → clean; slow oblique subset → green (already done in Task 3 Step 4, re-run if any commit landed since).

- [ ] **Step 4: Commit docs**

```bash
git add docs/architecture.md
git commit -m "docs: architecture - pipeline facade + config/orchestrator/cli split"
```

Merge/push is NOT part of this plan (house rule: Sina confirms; though pushes were broadly consented 2026-06-11, present the branch with the final review first).

---

## Self-review notes (already applied)

- Spec coverage: roadmap §1 item 1 = "split pipeline.py (config.py / orchestrator.py / cli.py)" — Tasks 1-3; the forward_model-globals half of the refactor gate shipped as v2.5.0 (#16), out of scope here.
- The Task-2 DECISION paragraph supersedes the first (struck-through-style) import sketch above it — implementer reads the whole step; the lazy `_pipeline.`-routing is the binding choice.
- Type consistency: facade name lists in Tasks 1-3 are the single source; Task 3 Step 2's config block must match Task 1 Step 2's exactly.
- Biggest residual risk: a patch target the recon missed. Task 3 Step 4's grep sweep + identical-failure-set rule is the net.
