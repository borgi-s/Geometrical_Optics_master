# ForwardContext refactor — replace the forward_model module-level mutable globals with an explicit threaded state object

**Date:** 2026-06-02
**Status:** Design proposed; pending Sina's review before plan-writing
**Release:** TBD. Slice 1 (#10 hardening) is bit-exact / no API break → patch or minor. Full migration (Item #16) is internal API churn with a back-compat shim → minor.
**Audit findings folded in:** #16 (globals → context) and #10 (`fm.Hg`/`fm.q_hkl` RuntimeError-at-a-distance + generator-body global-write race).

## Summary

`dfxm_geo.direct_space.forward_model` keeps the *entire* forward-model state as
module-level mutable globals: the geometry constants (`theta_0`, `theta`,
`Theta`, `Us`, `Ud`, the ray grid `rl`, `prob_z`, `_flat_indices`, the
`NN1/NN2/NN3` ray-grid dims), the loaded resolution kernel (`Resq_i`,
`qi{1,2,3}_start/step`, `npoints{1,2,3}`, `qi_starts/qi_steps`), the analytic
backend `_analytic_eval`, and the per-run strain field `Hg` / `q_hkl`. These
are read at call time by `precompute_forward_static`, `forward_from_static`
(and its fused numba kernel `_mc_lut_forward`), and `run_postprocess`, and are
*mutated in place* by `_load_default_kernel`, `_load_analytic_resolution`,
`reflection_theta_if_oblique`, and — the dangerous one — by the pipeline
generators and `_run_simulation_inner` writing `fm.Hg`/`fm.q_hkl`.

This single design choice is the root cause of three separate audit findings:

1. **`theta_0` staleness for non-default *simplified* reflections.** `theta_0`
   / `theta` / `Theta` are hardcoded at import to Al(1,1,1) @ 17 keV
   (8.98 deg). The oblique context manager rebuilds them only for
   `mode == "oblique"`. A *simplified* run at a different reflection still
   computes its forward geometry against the stale default `Theta`/`theta_0`
   even though the loaded kernel was bootstrapped at the correct Bragg angle.
2. **`reflection_theta_if_oblique` context-manager fragility.** It snapshots a
   7-tuple of globals (`theta_0, theta, Theta, xl_start, xl_range, rl,
   prob_z`), rebuilds them, and restores in a `finally`. Any new
   theta-dependent global silently drops out of the snapshot; an exception in
   the wrong place leaks mutated state; and the rebuild duplicates the
   import-time geometry construction.
3. **`fm.Hg`/`fm.q_hkl` RuntimeError-at-a-distance + generator-body
   global-write race (#10).** `run_postprocess` reads `fm.Hg` set by a prior
   `run_simulation` *in the same process* and raises if absent
   (pipeline.py ~:1092). `_iter_identification_multi` *writes* `fm.q_hkl`
   from inside the generator body (pipeline.py ~:1619). The planned
   persistent-worker pool (Phase 3) re-uses one warmed process across many
   configs/reflections; with these globals, config N's `fm.Hg`/`fm.q_hkl`/
   `fm.Theta`/`_analytic_eval` leak into config N+1, and concurrent
   generators racing on the same module dict is a data race.

The fix is to introduce an explicit, immutable `ForwardContext` (and a small
mutable per-run `ForwardState`) that owns this state and is **threaded** through
`precompute_forward_static` → `forward_from_static` → the writers and
`run_postprocess`, rather than read off the module. The numba kernel signature
is unchanged (it already takes everything as explicit args). A back-compat
shim keeps the module-level names alive (delegating to a process-global
"current context") so the migration can land in safe slices with bit-identity
verified at every step.

**Minimal first slice (`#10` hardening):** make `run_postprocess` and the three
identify generators *stop reading/writing module globals for `Hg`/`q_hkl`* —
pass them explicitly — and snapshot/restore the full forward state around each
`run_simulation`/`run_identification` so cross-config leakage is impossible.
This closes the persistent-worker-pool hazard without touching the
hot-path numba kernel. The full globals→context migration (Item #16) follows
in later slices behind the same shim.

## Motivation

### Why now

The DECIDED-NEXT work is a **persistent-worker pool** (Phase 3) that warms one
process (numba JIT, kernel load) and feeds it many configs/reflections to hit
the 100k-image ML-data throughput goal. That pool is exactly the workload the
current globals cannot survive:

- A worker that ran an *oblique* config leaves `theta_0`/`Theta`/`rl`/`prob_z`
  rebuilt — `reflection_theta_if_oblique` restores them only if the `with`
  block exits normally and only the 7 globals it knows about.
- A worker that ran an *analytic* config leaves `_analytic_eval` set; the next
  *MC* config on that worker silently takes the analytic branch unless
  `_load_resolution` clears it (it does today — but that is one more invariant
  the globals force every caller to remember).
- `fm.Hg`/`fm.q_hkl` carry the *previous* config's strain field; a
  `--postprocess-only` or a generator that forgets to set them reads stale data.
- Multi-reflection iteration (the Phase 3 companion) wants to hold *several*
  loaded kernels / thetas live at once and pick per-image — impossible when
  there is exactly one global `Resq_i`/`theta_0`.

### What the globals actually are

Three lifetimes are tangled into one namespace:

| Lifetime | Globals | Set by | Read by |
|---|---|---|---|
| **Instrument** (fixed per process) | `psize`, `zl_rms`, `Npixels`, `Nsub`, `NN1/2/3`, `Ud`, `Us`, `_flat_indices`, `yl_start`, `xl_steps`/etc. | import time | everywhere |
| **Geometry** (per reflection) | `theta_0`, `theta`, `Theta`, `xl_start`, `xl_range`, `rl`, `prob_z` | import (default) + `reflection_theta_if_oblique` | `precompute_forward_static` (via `Us`), `forward_from_static`, `Find_Hg*`, identify generators |
| **Resolution** (per reflection) | `Resq_i`, `qi*_start/step`, `npoints*`, `qi_starts/steps`, `_analytic_eval` | `_load_default_kernel` / `_load_analytic_resolution` | `forward_from_static` |
| **Strain** (per config/image) | `Hg`, `q_hkl` | `_run_simulation_inner` (writes `fm.Hg`/`fm.q_hkl`), generators (write `fm.q_hkl`) | `run_postprocess`, generators |

Item #16 is fundamentally about separating these four lifetimes so the
per-reflection and per-config state can be created, threaded, and discarded
without aliasing a process-wide mutable namespace.

### Read-set of the hot path (verified by code-read 2026-06-02)

- `precompute_forward_static(Hg)` reads: `Us`, `q_hkl`. Returns `base_qc`
  `(3, N)`, the loop-invariant `(Us @ Hg @ q_hkl).squeeze().T`.
- `forward_from_static(base_qc, phi, chi, TwoDeltaTheta, qi_return)` reads:
  `Resq_i`, `_analytic_eval`, `theta_0`, `NN1`, `NN2`, `Nsub`, `prob_z`,
  `_flat_indices`, `qi{1,2,3}_start/step`, `npoints{1,2,3}`. The numba kernel
  `_mc_lut_forward` takes ALL of these as explicit positional args already —
  it reads no globals.
- `run_postprocess` reads: `fm.Hg`, `fm.forward`, `fm.xl_start`, `fm.yl_start`,
  `fm.xl_steps`, `fm.yl_steps`, `fm.zl_steps`.
- Identify generators read: `fm.q_hkl`, `fm.theta`, `fm.psize`, `fm.zl_rms`,
  `fm.rl`, `fm.Z_shift`, `fm.Us`, `fm.Theta`, `fm.build_scan_grid`,
  `fm.Find_Hg_from_population` (indirectly). `_iter_identification_multi`
  additionally **writes** `fm.q_hkl` (pipeline.py ~:1619 — pure leftover; the
  value it writes is the one it just read).

## Design

### The `ForwardContext` shape

Two objects, split by mutability and lifetime. Both are plain
`@dataclass(frozen=True)` (or `frozen=True, slots=True`) — no behavior beyond
construction helpers — so they are trivially hashable/shareable and safe to
read concurrently from worker threads.

```python
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
    # (derived extents kept for postprocess plots)

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
    Resq_i: np.ndarray | None
    qi1_start: float; qi1_step: float
    qi2_start: float; qi2_step: float
    qi3_start: float; qi3_step: float
    npoints1: int | None; npoints2: int | None; npoints3: int | None
    analytic_eval: "AnalyticResolution | None"
    loaded_kernel_path: Path | None

@dataclass(frozen=True)
class ForwardContext:
    """Everything forward_from_static needs, bundled. Immutable + thread-safe."""
    instrument: InstrumentContext
    geometry: GeometryContext
    resolution: ResolutionContext
```

`Hg`/`q_hkl` are deliberately **not** in `ForwardContext`. They are per-config
*strain* — already threaded explicitly as `Hg` into `precompute_forward_static`
(returns `base_qc`) and as a `dfxm_geo` dict entry into each `ScanSpec`. The
only thing the refactor needs to do for strain is stop the
`fm.Hg = ...`/`fm.q_hkl = ...` *writes* and the `run_postprocess`/generator
*reads* off the module.

### What each context owns

- **InstrumentContext** is built once per process from the instrument config
  (`psize`, `zl_rms`, `Npixels`, `Nsub`). It owns the ray-grid dims and
  `flat_indices` scatter map (currently module-level `_flat_indices`). It never
  changes within a process, so the persistent-worker pool builds it once and
  shares it.
- **GeometryContext** is built per reflection from `(theta_0)` (simplified:
  derived from `(hkl, keV)`; oblique: the solver `theta_validated`). This is the
  object that *replaces* `reflection_theta_if_oblique`: instead of mutating
  module globals inside a `with` block and restoring, you construct a
  `GeometryContext` at the run's Bragg angle and pass it down. **This also
  fixes the simplified-reflection staleness**: the geometry is always built at
  the run's actual theta, not the import-time default, with no special-casing
  of `mode == "oblique"`.
- **ResolutionContext** is built per reflection by the (renamed) kernel/analytic
  loaders, which now *return* a `ResolutionContext` instead of mutating module
  globals. Exactly one of `Resq_i` / `analytic_eval` is non-`None`.

### How it threads through

```
SimulationConfig
   └─ build_instrument_context(config)        -> InstrumentContext   (once/process)
   └─ build_geometry_context(theta_0, instr)  -> GeometryContext     (per reflection)
   └─ load_resolution_context(config, geom)   -> ResolutionContext   (per reflection)
        ctx = ForwardContext(instr, geom, res)

run_simulation(config, out):
    ctx = build_forward_context(config)         # composes the three above
    Hg_base, q_hkl = Hg_provider(0.0, ctx)      # Find_Hg* take ctx (uses ctx.geometry.rl/Theta)
    write_simulation_h5(..., ctx=ctx, Hg=Hg_base, q_hkl=q_hkl, Hg_provider=...)
        # writer: base_qc = precompute_forward_static(Hg, ctx)
        #         _compute_frame: forward_from_static(base_qc, ctx, phi, chi, dt)

run_postprocess(out, config):
    ctx = build_forward_context(config)         # rebuilt, not read off module
    _, qi = forward(Hg_from_h5_or_arg, ctx, qi_return=True)   # Hg passed in, not fm.Hg

run_identification(config, out):
    ctx = build_forward_context(config)
    for spec in _iter_identification_X(config, ctx):   # generators take ctx, read ctx.geometry.*
        ...
```

The signatures become:

```python
def precompute_forward_static(Hg: np.ndarray, ctx: ForwardContext) -> np.ndarray: ...
def forward_from_static(base_qc, ctx: ForwardContext, phi=0, chi=0,
                        TwoDeltaTheta=0, qi_return=False) -> ...: ...
def forward(Hg, ctx: ForwardContext, phi=0, chi=0,
            TwoDeltaTheta=0, qi_return=False) -> ...: ...
```

### Keeping the numba kernel signature happy

`_mc_lut_forward` already takes everything as explicit scalar/array args and
reads no globals — **its signature does not change**. The only edit at the
kernel call site in `forward_from_static` is sourcing the args from `ctx`:

```python
_mc_lut_forward(
    base_qc, ang0, ang1, ang2, Theta,        # Theta from ctx.geometry (or dt-shifted)
    ctx.geometry.prob_z, ctx.instrument.flat_indices,
    ctx.resolution.Resq_i.reshape(-1), im_1.reshape(-1),
    ctx.resolution.qi1_start, ctx.resolution.qi1_step,
    ... ctx.resolution.npoints1, ..., np2*np3, np3,
)
```

numba caches on the *types* of the args, all unchanged (`float64`, C-contig
`ndarray`, `int`), so the JIT cache key and compiled artifact are identical —
**no recompile, no perf change**. The `Theta` passed is either
`ctx.geometry.Theta` (the common case) or, when `TwoDeltaTheta != 0`, the
`dt`-shifted rotation built from `ctx.geometry.theta_0` exactly as today.

### Backward-compatible shim path

The module-level names cannot vanish in one commit: `io/hdf5.py`,
`io/migrate.py`, tests, and the legacy root shims read `fm.Hg`,
`fm._loaded_kernel_path`, `fm.theta`, `fm.rl`, etc. The shim is a process-global
"current context" plus module-level descriptors that delegate to it:

- Introduce `fm._CURRENT_CTX: ForwardContext | None` and `fm._CURRENT_STRAIN:
  tuple[Hg, q_hkl] | None`, set by `build_forward_context` / the run functions
  for the duration of a run (and cleared after).
- Re-express the module-level names as **module `__getattr__`** (PEP 562)
  delegating to `_CURRENT_CTX` so `fm.theta`, `fm.Theta`, `fm.rl`, `fm.prob_z`,
  `fm.Resq_i`, `fm._analytic_eval`, `fm.Hg`, `fm.q_hkl`, `fm._loaded_kernel_path`
  keep reading the current context. Writes (`fm.Hg = ...`) still hit the module
  dict; the slice-1 work removes the *writes* in pipeline.py, so by the time the
  shim is the only writer there are none in production.
- Keep `psize`, `zl_rms`, `Npixels`, `Nsub`, `NN1/2/3`, `Us`, `Ud`,
  `_flat_indices`, `yl_start`, `xl_steps` as real module constants (the
  `InstrumentContext` is *constructed from* them) until a later slice — they are
  reflection-independent, so nothing depends on threading them for correctness.

This lets every consumer migrate independently and lets us delete the shim
attribute-by-attribute once all readers are converted.

### `reflection_theta_if_oblique` retirement

The context manager is replaced by `build_geometry_context(theta_0, instrument)`
which constructs the geometry at the run's theta directly. During the migration
the CM is kept as a thin wrapper that builds a `GeometryContext` and installs it
as `_CURRENT_CTX.geometry` for the `with` body (so the shim's `fm.theta`/`fm.rl`
reads still see the oblique geometry), restoring the prior context on exit. Once
all callers thread `ctx`, the CM and its 7-tuple snapshot are deleted. The
simplified-reflection staleness disappears the moment geometry is built from the
run's theta unconditionally — there is no longer a "default vs oblique" branch.

## Staged migration

Each slice is independently shippable, keeps the smoke tests + mypy green, and
verifies bit-identity before moving on.

### Slice 1 — `#10` hardening (minimal first slice; bit-exact, no hot-path edit)

The smallest change that removes the persistent-worker-pool hazard, *without*
introducing `ForwardContext` yet:

1. **Stop the generator-body global write.** Delete `fm.q_hkl = q_hkl` at
   `_iter_identification_multi` (pipeline.py ~:1619). It writes back the value
   it just read — pure no-op leftover. (TDD: a test that runs two generators
   "concurrently" — interleaved `next()` — and asserts neither mutates
   `fm.q_hkl`.)
2. **Pass `Hg`/`q_hkl` explicitly into `run_postprocess`.** Add an optional
   `Hg`/`q_hkl` parameter (default: read from the HDF5 `/1.1/dfxm_geo` group,
   which the writer already persists when `write_strain_provenance=True`).
   This removes the `fm.Hg is None` RuntimeError-at-a-distance: `--postprocess-only`
   recovers strain from the file instead of from a stale global. Keep the
   `fm.Hg` fallback behind the shim for one release.
3. **Snapshot/restore the full forward state around each run.** Wrap
   `run_simulation`/`run_identification` bodies in a `_forward_state_guard()`
   context manager that snapshots *every* mutable global
   (`theta_0, theta, Theta, xl_start, xl_range, rl, prob_z, Resq_i,
   qi*_start/step, npoints*, qi_starts/steps, _analytic_eval, Hg, q_hkl,
   _loaded_kernel_path`) on entry and restores on exit. This makes
   cross-config leakage on a persistent worker impossible *today*, and it
   subsumes `reflection_theta_if_oblique`'s restore responsibility (the CM
   stays for the geometry *rebuild*, the guard owns the *restore*).
4. **Verify:** Fd_find golden + all `*forward*`/`*reflection*`/`*oblique*`/
   `*identification*` tests stay green; mypy clean. No numba recompile.

### Slice 2 — introduce the contexts as *return values* (still shimmed)

5. Add `InstrumentContext`/`GeometryContext`/`ResolutionContext`/`ForwardContext`
   dataclasses + `build_*_context` constructors. Make `_load_default_kernel` /
   `_load_analytic_resolution` *also* return a `ResolutionContext` (keep setting
   the globals too, behind the shim). No caller changes yet — pure additive.
6. **Verify:** new unit tests assert a freshly built `ForwardContext` reproduces
   the current module globals field-by-field (the parity oracle for the whole
   refactor). mypy clean.

### Slice 3 — thread `ctx` through the hot path

7. Add `ctx: ForwardContext` params to `precompute_forward_static`,
   `forward_from_static`, `forward`. During this slice, default `ctx=None` →
   fall back to a `_context_from_globals()` builder so existing callers (tests,
   `io/hdf5.py`) keep working; new callers pass `ctx`.
8. Convert `io/hdf5.py` `_compute_frame` + the writers to carry and pass `ctx`
   (the `_FrameArgs` tuple gains nothing — `ctx` rides alongside as a writer-level
   arg, not per frame, since it is shared read-only).
9. **Verify bit-identity:** re-run the Fd_find golden + forward goldens; assert
   `forward(Hg, ctx)` == `forward(Hg)` (globals path) elementwise (`==`, not
   `allclose` — it must be bit-exact since the float ops are unchanged). Confirm
   no numba recompile via `_mc_lut_forward.signatures` length before/after.

### Slice 4 — thread `ctx` through pipeline + postprocess + generators

10. `run_simulation`/`run_identification` build `ctx` and pass it to the
    writers, `Hg_provider`, and the generators. Generators read `ctx.geometry.*`
    instead of `fm.*`. `run_postprocess` builds `ctx` and passes `Hg` explicitly.
11. Make `Find_Hg` / `Find_Hg_from_population` take `ctx` (they read `Theta`/`rl`).
12. **Verify:** full identify + forward HDF5 pipeline tests green; the v2.3.0
    oblique provenance + contrast tests green (this is the real regression
    surface for the geometry change). mypy clean.

### Slice 5 — delete the shim

13. Remove the `__getattr__` delegation and the now-unused module globals
    (`theta_0`, `theta`, `Theta`, `rl`, `prob_z`, `Resq_i`, `qi*`, `npoints*`,
    `_analytic_eval`, `Hg`, `q_hkl`), `reflection_theta_if_oblique`, and the
    `_context_from_globals` fallback. Update `io/migrate.py` and any remaining
    readers to source provenance from `ctx`/the HDF5 file.
14. **Verify:** grep for `fm.Hg`/`fm.q_hkl`/`fm.theta`/`fm.Resq_i`/
    `reflection_theta_if_oblique` returns only the deleted-test diffs; full
    targeted suite + mypy green.

## How bit-identity is verified at each stage

- **The numerical contract is bit-exact**, not approximate: the float operations
  in `precompute_forward_static`/`forward_from_static`/`_mc_lut_forward` are
  unchanged — only where the *inputs* come from changes. So every verification
  uses elementwise `==` (or `np.array_equal`), not `allclose`.
- **Slice gate (every slice):** the `tests/data/golden/Fd_find_smoke.npy` golden
  (the safety net under the whole cleanup) must reproduce. Run the targeted
  `test_forward_*`, `test_*reflection*`, `test_*oblique*`,
  `test_pipeline_identification*` files with the venv pytest. Never regenerate or
  touch any `*bit_equiv*` / `*snapshot*` / `*pickle_era*` golden — if a slice
  would alter one, the slice is wrong (the ops are unchanged), revert and
  diagnose.
- **Slice 2 parity oracle:** assert `build_forward_context(config)` reproduces the
  legacy globals field-by-field (`Theta == fm.Theta`, `rl == fm.rl`,
  `Resq_i == fm.Resq_i`, the qi grid scalars `==`, etc.).
- **Slice 3 dual-path assertion:** for a representative `(Hg, phi, chi)`, assert
  `forward(Hg, ctx=...)` and `forward(Hg)` (globals path, pre-`ctx`) produce
  `np.array_equal` images; and `len(_mc_lut_forward.signatures)` is unchanged
  (no recompile → identical compiled kernel).
- **Slice 4 geometry regression:** the v2.3.0 oblique contrast/provenance tests
  (`test_oblique_forward_contrast`, `test_pipeline_writes_oblique_provenance`,
  `test_identification_oblique_wiring`) are the watch-list — they exercise the
  one place where behavior *should* be unchanged but the code path moved.

## Out of scope

- Building the persistent-worker pool itself (Phase 3) — this refactor is its
  prerequisite, not the pool.
- Multi-reflection *iteration* (holding several kernels live) — the
  `ResolutionContext` makes it trivial later, but the API for selecting per-image
  is a separate spec.
- Promoting `InstrumentContext` fields (psize/Npixels/Nsub) to config-threaded —
  they are reflection-independent, so they can stay module constants until a
  later cleanup with no correctness impact.
- Any change to the `.cif`/lattice work (v3.0.0).

## Risks & mitigations

- **Risk: a missed global reader breaks at runtime, not in tests.** Mitigation:
  the `__getattr__` shim keeps all module names working until Slice 5; Slice 5's
  deletion is gated on a repo-wide grep showing zero non-test readers.
- **Risk: numba silently recompiles (perf regression).** Mitigation: assert
  `len(_mc_lut_forward.signatures)` constant across the dual-path test in
  Slice 3; arg dtypes are unchanged by construction.
- **Risk: oblique geometry behavior shifts when `reflection_theta_if_oblique`
  is replaced.** Mitigation: build `GeometryContext` from the identical
  expressions the CM uses (same `_R_y(theta)`, same `xl_start = yl_start /
  tan(2θ) / 3`, same `rl` mgrid), and gate on the v2.3.0 oblique tests with
  `==`. The *simplified* path's theta-staleness fix is an intentional
  behavior change for non-default simplified reflections — call it out in the
  release notes and add a test asserting a non-(-1,1,-1) simplified reflection
  now uses its own Bragg angle (currently it uses the stale default).
