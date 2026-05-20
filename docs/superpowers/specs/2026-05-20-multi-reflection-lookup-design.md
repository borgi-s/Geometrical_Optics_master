---
title: Multi-reflection kernel lookup in forward + identify (sub-project D)
status: approved
date: 2026-05-20
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - docs/superpowers/specs/2026-05-20-bootstrap-multi-reflection-design.md (sub-project A — produces the per-reflection kernel files D loads)
  - src/dfxm_geo/direct_space/forward_model.py (current pkl_fn / _load_default_kernel)
  - src/dfxm_geo/pipeline.py (current _ensure_kernel_loaded; SimulationConfig / IdentificationConfig dataclasses)
  - src/dfxm_geo/reciprocal_space/kernel.py (generate_kernel — bundled metadata target)
  - src/dfxm_geo/io/hdf5.py (provenance recording at line 414)
  - configs/default.toml, configs/identification_*.toml (TOML schema migration)
---

# Multi-reflection kernel lookup in forward + identify (sub-project D)

## Purpose

After sub-project A (merged 2026-05-20, `dbc3ecc`), `dfxm-bootstrap`
produces per-reflection kernel files named
`Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz`. Today, `dfxm-forward`
and `dfxm-identify` are still pinned to a single kernel via the
`pkl_fn` module-level constant in `src/dfxm_geo/direct_space/forward_model.py`
(currently `"Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz"`). This blocks
all multi-reflection workflows: there is no way to ask forward "use
the (2,0,0) kernel" without re-editing module-level state.

Sub-project D removes that constraint. Forward and identify read
`(hkl, keV)` from the `[reciprocal]` TOML block, look up the matching
kernel by globbing `pkl_files/`, verify the bundled metadata matches
the request, and run the simulation using that kernel.

Scope: per-run reflection lookup (one reflection per simulation
process). Multi-reflection-per-process (e.g., iterating over
reflections in one HDF5 file with `/N.M` BLISS-scan groups per
reflection) is a v1.3.0+ feature, explicitly out of scope here.

Targets the v1.2.0 release (alongside sub-projects B, C, E).

## Decisions (Q1–Q6)

Captured via brainstorming (`superpowers:brainstorming`) on 2026-05-20.

- **Q1 — Per-run reflection lookup (one reflection per process).**
  Multi-reflection-per-process is deferred to v1.3.0+. Each scan picks
  a single reflection; the kernel for that (h, k, l) is loaded by name.
- **Q2 — Reuse the `[reciprocal]` block** for `hkl` + `keV` in forward
  and identify TOML configs. Single source of truth: the TOML that
  bootstrapped the kernel IS the TOML that consumes it. Identify
  configs (which currently lack any `[reciprocal]` block) gain a
  minimal `hkl` + `keV` block.
- **Q3 — Multi-match: newest-mtime wins, WARN to stderr** listing all
  candidates with the chosen one marked. Sorting by `mtime` (not
  filename) is robust to clock skew on shared filesystems.
- **Q4 — Zero-match: hard `FileNotFoundError`** with explicit
  `dfxm-bootstrap --config <yourconfig.toml>` instruction. No auto-
  bootstrap; one tool, one job.
- **Q5 — Remove `pkl_fn` entirely.** No module-import auto-load. CLI
  entrypoints explicitly call `_lookup_and_load_kernel(hkl, keV)`
  from the config. REPL users now need one extra explicit load call;
  acceptable per `[[dfxm-no-backcompat-constraint]]`.
- **Q6 — Bundle `hkl` + `keV` in npz AND verify on load.** Bootstrap
  writes both into the npz scalar-meta dict; load verifies they match
  the lookup request. Catches manual file-renaming and copy-paste
  errors during host-to-host kernel synchronization.

## Approach: "Surgical replacement" (a)

Single PR. Direct swap of the `pkl_fn`-based machinery for a
config-driven lookup-and-load pipeline. Approach (b) "conservative
deprecation with both mechanisms coexisting" contradicts Q5 and was
rejected; approach (c) "new `lookup.py` module" is premature
abstraction for one small function and was rejected.

## Architecture

### Files modified

- **`src/dfxm_geo/direct_space/forward_model.py`** — main work:
  - Add `_lookup_kernel_path(hkl, keV, pkl_fpath) -> Path`.
  - Extend `_load_default_kernel(pkl_path, *, expected_hkl=None, expected_keV=None, compute_Hg=True)` with verification kwargs.
  - Add `_loaded_kernel_path: Path | None = None` module attribute (set by `_load_default_kernel` on successful load — drives downstream provenance).
  - Delete `pkl_fn` constant.
  - Delete module-import auto-load block (lines 423-428 of current main).
- **`src/dfxm_geo/pipeline.py`** —
  - Add `ReciprocalConfig` dataclass: `hkl: tuple[int, int, int]`, `keV: float`.
  - Field-add `reciprocal: ReciprocalConfig` to `SimulationConfig` AND `IdentificationConfig`. No default — required.
  - Replace `_ensure_kernel_loaded()` with `_lookup_and_load_kernel(hkl, keV)`.
  - Update `run_simulation` and `run_identification` to call the new variant with `config.reciprocal.hkl, config.reciprocal.keV`.
  - Update the effective-config print at line 295-303 to print the actually-loaded kernel path (not `fm.pkl_fpath{fm.pkl_fn}`).
- **`src/dfxm_geo/reciprocal_space/kernel.py`** —
  - In `generate_kernel`, add `hkl` + `keV` to the bundled `kernel_meta` dict. Both received as additional kwargs from `cli_main`.
  - `cli_main` passes the validated `hkl_tuple` and `keV_for_filename` through.
- **`src/dfxm_geo/io/hdf5.py`** —
  - Line 414: read `fm._loaded_kernel_path` (the actually-loaded basename) instead of `Path(_fm.pkl_fpath) / _fm.pkl_fn`.
  - Line 211: already records `Path(kernel_npz).name` — verify it receives the same actually-loaded path.
- **`src/dfxm_geo/io/migrate.py`** —
  - Line 154: same swap — `fm._loaded_kernel_path` not `fm.pkl_fn`.
- **`configs/identification_single.toml`**, **`identification_multi.toml`**, **`identification_zscan.toml`** —
  - Add minimal block at top:
    ```toml
    [reciprocal]
    hkl = [-1, 1, -1]
    keV = 17.0
    ```
  Preserves the implicit Al 111 @ 17 keV behavior they had pre-D.
- **`configs/default.toml`** — unchanged structurally (already has `[reciprocal]` from A). The MC params there are bootstrap-only; forward reads only `hkl` + `keV`.

### New API surface

```python
def _lookup_kernel_path(
    hkl: tuple[int, int, int],
    keV: float,
    pkl_fpath: Path | str,
) -> Path:
    """Find the newest kernel npz on disk matching the requested (hkl, keV).

    Globs `<pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz`, sorts by
    mtime descending, returns the newest. Emits a stderr WARN listing
    all matches when more than one exists. Raises FileNotFoundError
    with a `dfxm-bootstrap` instruction on zero matches.
    """

def _load_default_kernel(
    pkl_path: str | Path | None = None,
    *,
    expected_hkl: tuple[int, int, int] | None = None,
    expected_keV: float | None = None,
    compute_Hg: bool = True,
) -> None:
    """Load kernel npz into module-level state.

    If `expected_hkl` / `expected_keV` are given, verifies the npz's
    bundled metadata matches; raises ValueError on mismatch. Raises
    KeyError if the npz lacks `hkl`/`keV` metadata (pre-D legacy file).
    Sets `_loaded_kernel_path` on success.
    """

# pipeline.py
def _lookup_and_load_kernel(
    hkl: tuple[int, int, int],
    keV: float,
) -> None:
    """Compose: lookup → load with verification. Replaces _ensure_kernel_loaded."""
```

### Removed API

- `forward_model.pkl_fn` (module-level constant)
- Module-import auto-load at `forward_model.py:423-428`
- `pipeline._ensure_kernel_loaded()` (renamed/rewired to `_lookup_and_load_kernel`)

### Key invariant

The kernel basename consumed by a `run_simulation` / `run_identification`
call is determined 100% by `(hkl, keV)` from that call's config + the
contents of `<pkl_fpath>/`. The actually-loaded basename — exposed via
`fm._loaded_kernel_path` — is the canonical "which kernel did we use"
for HDF5 provenance and migration tooling.

## Data flow

```
User invokes: dfxm-forward --config myconfig.toml
    │
    ▼
SimulationConfig.from_toml(path)
    ├── parses [crystal], [scan], [io], [postprocess]  (existing)
    └── parses [reciprocal]  →  ReciprocalConfig(hkl=(-1,1,-1), keV=17.0)  (NEW)
        ├── [reciprocal] block absent → ValueError("missing [reciprocal] block...")
        ├── hkl OR keV missing → ValueError("missing `hkl` in [reciprocal]...")
        └── invokes _validate_reflection(hkl, keV, a=4.0495e-10) early
            (catches typos / Bragg-unsatisfiable BEFORE the kernel lookup)
    │
    ▼
run_simulation(config, output_dir):
    │
    ├── _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV):
    │     │
    │     ├── path = _lookup_kernel_path(hkl, keV, fm.pkl_fpath)
    │     │     ├── glob: <pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz
    │     │     ├── zero matches → FileNotFoundError with bootstrap instruction
    │     │     ├── multi matches → WARN stderr "found N kernels...; using <newest>"
    │     │     └── return newest-mtime Path
    │     │
    │     └── fm._load_default_kernel(path, expected_hkl=hkl, expected_keV=keV):
    │           ├── np.load(path); read bundled scalar meta
    │           ├── meta["hkl"] absent → KeyError (pre-D legacy)
    │           ├── meta["hkl"] != expected_hkl → ValueError (mismatch)
    │           ├── meta["keV"] != expected_keV → ValueError (mismatch)
    │           ├── populate fm.Resq_i, fm.qi*_range, fm.npoints*, fm.theta, fm.D, fm.d1, ...
    │           └── set fm._loaded_kernel_path = path  (drives provenance)
    │
    ├── existing forward simulation: Find_Hg, forward(), save_images_parallel, write HDF5
    │     └── HDF5 writer uses fm._loaded_kernel_path.name (NOT fm.pkl_fn) for provenance
    │
    └── return summary dict
```

**Timing change:** kernel load moves from **module-import time** (auto-load
at `forward_model.py:423-428`, deleted) to **`run_simulation` /
`run_identification` call time** (post-config-parse, pre-`Find_Hg`).
Importing `forward_model` does zero I/O now.

**Identify mirror:** `run_identification(config)` runs the identical
`_lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)`
step before the existing identify logic.

**Provenance trail:** every HDF5 file written records, under the
`provenance/kernel/pkl_fn` dataset (existing schema slot from the
`io/hdf5.py:209-211` code), the actually-loaded basename. No
synthesis from module constants.

## Lookup + metadata details

### Glob pattern

```
<pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz
```

`{h}`, `{k}`, `{l}` are str-cast ints; `{keV:g}` drops trailing zeros
(`17.0` → `"17"`, `17.5` → `"17.5"`). `*` matches `_build_kernel_filename`'s
`YYYYmmdd_HHMM` date stamp from sub-project A.

### Tie-breaking

```python
matches = sorted(
    pkl_fpath.glob(f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
```

Newest `mtime` wins. Sorting by `mtime` (not filename lexicographic) is
robust to clock skew between hosts on shared filesystems (rare but real
at facility scale).

### Multi-match WARN format (stderr)

```
warning: found 3 kernels matching hkl=(-1,1,-1) keV=17 in <pkl_fpath>:
  Resq_i_h-1_k1_l-1_17keV_20260521_0930.npz  (newest, will use)
  Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz
  Resq_i_h-1_k1_l-1_17keV_20260518_1142.npz
```

### Bundled metadata (new in D — modifies `generate_kernel.kernel_meta`)

Existing 20 scalars stay; two new keys:

```python
kernel_meta = {
    ...,  # 20 existing keys (Nrays, npoints*, theta, D, d1, beam, beamstop)
    "hkl": np.array(hkl, dtype=np.int64),  # NEW — shape (3,)
    "keV": np.float64(keV),                # NEW — scalar
}
```

### Verification on load

```python
data = np.load(pkl_path)
if expected_hkl is not None:
    if "hkl" not in data.files:
        raise KeyError(
            f"kernel at {pkl_path} lacks `hkl` metadata — pre-sub-project-D bootstrap.\n"
            f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
        )
    meta_hkl = tuple(int(x) for x in data["hkl"])
    if meta_hkl != tuple(expected_hkl):
        raise ValueError(
            f"kernel at {pkl_path} has hkl={meta_hkl} but lookup requested hkl={tuple(expected_hkl)} — "
            f"file may have been manually renamed or copied wrong."
        )
if expected_keV is not None:
    if "keV" not in data.files:
        raise KeyError(
            f"kernel at {pkl_path} lacks `keV` metadata — pre-sub-project-D bootstrap.\n"
            f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
        )
    meta_keV = float(data["keV"])
    if meta_keV != expected_keV:  # exact match — both come from same TOML
        raise ValueError(
            f"kernel at {pkl_path} has keV={meta_keV} but lookup requested keV={expected_keV} — "
            f"file may have been manually renamed or copied wrong."
        )
```

theta is NOT independently verified — it's a function of `hkl + keV + a`,
so matching `hkl` and `keV` (with the same hardcoded Al `a`) implies
matching theta modulo float precision. Adding a 3rd check is paranoid.

## Error handling

All hard errors raise; CLI entrypoints (`dfxm-forward`, `dfxm-identify`)
catch + exit 1 + stderr (no raw traceback).

### TOML parsing — `ReciprocalConfig.from_toml(data)`

- `[reciprocal]` block missing → `ValueError("missing [reciprocal] block — forward/identify require explicit hkl + keV; see configs/default.toml.")`
- `hkl` key missing → `ValueError("missing `hkl` in [reciprocal] — required for kernel lookup.")`
- `keV` key missing → `ValueError("missing `keV` in [reciprocal] — required for kernel lookup.")`
- `_validate_reflection(hkl, keV, a=4.0495e-10)` fails → propagates A's `ValueError`s verbatim.

### Lookup — `_lookup_kernel_path(hkl, keV, pkl_fpath)`

- Zero matches:
  ```
  FileNotFoundError(
      "no kernel found for hkl={hkl} at {keV} keV in {pkl_fpath}/.\n"
      "Run: dfxm-bootstrap --config <yourconfig.toml>\n"
      "(produces Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_<date>.npz, "
      "~50 s wall-clock at default Nrays=1e8)"
  )
  ```

### Load — `_load_default_kernel(path, *, expected_hkl, expected_keV)`

- Legacy npz lacking `hkl`/`keV` keys → `KeyError("kernel at {path} lacks `hkl` metadata — pre-sub-project-D bootstrap. Re-run: dfxm-bootstrap --config <yourconfig.toml>")`
- Metadata `hkl` mismatch → `ValueError("kernel at {path} has hkl={meta_hkl} but lookup requested hkl={expected_hkl} — file may have been manually renamed or copied wrong.")`
- Metadata `keV` mismatch → `ValueError("kernel at {path} has keV={meta_keV} but lookup requested keV={expected_keV} — file may have been manually renamed or copied wrong.")`
- Other `np.load` errors (corrupted file, permission denied) → bubble up Python-native errors (existing behavior).

### Warnings (stderr, not errors)

- Multi-match on lookup → as specced above.

### CLI surfacing pattern

```python
try:
    config = SimulationConfig.from_toml(args.config)
    run_simulation(config, args.output)
except (ValueError, FileNotFoundError, KeyError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
```

## Testing

Splitting tests per concern to unlock parallel implementer dispatch in
the implementation plan (per `[[feedback-parallelize-subagents]]`).

### `tests/test_kernel_lookup.py` (NEW)

Pure-function tests on `_lookup_kernel_path` and the verification
branches of `_load_default_kernel`. Uses a synthetic-kernel fixture
helper (~10 LOC) that writes tiny npz files via `np.savez` matching
the lookup pattern.

- `TestLookupKernelPath`
  - happy: 1 file matching → returns that path.
  - multi-match: 3 files all matching → returns newest by `mtime`; stderr WARN lists all 3 with `"(newest, will use)"` marker.
  - zero match → `FileNotFoundError` with bootstrap-instruction text.
  - glob isolation: `hkl=(2,0,0)` doesn't match `Resq_i_h-1_k1_l-1_17keV_*.npz` files in the same dir.
  - keV format invariance: `17` (int) and `17.0` (float) both match the same `*_17keV_*` pattern.

- `TestLoadDefaultKernelVerification`
  - happy: bundled `hkl` + `keV` match expected → loads cleanly; `fm._loaded_kernel_path` is set.
  - hkl mismatch → `ValueError`.
  - keV mismatch → `ValueError`.
  - legacy npz lacking `hkl`/`keV` keys → `KeyError`.

### `tests/test_kernel_cli.py` (EXTENDED)

- Extend `TestGenerateKernelOutputPath`: assert the produced npz has
  `hkl` (ndarray, shape (3,)) and `keV` (scalar) bundled in scalar
  params for a known `(hkl, keV)` invocation.

### `tests/test_pipeline_multi_reflection.py` (NEW)

Integration tests via `pipeline.run_simulation` and (a public form of)
`run_identification`. Monkey-patches `fm.forward`, `fm.Find_Hg`, and
the HDF5 writer; performs REAL lookup against `tmp_path`-staged kernel
files. Verifies wiring + provenance + error paths end-to-end.

- `test_forward_happy_path_with_explicit_hkl` — TOML with `[reciprocal] hkl=[2,0,0] keV=17`, kernel npz staged in `tmp_path` matching the pattern; `run_simulation` returns 0; `fm._loaded_kernel_path` is the staged path; HDF5 provenance dataset records that basename.
- `test_forward_missing_reciprocal_block` → exit 1 + stderr `"missing [reciprocal] block"`.
- `test_forward_missing_hkl_or_keV` (parametrized over hkl-only / keV-only) → exit 1 each with the corresponding stderr message.
- `test_forward_lookup_miss` → no matching kernel staged → exit 1 + stderr `"no kernel found for hkl=..."` with bootstrap instructions.
- `test_forward_multi_match_warns_and_uses_newest` — 2 timestamped files staged; stderr WARN; the newer one is in `fm._loaded_kernel_path`.
- `test_forward_metadata_mismatch` — file with deliberately-wrong bundled `hkl` (filename says one thing, metadata another) → exit 1 + clean stderr ValueError.
- `test_identify_happy_path_with_explicit_hkl` — mirror for identify.
- `test_identify_missing_reciprocal_block` → exit 1.

### `tests/test_pipeline.py` (UPDATED)

- Existing `_ensure_kernel_loaded` references → `_lookup_and_load_kernel`.
- Existing `test_dfxm_forward_with_sample_remount_S2_runs` (currently `@pytest.mark.slow`) gets `[reciprocal]` added to its config fixture.

### `tests/test_kernel_format.py` (UPDATED)

- Grep for `pkl_fn` references; update or remove.

### Coverage budget

- ~20–25 new tests.
- ~+300–350 LOC test code (dominated by `test_pipeline_multi_reflection.py`).
- All pure-function tests < 0.1 s each; integration tests < 1 s (monkey-patched MC). Total added wall-clock < 30 s.

### Removed

- Module-import auto-load smoke test (if any) at the bottom of `forward_model.py` — `fm.Resq_i` is now `None` on bare import; downstream tests that previously asserted populated state need to either explicitly load or skip.

## Out of scope (deferred)

- **Multi-reflection per process** (a v1.3.0+ feature). One process, one reflection. Sweeping multiple reflections in one HDF5 file with `/N.M` BLISS groups per reflection is a separate design.
- **Auto-bootstrap on lookup miss** (the rejected Q4 option (b) or the rejected (c) opt-in flag). Operationally: user runs `dfxm-bootstrap` explicitly, then runs `dfxm-forward`. Two tools, two responsibilities.
- **Materials other than Al** (`a` is still hardcoded `4.0495e-10` in `_validate_reflection`). Sub-project A deferred this; D doesn't unblock it. A future sub-project (or simple constant extraction) can address it when concrete need arises.

## Operational follow-ups

After this sub-project lands:

- Run `dfxm-bootstrap --config configs/default.toml` once on laptop + once on the cluster to produce new `Resq_i_*.npz` files with the new `hkl` + `keV` metadata bundled. Legacy pre-D npz files (currently on disk from A's 2026-05-20 bootstrap) will fail load verification with the clear `KeyError` until they're regenerated. The error message tells the user exactly what to do — no surprise.
- After regen, no `pkl_fn` update needed: lookup happens dynamically.

## Version target

Lands as part of the v1.2.0 release alongside sub-projects B, C, E.
PyPI publish remains held until v1.2.0 ships.
