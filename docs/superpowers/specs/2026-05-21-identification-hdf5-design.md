# Sub-project E — Identification mode → HDF5, master + per-scan layout (v1.2.0)

**Date:** 2026-05-21
**Status:** Design proposed; pending Sina's review before plan-writing
**Release:** v1.2.0 (breaking format change for forward mode + new format for identify mode)
**Supersedes:** Single-file HDF5 layout introduced in v1.1.0

## Summary

Sub-project E is the final piece for v1.2.0. It does two coupled things:

1. **Identification mode** (all three sub-modes: `single`, `multi`, `z-scan`) writes HDF5 instead of the current `.npy + manifest.csv + PNG-preview` directory trees.
2. **Forward mode** is retrofitted from the v1.1.0 single-file layout to the same new layout.

The new layout is a faithful mirror of an ESRF BLISS dataset on disk: one **master** HDF5 file at the top of the output directory, with per-scan **LIMA-style** detector files in `scan0001/`, `scan0002/`, … subdirectories. The master holds positioners, sample identity, and provenance; the per-scan files hold only pixel data. The two are linked via HDF5 `ExternalLink`.

All three identification sub-modes additionally extend to consume the `[scan.<axis>]` schema introduced in sub-project B+C — each "config" inside an identification run can now produce a phi/chi grid of frames (rocking / rolling / mosa), not just a single image.

## Motivation

- **Beamline fidelity.** Real ESRF BLISS datasets put each scan in its own directory (e.g. `Al1050_dislocations_wireRocking_G154_15layer_w1_50ticks.h5` master + `scan0001/pco_ff_0000.h5` + `scan0002/pco_ff_0000.h5` …). The v1.1.0 single-file layout, while BLISS-compliant inside one file, doesn't match the experiment-directory shape that beamline tooling expects on disk. Simulated data should be drop-in interoperable with real data — same loaders, same processing flows.
- **Identification mode parity.** Identification's current `.npy + manifest.csv` output is incompatible with `silx`, `darfix`, and `darling`. Moving to HDF5 closes that gap.
- **Scan-vocabulary unification.** Identification sub-modes today each have their own ad-hoc handling of "do we vary phi?" / "do we scan chi?". The B+C `[scan.<axis>]` schema is the canonical answer for forward mode; reusing it for identification eliminates a parallel vocabulary.
- **One release for the format break.** Coupling the forward retrofit with E means v1.2.0 ships one consistent format. Otherwise v1.2.0 would have identify-on-Approach-2 / forward-on-v1.1.0-layout — two different shapes for two modes of the same tool.

## Scope

### In scope (v1.2.0)

1. New on-disk layout: master + per-scan-dirs with `ExternalLink`.
2. Forward mode (`dfxm-forward`) retrofitted to the new layout. Output is still `out_dir/dfxm_geo.h5` (the master), but pixel data lives in `out_dir/scan0001/` and `out_dir/scan0002/`.
3. Identification mode (`dfxm-identify`, all three sub-modes) writes the new layout. Master filename `dfxm_identify.h5`.
4. All three identification sub-modes consume `[scan.phi]` / `[scan.chi]` from the B+C schema. Each `/N.1` inside an identification master holds an `(N_frames, H, W)` stack, where N_frames depends on which axes are scanned.
5. `IdentificationZScanConfig` simplified: drops `phi_range_deg, phi_steps, chi_range_deg, chi_steps` (duplicates of `[scan.<axis>]`); keeps `z_offsets_um, include_secondary, secondary_rng_offset`.
6. `IdentificationMonteCarloConfig` gains opt-in `render_per_dislocation: bool = False`. When set, each multi-mode scan dir also writes per-dislocation detector files for unambiguous instance labels.
7. Migration tools: `dfxm-migrate-output` (existing, `.npy` → HDF5) updated to emit the new layout; new `dfxm-migrate-h5` for converting v1.1.0 single-file HDF5 → v1.2.0 master+per-scan layout.
8. Drop identification sidecars: `manifest.csv` and `images/*.png` previews. Info is fully captured in `/N.1/sample/`.
9. Documentation: full rewrite of `docs/output-format.md`; new `docs/release-notes-1.2.0.md`; `CLAUDE.md` working-notes updates.

### Out of scope (deferred to v1.3.0+)

- Wiring `[scan.two_dtheta]` and `[scan.z]` into the forward or identification kernels. Both axes remain guarded with `ValueError` in `run_simulation` and (newly) `run_identification`.
- Folding z-scan mode into `single` + `[scan.z]` (a future consolidation; flagged for v1.3.0+).
- Pixel-level instance segmentation masks for multi mode (only point labels via `position_um` for now; the opt-in per-dislocation rendering covers the next step up).
- A `render_per_dislocation` analogue for z-scan's primary/secondary pair.
- Extending `_SLIP_SYSTEM_111` to cover all 12 FCC slip systems (already deferred from B+C).
- Other B+C-deferred follow-ups (see `[[session-handoff-2026-05-21]]`).
- Sub-project F (default config flip to "simple") → v2.0.0.

## On-disk layout

```
out_dir/
  dfxm_geo.h5                                    ← master (forward) OR
  dfxm_identify.h5                               ← master (identify)
    /dfxm_geo/                                   ← global provenance (one per master)
      version, git_sha, git_dirty, hostname,
      python_version, numpy_version, generated_at,
      cli, config_toml
      kernel/{pkl_fn, sha256, ...}               ← bundled kernel metadata
    /1.1/                                        ← BLISS scan entry, metadata-only in master
      @scan_mode, @scanned_axes, @crystal_mode (forward) | @identify_mode (identify)
      title, start_time, end_time
      sample/                                    ← per-scan identity (see per-mode section)
      instrument/
        dfxm_sim_detector/
          @NX_class = "NXdetector"
          data → ExternalLink('scan0001/dfxm_sim_detector_0000.h5',
                              '/entry_0000/dfxm_sim_detector/image')
        positioners/
          @NX_class = "NXcollection"
          phi, chi  ← (N_frames,) for scanned, scalar for fixed; degrees with @units
      measurement/                               ← BLISS soft-links
        @NX_class = "NXcollection"
        dfxm_sim_detector, phi, chi   ← SoftLink to instrument/...
      dfxm_geo/                                  ← per-scan sim-specific
        Hg, q_hkl, theta, psize, zl_rms
        analysis/...                             ← present only after forward postprocess
    /2.1/                                        ← next scan; same structure
    /3.1/, /4.1/, ...                            ← only identification masters have >2
  scan0001/
    dfxm_sim_detector_0000.h5                    ← LIMA-style detector file, pixels only
      / @NX_class="NXroot", @creator="dfxm-geo", @default="entry_0000"
      /entry_0000/
        @NX_class = "NXentry"
        title, start_time, end_time
        dfxm_sim_detector/
          @NX_class = "NXdetector"
          image  (N_frames, H, W) float64
            chunks=(1, H, W), gzip-4 + shuffle
            @interpretation = "image"
        plot/  @NX_class="NXdata", @signal="image"
          image → SoftLink(/entry_0000/dfxm_sim_detector/image)
        measurement/ → SoftLink(/entry_0000/dfxm_sim_detector/image)
  scan0002/dfxm_sim_detector_0000.h5
  ...
  scanNNNN/dfxm_sim_detector_0000.h5
```

**External link targets are stored as relative paths** so the entire `out_dir/` tree is portable. Moving or copying the directory preserves all links.

## Per-`/N.1` attrs

| Attr | Forward | Identify | Example values |
|---|:---:|:---:|---|
| `scan_mode`     | ✓ | ✓ | `"single"`, `"rocking"`, `"rolling"`, `"mosa"` |
| `scanned_axes`  | ✓ | ✓ | `[]`, `["phi"]`, `["chi"]`, `["phi", "chi"]` |
| `crystal_mode`  | ✓ | — | `"centered"`, `"wall"`, `"random_dislocations"` |
| `identify_mode` | — | ✓ | `"single"`, `"multi"`, `"z-scan"` |

`scan_mode` and `scanned_axes` are derived from `config.scan.derived_mode_name()` and `config.scan.scanned_axes()` (already in the B+C schema). `crystal_mode` is already written by sub-project C. `identify_mode` is new in E.

## Frame ordering

fscan2d convention preserved from v1.1.0: `k = chi_idx * phi_steps + phi_idx` (phi inner, chi outer). N_frames per `/N.1` depends on the scan mode:

| scan_mode | scanned_axes | N_frames |
|---|---|---|
| `single`  | `[]`               | 1 |
| `rocking` | `["phi"]`          | `phi_steps` |
| `rolling` | `["chi"]`          | `chi_steps` |
| `mosa`    | `["phi", "chi"]`   | `phi_steps × chi_steps` |

Positioners follow the same rule: scanned axes get `(N_frames,)` 1-D arrays; fixed axes get scalars equal to `axis.value`.

## Per-mode `sample/` layouts

### Forward mode

Unchanged from v1.1.0 schema; just moves from old single-file to new master:

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocations" | "simulated, perfect crystal"
  dis                 scalar float64    (wall mode only; sentinel/absent for centered + random)
  ndis                scalar int64
  sample_remount      scalar string     (wall mode only; "N/A" otherwise)
```

(Note: the "sentinel `-1.0` for non-wall modes" issue flagged in `[[session-handoff-2026-05-21]]` is **not** fixed here — that's a separate v1.3.0 cleanup.)

### Identify single mode

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (single)"
  slip_plane_normal   (3,) int32          e.g. [1, 1, 1]
  burgers             (3,) int32          e.g. [1, 0, 1]  (scaled by √2 to integers)
  rotation_deg        scalar float64      e.g. 45.0
```

### Identify multi mode

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (multi)"
  dislocations/
    @NX_class = "NXcollection"
    0/
      @NX_class = "NXsample"
      slip_plane_normal, burgers, rotation_deg
      position_um       (3,) float64      lab-coord (x, y, z=0)
    1/
      @NX_class = "NXsample"
      slip_plane_normal, burgers, rotation_deg
      position_um       (3,) float64
```

Dislocation index (0, 1) is draw order, not a spatial ordering. The trainer correlates pixel features to which dislocation via `position_um` projected through the detector geometry.

### Identify z-scan mode

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (z-scan)"
  z_offset_um         scalar float64
  primary/
    @NX_class = "NXsample"
    slip_plane_normal, burgers, rotation_deg
    position_um       (3,) float64       (always [0, 0, 0] currently)
  secondary/                              ← only present if include_secondary=True
    @NX_class = "NXsample"
    slip_plane_normal, burgers, rotation_deg
    position_um       (3,) float64       (always [0, 0, 0] currently — pos_std_um=0 in current draw)
```

## Multi-mode opt-in: `render_per_dislocation`

New field in `IdentificationMonteCarloConfig`:

```toml
[multi]
n_samples = 10000
pos_std_um = 5.0
render_per_dislocation = false      # default; set true for opt-in
n_png_previews = 0                  # legacy field; safe to ignore or remove
```

**Default (`false`):** each scan dir has one detector file as shown above. Total compute = N_samples × 1 forward() call.

**`true`:** each scan dir contains THREE detector files:

```
scan0001/
  dfxm_sim_detector_0000.h5            ← both dislocations summed (canonical detector)
  dfxm_sim_detector_dis0_0000.h5       ← first dislocation only
  dfxm_sim_detector_dis1_0000.h5       ← second dislocation only
```

The master's `/N.1/instrument/` carries three `NXdetector` groups (`dfxm_sim_detector`, `dfxm_sim_detector_dis0`, `dfxm_sim_detector_dis1`), each with its own `ExternalLink`. Total compute = N_samples × 3 forward() calls. Real beamlines do exactly this pattern (multiple detectors per scan), so it stays BLISS-faithful.

## Code organization

### `io/hdf5.py` refactor

Current `io/hdf5.py` (~520 lines, forward-specific) reorganized into four internal layers in the same module:

#### 1. Helpers (mostly unchanged)
- `_set_nx_class`, `_get_git_sha_and_dirty`, `_sha256_of`, `_auto_max_workers`
- `_compute_frame` (parallel forward() worker)
- Naming constants: `MASTER_FORWARD = "dfxm_geo.h5"`, `MASTER_IDENTIFY = "dfxm_identify.h5"`, `SCAN_DIR_FMT = "scan{:04d}"`, `DETECTOR_FILE_FMT = "{name}_0000.h5"`

#### 2. Per-scan detector file writer
- `_write_detector_file(path, image_stack)` — pre-computed `(N, H, W)` array → LIMA-style file (called when images are already in RAM, e.g. identification single/multi modes computing one frame at a time).
- `_compute_and_write_detector_file_parallel(path, args_list, max_workers)` — workers run `forward()` in a ThreadPoolExecutor and stream into a pre-allocated file (replaces today's `_save_scan_parallel_to_h5`, but writing to its own file instead of a shared master).

Both write the `/entry_0000/...` LIMA structure with `NXroot/NXentry/NXdetector/NXdata`, chunks `(1, H, W)`, gzip-4 + shuffle, `@interpretation="image"`.

#### 3. `MasterWriter` context manager
Owns the master HDF5 handle for the duration of a run. Provides `add_scan(...)` for appending one `/N.1` per call.

```python
class MasterWriter:
    def __init__(self, path, *, cli, config_toml, kernel_npz=None): ...
    def __enter__(self) -> "MasterWriter": ...
    def __exit__(self, *exc): ...   # writes /dfxm_geo/ provenance on close
    def add_scan(
        self,
        *,
        scan_id: str,                              # "1.1", "2.1", ...
        title: str,
        start_time: str, end_time: str,
        sample: dict,                              # mode-specific; see per-mode layouts
        positioners: dict[str, ndarray | float],   # e.g. {"phi": (N,) array, "chi": 0.0}
        detector_links: dict[str, tuple[Path, str]],  # name → (rel_file_path, internal_h5_path)
        dfxm_geo: dict,                            # {"Hg": ..., "q_hkl": ..., ...}
        attrs: dict[str, str | list[str]],         # {"scan_mode": ..., "crystal_mode": ...}
    ) -> None: ...
```

`detector_links` is a dict to support multi-detector scans (e.g. `render_per_dislocation`).

#### 4. Public orchestrators
- `write_simulation_h5(...)` — same external signature as today; rewritten internally to use `MasterWriter` + `_compute_and_write_detector_file_parallel`. Called by `pipeline.run_simulation`.
- `write_identification_h5(config, output_dir, scan_iter)` — new. `scan_iter` is an iterator of per-config tuples that the runner functions yield.

### Pipeline updates

`pipeline.run_simulation` keeps its current API, just wires through the new writer:

```python
def run_simulation(config, output_dir):
    ...
    with MasterWriter(output_dir / "dfxm_geo.h5", cli=..., config_toml=..., kernel_npz=...) as master:
        # /1.1 dislocations
        scan_dir = output_dir / "scan0001"
        det_path = scan_dir / "dfxm_sim_detector_0000.h5"
        _compute_and_write_detector_file_parallel(det_path, args_dislocations)
        master.add_scan(scan_id="1.1", ..., detector_links={"dfxm_sim_detector": (det_path.relative_to(output_dir), "/entry_0000/dfxm_sim_detector/image")}, ...)
        if config.io.include_perfect_crystal:
            # /2.1 perfect
            scan_dir = output_dir / "scan0002"
            det_path = scan_dir / "dfxm_sim_detector_0000.h5"
            _compute_and_write_detector_file_parallel(det_path, args_perfect)
            master.add_scan(scan_id="2.1", ...)
```

`pipeline.run_identification` and the three `_run_identification_*` runners are restructured so the runner generates per-config tuples consumed by a shared write loop:

```python
def write_identification_h5(config, output_dir, scan_iter):
    master_path = output_dir / "dfxm_identify.h5"
    with MasterWriter(master_path, cli=..., config_toml=..., kernel_npz=...) as master:
        for k, scan_spec in enumerate(scan_iter):
            scan_dir = output_dir / SCAN_DIR_FMT.format(k + 1)
            scan_dir.mkdir(parents=True, exist_ok=True)
            # Write detector file(s) for this scan
            detector_links = {}
            for detector_name, args_list in scan_spec.detectors.items():
                det_path = scan_dir / DETECTOR_FILE_FMT.format(name=detector_name)
                _compute_and_write_detector_file_parallel(det_path, args_list)
                detector_links[detector_name] = (
                    det_path.relative_to(output_dir),
                    "/entry_0000/dfxm_sim_detector/image",
                )
            master.add_scan(
                scan_id=f"{k + 1}.1",
                ..., detector_links=detector_links, ...,
            )
```

The three runner functions (`_run_identification_single`, `_run_identification_multi`, `_run_identification_zscan`) become **generators** that yield `ScanSpec` named-tuples. Each handles its own loop structure (Cartesian / Monte Carlo / nested z-loop) but delegates the actual file writing.

### `IdentificationZScanConfig` simplification

Today:
```python
@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    z_offsets_um: list[float]
    phi_range_deg: float                  # DROP — duplicates [scan.phi]
    phi_steps: int                        # DROP
    chi_range_deg: float                  # DROP — duplicates [scan.chi]
    chi_steps: int                        # DROP
    include_secondary: bool = True
    secondary_rng_offset: int = 1
```

After:
```python
@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    z_offsets_um: list[float]
    include_secondary: bool = True
    secondary_rng_offset: int = 1
```

The `_run_identification_zscan` runner reads `config.scan.phi` / `config.scan.chi` for the per-config rocking grid. The 3 TOML files in `configs/identification_zscan.toml` are migrated accordingly: their inline `[zscan]` fields move to `[scan.phi]` / `[scan.chi]`.

### `IdentificationMonteCarloConfig` extension

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    n_samples: int
    pos_std_um: float
    n_png_previews: int = 0           # legacy; sidecar drop makes this dead — remove from dataclass
    render_per_dislocation: bool = False   # NEW: opt-in per-dis instance files
```

`n_png_previews` is removed (PNG previews dropped entirely; see Sidecars below).

### Eager guards (preserved)

`run_identification` reuses the same guard pattern `run_simulation` uses:

```python
if config.scan.two_dtheta.is_scanned or config.scan.z.is_scanned:
    unwired = [a for a in ("two_dtheta", "z") if config.scan.is_scanned(a)]
    raise ValueError(
        f"scan axes {unwired} are configured but not yet wired into "
        f"identification (v1.2.0 scope). For now, set range+steps only on "
        f"[scan.phi] and/or [scan.chi]."
    )
```

z-scan mode keeps its existing `mode='z-scan' + [scan.z]` mutual-exclusion guard (`__post_init__` in `IdentificationConfig`).

## Migration

### `dfxm-migrate-output` (existing tool, updated)

The `.npy` → HDF5 migrator already exists for forward-mode outputs. Updated to emit the new master + per-scan-dirs layout instead of a single file. Behavior change documented in release notes; CLI unchanged.

### `dfxm-migrate-h5` (new tool)

CLI: `dfxm-migrate-h5 <v110_dfxm_geo.h5> [--output <new-out-dir>]`

Default output: `<v110_dfxm_geo.h5>.v120/` (sibling directory; original file untouched).

Operation:
1. Open source v1.1.0 file in read mode.
2. For each scan_id in source (typically `"1.1"` and `"2.1"` for forward; could be more for identify if v1.1.x ever shipped identification HDF5 — it didn't, so only forward).
3. Extract pixel stack from `/<scan_id>/instrument/dfxm_sim_detector/data`.
4. Create `<new-out-dir>/scan{N:04d}/dfxm_sim_detector_0000.h5` LIMA-style file containing the stack.
5. Open destination master `<new-out-dir>/dfxm_geo.h5`; copy all non-pixel-data nodes from source (`/dfxm_geo/`, `/N.1/sample/`, `/N.1/instrument/positioners/`, `/N.1/dfxm_geo/`, `/N.1/dfxm_geo/analysis/`, `/N.1/measurement/` soft-links, `/N.1/title/start_time/end_time`).
6. Replace `/N.1/instrument/dfxm_sim_detector/data` with `ExternalLink` pointing at the new scan-dir file.

Round-trip test: `load_h5_scan(migrated_master, scan_id="1.1") == load_h5_scan(original, scan_id="1.1")` byte-for-byte.

Both tools live in `src/dfxm_geo/io/migrate.py` (existing module gets extended). CLI entry points added to `pyproject.toml`:

```toml
[project.scripts]
dfxm-migrate-output = "dfxm_geo.io.migrate:cli_main_npy_to_h5"
dfxm-migrate-h5     = "dfxm_geo.io.migrate:cli_main_h5_to_h5"
```

## Postprocess & analysis storage

`run_postprocess` (forward-only) is unchanged in behavior. It reads `/1.1/.../data` and `/2.1/.../data` via `load_h5_scan`, which transparently follows `ExternalLink`s (h5py handles this; no code changes needed for reads). Analysis output continues to land at `/1.1/dfxm_geo/analysis/` inside the master (`phi_list`, `chi_list`, `qi_field`, `chi_shift_deg`).

Identification mode has no postprocess in v1.2.0.

## Sidecars

Dropped:
- Identification mode's `manifest.csv` — all info now in `/N.1/sample/`.
- Identification mode's `images/*.png` previews — replaceable by a small `silx view` invocation or a one-shot extraction script (not bundled).
- Identification mode's `im_data/*.npy` per-image files — replaced by detector HDF5.

Kept:
- `dfxm_geo_random_dislocations.json` sidecar (forward mode with `crystal.mode='random_dislocations'`; introduced in sub-project C; unchanged).

## Testing strategy

### New test files

- `tests/io/test_master_writer.py` — `MasterWriter` unit tests:
  - Context manager open/close
  - Single `add_scan` writes correct `/N.1` structure
  - Multiple `add_scan` calls accumulate independent scans
  - `add_scan` is idempotent on `scan_id` (re-call overwrites cleanly)
  - External link targets are relative paths
  - Multi-detector `detector_links` produce multiple `NXdetector` groups
  - `/dfxm_geo/` provenance written exactly once on `__exit__`
- `tests/io/test_detector_file.py` — LIMA-style file writer unit tests:
  - `_write_detector_file` produces NXroot/NXentry/NXdetector/NXdata structure
  - Chunks `(1, H, W)`, gzip-4 + shuffle
  - `@interpretation="image"` attr present
  - `plot/` and `measurement/` SoftLinks correctly point at `image`
  - `_compute_and_write_detector_file_parallel` round-trips identical pixels to a serial reference
- `tests/test_pipeline_identification_hdf5.py` — end-to-end per sub-mode (single, multi, z-scan):
  - Master file written at `out_dir/dfxm_identify.h5`
  - Correct number of `/N.1` entries per sub-mode + config
  - Correct number of `scanNNNN/` directories on disk
  - External links resolve via h5py and yield identical pixels to a direct forward() call
  - Per-mode `sample/` layouts match this spec
  - Per-`/N.1` attrs (`scan_mode`, `scanned_axes`, `identify_mode`) correctly populated
- `tests/test_identification_multi_per_dis.py` — opt-in `[multi].render_per_dislocation=true`:
  - Each `scanNNNN/` has 3 detector files
  - `/N.1/instrument/` has 3 `NXdetector` groups
  - Pixel sums: `dis0 + dis1 ≠ combined` in general (because forward() isn't strictly linear in the strain superposition — verify the contract is "each detector is its own independent forward() call, all using the same sample/dislocations/0 and /1 spec but with the other zeroed out").
- `tests/test_identification_scan_modes.py` — identification with `[scan.phi]` / `[scan.chi]` grids per config:
  - `single` + `[scan.phi]` scanned → each `/N.1` has `(phi_steps, H, W)` detector data
  - `multi` + `[scan.phi]` + `[scan.chi]` → each `/N.1` has `(phi_steps × chi_steps, H, W)` per MC sample
  - `z-scan` with `[scan.phi]` / `[scan.chi]` from B+C schema (now the only path) → same shape as today
  - Positioners stored correctly: scanned axes as `(N,)` arrays, fixed as scalars
- `tests/io/test_migrate_h5.py` — `dfxm-migrate-h5` round-trip:
  - Given a v1.1.0 fixture file, migrate produces master + 2 scan dirs
  - `load_h5_scan` of migrated master returns byte-identical pixels to `load_h5_scan` of original
  - `/dfxm_geo/` provenance copied losslessly
  - All `/N.1/sample/`, `/N.1/dfxm_geo/`, `/N.1/dfxm_geo/analysis/` content preserved

### Updated tests (fixture changes)

- `tests/test_pipeline.py`
- `tests/test_hdf5_provenance.py`
- `tests/test_pipeline_identification.py` (current `.npy`/manifest-based tests; rewrite for HDF5)
- `tests/test_pipeline_scan_modes.py`
- `tests/test_pipeline_crystal_modes.py`
- `tests/test_pipeline_multi_reflection.py`
- Any other test that writes a fake `dfxm_geo.h5` for fixture purposes needs to emit the new master + per-scan layout.

### Pre-existing skips / xfails (untouched)

- `test_hdf5_writer_bit_equivalent_to_legacy_npy_golden` — Find_Hg-seed xfail; orthogonal to layout change.
- `test_dfxm_forward_with_sample_remount_S2_runs` — `slow` marker; unaffected.

### Manual verification

- Open a v1.2.0 forward `dfxm_geo.h5` in `silx view`; verify the BLISS scan navigator shows `/1.1` and `/2.1` correctly and the detector images render.
- Open the same file in `darfix`; verify the rocking scan loads as a 2-D dataset.
- Repeat for `dfxm_identify.h5` (single mode default, then with `[scan.phi]` scanned).
- Capture both checks as a final manual-verification phase in the implementation plan.

### Quality gates

- mypy 0 errors on `src/dfxm_geo/`
- `python -m pytest -q` green (no new failures relative to current `main`)
- The B+C-era 430 passed / 4 skipped / 10 deselected / 1 xfailed baseline holds, with the new HDF5 tests added on top.

## Documentation updates

- `docs/output-format.md` — full rewrite for master + per-scan layout. Sections:
  - High-level structure (the tree diagram from this spec)
  - Master file (`/dfxm_geo/`, `/N.1/`, attrs)
  - Per-scan LIMA-style files
  - Frame ordering (unchanged from v1.1.0)
  - NX_class table
  - Reading via `load_h5_scan`, h5py, darfix, darling
  - Compression
  - Migration from v1.1.0 (link to `dfxm-migrate-h5`)
- `docs/release-notes-1.2.0.md` — new. Covers A + B + C + D (already on main, untagged) + E. Calls out the breaking format change and migration path explicitly.
- `CLAUDE.md` working notes — update the pipeline-features arc table (mark E as shipped on tag), update the tag-chain reference (`v1.1.0` → `v1.2.0`), refresh the "what's next" pointer to F.
- README — verify whether it discusses output format; if so, replace the v1.1.0 reference.

## Deferred follow-ups (not in v1.2.0; tracked elsewhere)

- `[scan.two_dtheta]` and `[scan.z]` wiring (v1.3.0; tracked in B+C deferred follow-ups).
- z-scan mode consolidation (fold into `single + [scan.z]` once that wiring lands).
- Pixel-level segmentation masks for multi mode.
- z-scan `render_per_dislocation` analogue.
- `sample_dis = -1.0` sentinel cleanup (forward mode, non-wall crystals).
- `Find_Hg_from_population` unused params (`psize`, `zl_rms`).
- `_SLIP_SYSTEM_111` table extension (6/12 → all 12 FCC slip systems).
- `/2.1` HDF5 attrs (`scan_mode`, `scanned_axes`, `crystal_mode` currently only on `/1.1`).
- All other B+C-deferred follow-ups (see `[[session-handoff-2026-05-21]]`).

## Open questions / risks

- **Find_Hg seeding xfail.** Still pending; bit-equivalence with the IUCrJ-2024 pickle golden remains xfailed. Unaffected by this layout change but worth keeping on the radar.
- **Many small files for `dfxm-identify single`.** A default-config single run produces ~456 scan directories × 1 detector file = ~456 files (+ 1 master). On Windows this can be slow under some antivirus configurations. Acceptable for now; if it becomes a real-world problem we can add a `[io].batch_scans_per_file` knob to pack N configs per detector file (v1.3.0+).
- **`render_per_dislocation` and Poisson noise.** When per-dislocation rendering is enabled, the per-dis files and the combined file each get their *own* Poisson draw (different `rng` state would give incorrect signal/background balance vs. summing). The combined file uses the canonical noise stream; the per-dis files use a separate derived stream so they're independent samples of the noiseless underlying images. To be confirmed in the implementation plan: the simplest correct behavior is "noise applied only to the combined detector; per-dis detectors are noiseless". Lock this down in the plan-writing phase.
- **External link path stability under symlinks.** Relative external links resolve from the master file's directory. If the user creates a symlink to the master, h5py follows the symlink and resolves links from the symlink target's directory, not the symlink's directory. Document this in `docs/output-format.md`.
