---
title: Default config flip to "simple" (sub-project F)
status: draft
date: 2026-05-23
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - docs/superpowers/specs/2026-05-21-scan-modes-crystal-layouts-design.md (sub-project B+C — established the discriminated-union CrystalConfig + per-axis ScanConfig that F builds on; explicitly defers F)
  - docs/superpowers/specs/2026-05-20-multi-reflection-lookup-design.md (sub-project D — ReciprocalConfig + kernel lookup; F adds defaults but keeps the lookup contract)
  - src/dfxm_geo/pipeline.py (current CenteredCrystalConfig, WallCrystalConfig, RandomDislocationsConfig, CrystalConfig, ReciprocalConfig, SimulationConfig, IdentificationConfig, IdentificationCrystalConfig, IdentificationNoiseConfig)
  - configs/default.toml, configs/identification_*.toml, configs/variants/*.toml (11 TOML configs; F rewrites default.toml and the 3 identification configs, leaves the 7 variants alone)
---

# Default config flip to "simple" (sub-project F)

## Purpose

Sub-project B+C delivered the discriminated-union `CrystalConfig` and the
per-axis `ScanConfig`. Sub-project D delivered `ReciprocalConfig` and the
config-driven kernel lookup. Both designs left the *dataclass-level
defaults* anchored to the legacy IUCrJ-2024 wall layout — `[crystal]`
and `[reciprocal]` blocks are mandatory in TOML, and `WallCrystalConfig`
silently substitutes publication-grade values (`dis=4.0`, `ndis=151`,
`sample_remount="S1"`) for any field the user omits.

Sub-project F flips the conceptual default. After F, the simplest
possible TOML — an empty file — produces a valid, minimal forward or
identification run: **one centered dislocation, one detector image, Al
111 @ 17 keV.** Everything else (scans, wall, random_dislocations,
non-Al reflections, post-processing tweaks) is opt-in. Concurrently,
F removes the publication-grade defaults from `WallCrystalConfig` so
that wall-mode runs always state their parameters explicitly.

F is the v2.0.0 release. The breaking change is exactly the
`WallCrystalConfig` defaults strip: any caller that constructed
`WallCrystalConfig()` with no arguments now gets a `TypeError`. All 5
in-repo `configs/variants/*wall*.toml` files already specify the three
fields explicitly (verified during brainstorming), so no TOML config in
the repository is broken by the strip — only Python construction sites
are.

## Decisions (Q1–Q6)

Captured via `superpowers:brainstorming` on 2026-05-23.

- **Q1 — "Simple" = 1 dislocation + 1 image.** Empty/minimal TOML must
  produce `mode="centered"` single dislocation at origin + `ScanConfig`
  with all axes fixed at 0 ("single" derived mode). The most reductive
  reading of "simple"; every scan, wall layout, and noise behavior is
  opt-in.
- **Q2 — `[reciprocal]` defaults to Al 111 @ 17 keV.** Empty TOML truly
  runs (no "missing [reciprocal] block" error). The existing
  `_lookup_kernel_path` machinery already errors loudly when the
  matching bootstrap kernel is missing, so the silent-mismatch risk is
  bounded.
- **Q3 — Strip `WallCrystalConfig` defaults.** `dis`, `ndis`,
  `sample_remount` become required kw-only fields. Wall mode users
  always declare their layout parameters. This is the breaking change
  that drives the v2.0.0 bump.
- **Q4 — `CenteredCrystalConfig` defaults to canonical FCC primary.**
  `b=(1, 0, -1)`, `n=(1, 1, 1)`, `t=(1, -2, 1)`. Satisfies `b·n=0` and
  `t∥(n×b)` (line direction parallel to the cross product, sign
  irrelevant). Most-textbook choice from the 12 FCC {111}<110> systems.
- **Q5 — `default.toml` stays verbose, marked "override-only".** Header
  comment explains that every block below is the *default* value and is
  safe to delete. File becomes a tutorial-style overlay reference,
  functionally redundant with the empty case. (Alternatives considered:
  minimal stub; split into `default.toml` + `reference.toml`. Both
  rejected: verbose-marked file is the lowest-cognitive-load tutorial.)
- **Q6 — Identification gets the same cascade.** `dfxm-identify` with
  an empty TOML produces `mode="single"` + canonical hypothesis sweep
  over {111} + default noise. `IdentificationCrystalConfig`,
  `IdentificationConfig.mode`, and `IdentificationConfig.crystal` all
  gain defaults. `IdentificationZScanConfig.z_offsets_um` stays required
  (only consumed when `mode="z-scan"`; existing gating preserved).

## Approach

Single PR, hard cut-over. Matches the B+C, E, A+D, and v1.3.0
precedents. Per [[dfxm-no-backcompat-constraint]], no deprecation shim
or two-cycle migration. All test fixtures touching
`WallCrystalConfig()` with no args are updated in the same PR.

Alternatives considered and rejected:
- **Two-phase additive then breaking.** PR1 adds defaults (additive,
  v1.4.0); PR2 strips wall defaults (v2.0.0). Doubles release/PR
  overhead with no migration benefit (Sina is sole user).
- **Forward now, identification deferred.** Forward in v2.0.0,
  identification cascade in v2.1.0. Rejected: Q6 explicitly chose
  bundled; splitting after deciding to bundle reads inconsistent and
  fragments the release notes.

## Architecture

### Files modified

- **`src/dfxm_geo/pipeline.py`** — main work:
  - **`CenteredCrystalConfig`** — gain class-level defaults:
    - `b: tuple[int, int, int] = (1, 0, -1)`
    - `n: tuple[int, int, int] = (1, 1, 1)`
    - `t: tuple[int, int, int] = (1, -2, 1)`
    - `__post_init__` validation (`b·n=0`, `t∥(n×b)`) unchanged.
  - **`WallCrystalConfig`** — strip all three defaults:
    - `dis: float`  (no default; required)
    - `ndis: int`   (no default; required)
    - `sample_remount: str`  (no default; required)
    - `__post_init__` `sample_remount in SAMPLE_REMOUNT_OPTIONS`
      validation unchanged.
    - Dataclass should be `kw_only=True` so that the strip surfaces
      cleanly as `TypeError: WallCrystalConfig() missing 3 required
      keyword-only arguments` rather than positional-arg confusion.
      (Already `@dataclass` without kw_only; the cleanest signal is to
      add `kw_only=True` in the same commit that strips defaults.)
  - **`ReciprocalConfig`** — gain defaults + relax `from_dict`:
    - `hkl: tuple[int, int, int] = (-1, 1, -1)`  (Al 111)
    - `keV: float = 17.0`
    - `from_dict(None)` returns `ReciprocalConfig()` (default-constructed)
      instead of raising. `from_dict({})` likewise.
    - `from_dict(data)` with `data` non-empty: existing key-presence
      checks (`hkl`, `keV` required if `data` is non-empty) **soften**:
      missing keys fall back to defaults rather than raising. Rationale:
      partial overrides should work (e.g. `[reciprocal] keV = 21.0`
      keeps default hkl). Keeps the override path ergonomic.
    - `_validate_reflection(hkl, keV, 4.0495e-10)` still runs on the
      resolved values.
  - **`CrystalConfig`** — gain default path on `from_dict`:
    - New module-level helper or classmethod `CrystalConfig.default()`
      returns `CrystalConfig(mode="centered",
      centered=CenteredCrystalConfig())`. Used as the
      `SimulationConfig.crystal` default factory and as the empty-TOML
      fallback.
    - `from_dict(None)` and `from_dict({})` → `CrystalConfig.default()`.
    - `from_dict({"mode": "<m>"})` with `m` valid but no sub-block:
      preserves current strict behavior — raises `ValueError("crystal
      mode='wall': [crystal.wall] sub-block is required")`. Rationale:
      writing `[crystal] mode = "wall"` is an explicit declaration of
      intent; silently substituting wall defaults would be exactly the
      footgun Q3 strips away.
    - `from_dict({"mode": "centered"})` with no `[crystal.centered]`
      sub-block: same — raises. The default path is "omit `[crystal]`
      entirely", not "write `[crystal] mode='centered'`". Keeps the
      "explicit intent gets explicit handling" rule symmetric across
      modes.
    - `__post_init__` validations unchanged (mode/sub-block consistency,
      no extras).
  - **`SimulationConfig`** — field defaults flip:
    - `crystal: CrystalConfig = field(default_factory=CrystalConfig.default)`
      (was: no default — required).
    - `reciprocal: ReciprocalConfig = field(default_factory=ReciprocalConfig)`
      (was: `ReciprocalConfig | None = None`; the `Optional` collapses
      to the always-present default).
    - `scan`, `io`, `postprocess` defaults unchanged (already
      default-constructible).
    - `from_toml(path)`: unchanged at the call site (calls
      `CrystalConfig.from_dict(raw.get("crystal"))` etc., which now
      handle `None`/`{}` gracefully).
    - `run_simulation` no longer needs the runtime `if reciprocal is
      None: raise` guard (collapsed by the type flip).
  - **`IdentificationCrystalConfig`** — gain default:
    - `slip_plane_normal: tuple[int, int, int] = (1, 1, 1)`
    - All other fields already had defaults.
  - **`IdentificationConfig`** — flip mode + crystal defaults:
    - `mode: Literal["single", "multi", "z-scan"] = "single"`
    - `crystal: IdentificationCrystalConfig =
       field(default_factory=IdentificationCrystalConfig)`
    - `scan: ScanConfig = field(default_factory=ScanConfig)` — already
      done in B+C; no change.
    - `noise: IdentificationNoiseConfig =
       field(default_factory=IdentificationNoiseConfig)` — already done;
      no change.
    - `multi: IdentificationMonteCarloConfig | None = None` — unchanged
      (only consumed when `mode="multi"`; existing validation gates).
    - `zscan: IdentificationZScanConfig | None = None` — unchanged
      (only consumed when `mode="z-scan"`).
    - `from_toml(path)`: unchanged at call sites (defaults cascade
      through `field(default_factory=...)`).

- **`configs/default.toml`** — header comment rewrite + flagging:
  - Top header: explains that every block below is the *default*
    value, safe to delete; an empty TOML produces a single-image
    centered-dislocation run on Al 111 @ 17 keV.
  - All current blocks retained (reciprocal, scan.phi, scan.chi,
    crystal, crystal.centered, io, postprocess).
  - One small block-level comment per block (e.g. `# Defaults shown.
    Delete to fall back.`) so a user editing the file sees the
    override-only framing inline.
  - The current `[scan.phi]` + `[scan.chi]` blocks **stay** (so the
    out-of-the-box experience with `dfxm-forward --config
    configs/default.toml` still produces the recognizable 2D mosa
    grid — only an explicitly-empty TOML drops to single-image
    behavior).
  - Commented-out `[crystal.wall]` and `[crystal.random_dislocations]`
    sibling-block examples stay below `[crystal.centered]`.

- **`configs/identification_single.toml`**,
  **`identification_multi.toml`**, **`identification_zscan.toml`** —
  same header treatment. Each block stays visible with an
  override-only comment. No structural change — these configs already
  exercise the non-default code paths (`mode="multi"`, `mode="z-scan"`,
  noise tweaks).

- **`configs/variants/*.toml`** (7 files: `dis_*.toml`,
  `forward_strain_scan.toml`, `forward_z_scan.toml`,
  `sample_remount_S2.toml`) — **untouched**. These are deliberate
  overrides of the default; the override-only framing applies
  implicitly.

- **Tests** —
  - **NEW** `tests/test_defaults_simple.py`: unit tests for the new
    default factories. Covers `CenteredCrystalConfig()` returns
    canonical FCC primary; `ReciprocalConfig()` returns Al 111 @ 17 keV;
    `CrystalConfig.default()` returns mode="centered" + canonical
    centered sub-block; `WallCrystalConfig()` raises `TypeError`;
    `SimulationConfig()` is constructible with zero args;
    `IdentificationConfig()` likewise.
  - **NEW** `tests/test_empty_toml_runs.py`: end-to-end smoke. Writes
    a literally-empty `.toml` to a tmp dir, runs the forward pipeline
    via `SimulationConfig.from_toml`, asserts the kernel lookup
    succeeds, one detector image written to HDF5, file readable.
    Parallel test for `dfxm-identify`.
  - **NEW** `tests/test_partial_reciprocal_override.py`: verify
    `[reciprocal] keV = 21.0` (with no `hkl`) resolves to the default
    `hkl=(-1,1,-1)` + user `keV=21.0`. Same for `hkl`-only.
  - **NEW** `tests/test_wall_no_defaults.py`: `WallCrystalConfig()`
    raises `TypeError`; `WallCrystalConfig(dis=4.0)` raises (missing 2);
    `WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")`
    succeeds.
  - **RENAMED** `tests/test_version_is_1_2_0.py` →
    `tests/test_version_is_2_0_0.py` (`git mv` + edit the asserted
    version string). Same single-line check that the installed package
    version matches the release.
  - **UPDATED** any existing test that called `WallCrystalConfig()`
    bare. Grep during implementation; expected count: small (<5).
    Replacement: explicit `WallCrystalConfig(dis=4.0, ndis=151,
    sample_remount="S1")` if the test wanted the IUCrJ values.
  - **UPDATED** `tests/test_pipeline.py` smoke fixtures that relied on
    `SimulationConfig.from_toml` raising on missing `[crystal]` /
    `[reciprocal]`: either delete the assertion (no longer raises) or
    flip it to check the default values cascade through.

### Removed API

None outright. Behavior shifts:

- `ReciprocalConfig.from_dict(None)` no longer raises `ValueError`.
  Returns the default. Callers that depended on the raise (if any) need
  updating — grep during implementation; expected: only `from_toml`,
  which now transparently benefits.
- `CrystalConfig.from_dict(None)` no longer raises. Returns the default.
- `SimulationConfig.reciprocal: ReciprocalConfig | None`'s `None` branch
  is gone — type tightens to `ReciprocalConfig`. Anyone narrowing on
  `is None` will now lint-trip (mypy `unreachable`).
- `WallCrystalConfig()` with no args raises `TypeError` (was: returned
  IUCrJ-default-constructed instance).

### Key invariants (preserved + new)

- **Preserved**: `CrystalConfig.<mode>_sub_block is not None ⇔
  CrystalConfig.mode == <mode>`. Defaults don't bypass this — the
  default path always sets `mode="centered"` AND
  `centered=CenteredCrystalConfig()` together.
- **Preserved**: `ScanConfig.derived_mode_name()` is the only mode-name
  label source.
- **NEW**: An empty TOML file is a valid `SimulationConfig` input. The
  resolved config is byte-equivalent to
  `SimulationConfig(crystal=CrystalConfig.default(),
  scan=ScanConfig(), io=IOConfig(),
  postprocess=PostprocessConfig(), reciprocal=ReciprocalConfig())`.
- **NEW**: Writing `[crystal] mode = "<m>"` without `[crystal.<m>]`
  still raises. The "default" path is omission, not declaration —
  silent wall-defaults substitution is exactly what F kills.
- **NEW**: `ReciprocalConfig.from_dict` accepts partial overrides
  (one of `hkl`/`keV` provided, the other defaults). Symmetric for
  both keys.

## TOML schema impact

### Forward (`dfxm-forward`)

| Block | Pre-F status | Post-F status |
|---|---|---|
| `[reciprocal]` | Required (`from_dict(None)` raised) | Optional; defaults to Al 111 @ 17 keV; partial overrides supported |
| `[crystal]` | Required (`from_dict(None)` raised) | Optional; defaults to `mode="centered"` + canonical FCC primary |
| `[crystal.centered]` (when `mode="centered"`) | Required | Required (if `mode` is declared) |
| `[crystal.wall]` (when `mode="wall"`) | Optional (silent IUCrJ defaults) | **Required**; `dis`/`ndis`/`sample_remount` all explicit |
| `[crystal.random_dislocations]` (when mode set) | Required | Required (unchanged) |
| `[scan]` and `[scan.<axis>]` | Optional (defaults to single) | Optional (unchanged) |
| `[io]`, `[postprocess]` | Optional | Optional (unchanged) |

### Identification (`dfxm-identify`)

| Block | Pre-F status | Post-F status |
|---|---|---|
| `[mode]` field on root | Required | Optional; defaults to `"single"` |
| `[crystal]` (hypothesis sweep) | Required | Optional; defaults to `slip_plane_normal=(1,1,1)` + full {111} sweep |
| `[scan]` | Optional (per B+C) | Optional (unchanged) |
| `[noise]` | Optional | Optional (unchanged) |
| `[multi]` (when `mode="multi"`) | Required | Required (unchanged) |
| `[zscan]` (when `mode="z-scan"`) | Required | Required (unchanged); `z_offsets_um` stays mandatory |

## `configs/default.toml` shape (post-F)

```toml
# Default DFXM forward-simulation config.
#
# Every block below shows the DEFAULT value the pipeline would use if
# the block were omitted entirely. A literally-empty .toml file is a
# valid input: dfxm-forward --config <empty.toml> produces a single
# detector image of a single canonical FCC dislocation at the origin,
# Al 111 reflection at 17 keV.
#
# Edit any block to override; delete any block to fall back to the
# default shown here.
#
# The active [crystal] + [scan] blocks below are tuned to produce the
# recognizable 2D mosa rocking grid out-of-the-box for users running
# `dfxm-forward --config configs/default.toml` — this is NOT what an
# empty .toml produces (empty = single image).

[reciprocal]
# Default: Al 111 reflection at 17 keV. Bundled kernel matches.
hkl        = [-1, 1, -1]
keV        = 17.0
# Bootstrap-only params (consumed by `dfxm-bootstrap`, not by
# forward/identify). Safe to delete this section if not (re)bootstrapping.
Nrays      = 100_000_000
npoints1   = 400
# ... (rest of bootstrap params unchanged from current default.toml)

[scan.phi]
# Override-only: omit for fixed-at-0. Together with [scan.chi] below this
# gives the 2D "mosa" rocking grid.
range = 6e-4
steps = 61

[scan.chi]
range = 2e-3
steps = 61

# [scan.two_dtheta]    # omitted -> fixed at 0
# [scan.z]             # omitted -> fixed at 0

[crystal]
# Default: mode = "centered" with canonical FCC primary slip system.
# Override-only: change `mode` and provide the matching sub-block.
mode = "centered"

[crystal.centered]
# Default: canonical FCC {111}<110> primary slip system.
# (b=(1,0,-1), n=(1,1,1), t=(1,-2,1) is what an empty .toml would use.)
b = [1, -1, 0]   # this default.toml uses the IUCrJ paper's b/n/t
n = [1, 1, 1]    # to keep historical reproducibility for users
t = [1, 1, -2]   # running `dfxm-forward --config configs/default.toml`

# [crystal.wall]       # required: dis, ndis, sample_remount — no defaults
# dis = 4.0
# ndis = 151
# sample_remount = "S1"

# [crystal.random_dislocations]
# ndis = 4
# sigma = 5.0
# min_distance = 4.0
# seed = 42

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "images10"
perfect_dirname = "images10_perf_crystal"
include_perfect_crystal = true

[postprocess]
enabled = true
chi_oversample = 20
phi_oversample = 20
chi_oversample_for_shift = 100
figures_dirname = "figures"
data_dirname = "analysis"
```

Note: the `default.toml` `[crystal.centered]` block keeps the IUCrJ
b/n/t `(1,-1,0)/(1,1,1)/(1,1,-2)` rather than the new dataclass
canonical `(1,0,-1)/(1,1,1)/(1,-2,1)`. Rationale: users running
`dfxm-forward --config configs/default.toml` reproduce the existing
forward output bit-for-bit; only the *empty TOML* fall-through gets
the new canonical defaults. The header comment makes this explicit.

## Migration / breaking changes

The only breaking change is the `WallCrystalConfig` defaults strip.
Affected surfaces:

1. **Python construction sites** — anywhere code calls
   `WallCrystalConfig()` with fewer than 3 args (or positionally,
   relying on defaults). All call sites in `src/dfxm_geo/` and `tests/`
   are updated in the same commit that strips the defaults. Grep target:
   `\bWallCrystalConfig\s*\(\s*\)` and `\bWallCrystalConfig\s*\(\s*[^)]*\)`
   (then triage which calls miss required args).

2. **TOML configs** — verified during brainstorming: all 5 wall-mode
   variant configs (`dis_0p25.toml`, `dis_0p5.toml`, `dis_1.toml`,
   `dis_2.toml`, `sample_remount_S2.toml`) already specify `dis`, `ndis`,
   `sample_remount` explicitly. **No TOML migration needed.**

3. **External / cluster scripts** — any DTU HPC script or notebook that
   constructed `WallCrystalConfig()` bare will surface `TypeError` on
   the next run. Surfaces in the v2.0.0 release notes as the headline
   breaking change with a copy-paste fix line.

The non-breaking softenings:

- `ReciprocalConfig.from_dict(None)` / `from_dict({})` / partial
  override: any caller that previously caught the `ValueError("missing
  [reciprocal] block")` now silently sees a default config. Search for
  callers that rely on the raise — expected: none beyond
  `SimulationConfig.from_toml`, which benefits.
- `CrystalConfig.from_dict(None)`: same story.
- `SimulationConfig.crystal` / `.reciprocal` type tightening: mypy may
  flag any `if config.reciprocal is None:` branch as unreachable. Fix
  the call site (delete the branch).

## Test surface plan

Detailed list under "Files modified > Tests". Summary of test deltas:

- **4 new test files** (defaults_simple, empty_toml_runs,
  partial_reciprocal_override, wall_no_defaults).
- **1 renamed test file**: `test_version_is_1_2_0.py` →
  `test_version_is_2_0_0.py` (single-line version-string change).
- **~3-5 existing tests updated** for the `WallCrystalConfig()` strip
  and the from_dict no-longer-raises behavior (exact count discovered
  during implementation via grep).
- **No fixture rewrites** for the 7 variant configs — they're already
  explicit.
- **Bit-equivalence safety net**: `tests/data/golden/Fd_find_smoke.npy`
  is unaffected because `configs/default.toml`'s active `[crystal.centered]`
  block keeps the historical b/n/t (the new canonical defaults are
  only reached via empty TOML, which the golden doesn't cover).

## HDF5 schema impact

None. The new BLISS `/N.1` attrs added in B+C (`scan_mode`,
`scanned_axes`, `crystal_mode`) are populated identically:
empty-TOML runs write `scan_mode="single"`, `scanned_axes=[]`,
`crystal_mode="centered"`. The full `config_toml` round-trip still
captures the resolved config (defaults materialized).

## Out of scope (v2.0.0)

- **Migrating module-level FOV constants** (`Npixels`, `psize`,
  `zl_rms` in `forward_model.py`) to a `[detector]` config block.
  Still v1.3.0+ follow-up tracked in B+C.
- **Random-dislocations defaults** (`sigma`, `min_distance`). Already
  have `None`-defaults that resolve at draw time. F doesn't touch.
- **Find_Hg seeding.** Two xfailed bit-equivalence tests stay xfailed.
  Orthogonal concern, separate spec.
- **TOML deprecation warnings.** Per [[dfxm-no-backcompat-constraint]]
  + the no-shim approach used across A/D/B/C/E, F breaks loudly via
  `TypeError`; no soft warnings.
- **darling / darfix interop**. Tracked in
  [[followups-darling-external-link-blind]].
- **Collaborator branch rebases.** Phase 11 is closed (collab branches
  stay as ground-truth references) — F doesn't push to them.
- **Bootstrap re-run on cluster**. The kernel lookup behavior is
  unchanged; existing v1.2.0+ bootstraps remain valid for v2.0.0.

## Release plan

- **Branch**: `feature/v200-default-simple` (matches the
  `feature/v1<MM>-<topic>` naming used by v1.3.0-A / v1.3.1).
- **Commits**: 10-15 atomic — one per dataclass change, one for
  `default.toml`, one for each new test file, one for the version bump
  + rename, one for release notes.
- **Suite gate**: must remain 476+ passed / 2 xfailed (current v1.3.1
  baseline) modulo the new test count. Any failure inside the strip
  is fixed in the same commit, not a follow-up.
- **mypy gate**: `mypy src/dfxm_geo/` reports 0 errors.
- **Version bump**: `pyproject.toml` 1.3.1 → 2.0.0.
- **Release notes**: `docs/release-notes-2.0.0.md`. Headline section:
  "Breaking change: WallCrystalConfig requires explicit dis/ndis/
  sample_remount". Copy-paste migration snippet for legacy bare
  `WallCrystalConfig()` callers. Then sections for "Empty TOML now
  runs" (forward + identify) and "Partial [reciprocal] overrides".
- **Tag**: `v2.0.0` annotated, pushed last (after main merge). Triggers
  `publish.yml` → TestPyPI auto-publish + PyPI manual approval gate.
- **CLAUDE.md updates**: mark sub-project F shipped; update "main
  HEAD" + "Latest release tag" lines; close the F row in the
  pipeline-features arc table.

## Open questions / TBD

- **Identification empty-TOML smoke fidelity.** The end-to-end smoke
  test for `dfxm-identify` on an empty TOML needs to confirm the
  default hypothesis sweep + canonical slip plane + default noise
  actually produces parseable output. If the sweep is computationally
  heavy (full {111} family × full angle range × 1000 multi samples),
  the test may need a marker or a reduced-sweep override. Resolve
  during implementation; default to keeping the test lean by overriding
  `IdentificationConfig.crystal.sweep_all_slip_planes = False` or
  similar in the test fixture if the bare-defaults run takes >5s.
- **`SimulationConfig.reciprocal: ReciprocalConfig | None = None`**
  call sites. Spec assumes only `from_toml` and `run_simulation` care;
  grep during implementation to confirm no test fixture relies on
  passing `reciprocal=None` explicitly.
- **`kw_only=True` blast radius on `WallCrystalConfig`.** Adding
  `kw_only=True` is recommended for clean `TypeError` messages but may
  surface as a separate breaking change for any caller using positional
  args. Verify during implementation; if positional callers exist,
  decide whether to also rewrite them or drop the `kw_only` add.
