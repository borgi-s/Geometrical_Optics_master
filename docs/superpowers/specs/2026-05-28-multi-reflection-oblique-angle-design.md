# Oblique-angle DFXM geometry + multi-reflection iteration — design

**Status:** draft (brainstorm-approved).
**Target releases:** Phase A = v2.3.0, Phase B = v2.4.0.
**Author:** Sina Borgi (DTU) with Claude Code.
**Date:** 2026-05-28.

## Goal

Add oblique-angle DFXM geometry and multi-reflection iteration to `dfxm_geo`,
matching the *Detlefs et al. 2025* / paper-arXiv:2503.22022 formulation. The
arc enables (a) reproducing a single oblique-angle paper figure (Phase A), and
(b) running a bundled multi-reflection simulation suitable for deformation-
gradient-tensor reconstruction from the same crystalline volume (Phase B).

This is the precursor to the eventual v3.0.0 release, which will add `.cif`
support and drop the `simplified` geometry mode.

## Motivation

`dfxm_geo` today implements the *simplified geometry* of Poulsen et al. 2021:
the cradle tilt μ is used to bring each reflection into the symmetric Bragg
condition, η = 0 by construction, and each reflection probes a *different*
crystalline volume. For multi-reflection deformation reconstruction, the user
needs reflections that probe the *same* volume — that requires the oblique
geometry of Detlefs et al. 2025, which locks μ = 0 and η = η_C (a constant)
and varies only ω. The whole point of oblique geometry is to make multi-
reflection imaging from a common volume practical; the two features are
naturally bundled.

Reference materials:
- Paper PDF (this arc's source): `arXiv:2503.22022v1`, particularly §3.3, §4,
  Appendix A (search algorithm), Appendix F (closed-form Gaussian resolution).
- Prototype (uncommitted, pre-cleanup era): `C:\Users\borgi\Documents\Oblique_Angle`.
- Reference implementation: Henningsson's `darkmod` (GitHub).
- Future-related: `.cif` support, [[followups-cif-crystal-structures]].

## Non-goals (deferred, listed for clarity)

- `.cif` parsing or non-cubic lattices — v3.0.0.
- Dropping `simplified` geometry mode — v3.0.0.
- Goniometer μ motor as a free axis — v3.0.0 or later (oblique mode locks μ=0).
- Persistent-worker pool — separate arc, see [[session-handoff-2026-05-27-phase2-float32-fanout]].
- ML dataset generation — separate arc on top of the persistent pool.

## Phasing

Two phases, one arc. Each ships independently and is **bit-exact backward-
compatible** with v2.2.0 when its new features aren't used. The bit-identity
constraint is the hardest, gating, regression test.

### Phase A — Oblique-angle infrastructure (v2.3.0)

**Goal:** a single oblique-angle reflection (`mode="oblique"`, η ≠ 0, single
hkl) reproduces a paper figure end-to-end. No `[[reflections]]` list yet.

### Phase B — Multi-reflection layer (v2.4.0)

**Goal:** bundled multi-reflection runs validated against paper Table A.2 +
darkmod cross-check. `[[reflections]]` list, shared LUT per group, shared
`Find_Hg` per group.

## Architecture

### New module: `dfxm_geo/crystal/oblique.py`

Pure math, no I/O, no globals. Houses:

```python
@dataclass(frozen=True, kw_only=True)
class CrystalMount:
    lattice: Literal["cubic"]      # v2.3.0 only; widens in v3.0.0
    a: float                       # lattice parameter, metres
    mount_x: tuple[int, int, int]  # Miller indices aligned with lab x̂
    mount_y: tuple[int, int, int]  #                                 ŷ
    mount_z: tuple[int, int, int]  #                                 ẑ

    def __post_init__(self) -> None:
        # Validate: each mount_* is integer length-3; mutually orthogonal (cubic).
        ...

    @cached_property
    def C_s(self) -> np.ndarray:
        """Cell matrix s.t. (2π) C_s^{-T} G_hkl = Q_s^{(0)}  (paper eq 24)."""
        return self.a * np.eye(3)

    @cached_property
    def U_mount(self) -> np.ndarray:
        """Crystal→lab rotation from mount Miller indices."""
        ...


@dataclass(frozen=True)
class ReflectionGeometry:
    hkl: tuple[int, int, int]
    keV: float
    omega_1: float     # rad — first ω solution (NaN if no real solution)
    eta_1:   float     # rad — η at ω_1
    theta_1: float     # rad — Bragg angle (== theta_2)
    omega_2: float     # rad — second ω solution (NaN if no real solution)
    eta_2:   float
    theta_2: float


def compute_omega_eta(
    mount: CrystalMount,
    hkl: tuple[int, int, int],
    keV: float,
    *,
    rotation_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),  # ẑ_l per paper §A
) -> ReflectionGeometry:
    """Solve paper eq A.6 / A.7 / A.8 / A.10 for ω; derive η via A.12."""
    ...


def find_reflections(
    mount: CrystalMount,
    keV: float,
    *,
    theta_range: tuple[float, float] = (0.0, np.deg2rad(16.25)),
    hkl_max: int = 5,
    eta_target: float | None = None,
    eta_tol: float = 1e-6,
) -> list[ReflectionGeometry]:
    """Enumerate accessible reflections sorted by η then θ.
    Reproduces paper Table A.2 for Al, 19.1 keV, θ ≤ 16.25°.
    Phase A: function and unit tests implemented; not invoked by any CLI or
    config loader yet. Phase B: wired into config-load validation and exposed
    via the `dfxm-find-reflections` CLI."""
    ...


def R_lab_to_image(eta: float, theta: float) -> np.ndarray:
    """Lab → imaging-detector-frame rotation, R_x(η) @ R_y(-2θ).
    At η=0 this collapses bit-identically to v2.2.0's implicit rotation."""
    return _R_x(eta) @ _R_y(-2 * theta)
```

### Modified modules (Phase A)

| Module | Change |
|---|---|
| `reciprocal_space/resolution.py` (MC LUT) | Add `eta` kw to `reciprocal_res_func`; thread through `R_lab_to_im`, `dk_in/out`, `q_imaging`. eta=0 → v2.2.0 numerics. |
| `reciprocal_space/analytic.py` (closed form) | Re-derive Gaussian closed form at η ≠ 0 per paper Appendix F. eta=0 → v2.1.0 numerics. |
| `reciprocal_space/kernel.py` (bootstrap CLI) | New TOML keys; new filename pattern; back-compat shim for legacy filename. |
| `direct_space/forward_model.py` | `forward()` accepts `eta`; new module global `fm.eta`, `fm.R_image`. |
| `pipeline.py` | `write_simulation_h5` records `eta` + `mount` provenance in master attrs. |

### Modified modules (Phase B)

| Module | Change |
|---|---|
| `pipeline.py` | New orchestration: `[[reflections]]` → iterate, share LUT + `Find_Hg`, write per-reflection `/N.1` groups. |
| `pyproject.toml` + conda recipe | New `[project.scripts]` entry: `dfxm-find-reflections`. |
| `crystal/oblique.py` | `find_reflections` activated (wired into config loader). |

### Unaffected modules

- `forward_throughput_arc` deliverables (Phase 1 fusion, Phase 2 float32/fanout
  /write_strain_provenance flag) — completely orthogonal to this arc.
- Persistent worker pool — separate arc.
- Existing test goldens — defaults to `simplified`, eta=0, bit-identical.

## TOML schema

### Phase A — single-reflection oblique

```toml
[crystal]
lattice  = "cubic"          # only value accepted in v2.3.0
a        = 4.0495e-10       # lattice parameter, m
mount_x  = [1, 0, 0]        # Miller indices aligned with lab x̂
mount_y  = [0, 1, 0]
mount_z  = [0, 0, 1]
                            # Default mount == paper Al setup (above).

[geometry]
mode = "simplified"         # "simplified" (legacy, μ varies, η=0) or "oblique" (μ=0, η=η_C)
eta  = 0.0                  # rad; only consulted when mode="oblique"

[reflection]
hkl = [-1, 1, -1]
keV = 17.0
                            # In oblique mode, omega is derived (Appendix A); not in config.
```

### Phase B — multi-reflection oblique

```toml
[crystal]
lattice  = "cubic"
a        = 4.0495e-10
mount_x  = [1, 0, 0]
mount_y  = [0, 1, 0]
mount_z  = [0, 0, 1]

[geometry]
mode = "oblique"            # multi-reflection requires oblique
eta  = 0.3531               # rad — η = 20.233°, paper Table A.2 group 1
                            # (the shared Bragg angle θ = 15.417° = 0.2691 rad
                            #  is derived from the reflections, not in config)
keV  = 19.1                 # shared across all reflections in this run

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
[[reflections]]
hkl = [1, -1, 3]
[[reflections]]
hkl = [-1, 1, 3]
                            # All entries must share (η, θ) within tolerance;
                            # validated at config load via compute_omega_eta.
```

## Back-compat contract

Precise rules (gated by `test_v220_configs_bit_identical_simplified.py`):

1. **Config has no `[geometry]` block** (every v2.2.0 config): identical to
   v2.2.0. `mode = "simplified"`, `eta = 0`. Legacy LUT filename pattern.
   Forward output bit-identical to v2.2.0.
2. **Config has `[geometry] mode = "simplified"` explicitly**: same as (1).
3. **Config has `[geometry] mode = "oblique"`**: new code path. New LUT
   filename pattern. `eta` must be present and finite; `mode="oblique"` with
   `eta=0` is a valid degenerate configuration that produces equivalent
   numerics to simplified at eta=0 but uses the new LUT cache.
4. **Config has `[crystal]` block but no `[geometry]`**: ValueError. Force
   the user to be explicit about which geometry mode they want.

## eta validation: both, validated

In oblique mode, the user provides `[geometry] eta` directly. Bootstrap also
calls `compute_omega_eta(mount, hkl, keV)` to derive the expected η₁ / η₂. If
neither matches the config's eta within `1e-6 rad`, bootstrap errors with the
diff (no auto-correct — silent footgun risk).

This is the "both, validated" mode: explicit input + derived sanity check.

## LUT cache: filename, lookup, back-compat

### Filename patterns

| Geometry mode | Pattern |
|---|---|
| `simplified` (default, legacy) | `Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz` |
| `oblique` | `Resq_i_theta{θ:.4f}rad_eta{η:.4f}rad_{keV:g}keV_{date}.npz` |

Both patterns coexist. The simplified pattern is unchanged — all v2.2.0 kernels
on disk continue to be found by lookup.

### Kernel metadata (extended)

```python
{
    # ... existing fields (Nrays, qi*_range, theta, ...) ...
    "hkl":            ...,              # already there (sub-project D)
    "keV":            ...,              # already there
    "eta":            np.float64(...),  # NEW — 0.0 in simplified mode
    "geometry_mode":  np.str_(...),     # NEW — "simplified" | "oblique"
    "lattice":        np.str_("cubic"), # NEW — v3.0.0 widens
    "a":              np.float64(...),  # NEW
    "mount_x":        np.array(...),    # NEW
    "mount_y":        np.array(...),
    "mount_z":        np.array(...),
    "omega":          np.float64(...),  # NEW — per-reflection ω; 0.0 simplified
}
```

All new fields default to legacy values when loading a v2.2.0-era LUT (missing
keys → defaults via `npz.get`). Forward continues to consume v2.2.0 LUTs
without regeneration, gated by `test_v220_kernels_loadable.py`.

### Lookup behaviour

```python
def _lookup_kernel_path(config) -> Path:
    if config.geometry_mode == "simplified":
        return _lookup_legacy(config.hkl, config.keV)
    elif config.geometry_mode == "oblique":
        # Verify by (θ, η, keV); for [[reflections]], verify the LUT covers
        # this reflection's group (not "this LUT IS this hkl's kernel").
        return _lookup_oblique(config.theta, config.eta, config.keV,
                               expected_hkl=config.hkl_or_list)
```

Failure modes (consistent with Sub-project D pattern):
- No matching LUT → `KeyError` with the `dfxm-bootstrap` re-run hint.
- Multiple matches → WARN, pick newest by mtime (today's behaviour).
- Metadata mismatch → `KeyError` with the diff.

### Optional helper: `dfxm-list-kernels`

Single Python entry point that prints all kernels in `pkl_fpath` as a table:
`(mode, θ, η, hkl|group, keV, date)`. Useful when many reflections are
bootstrapped. Optional — can be cut if CLI surface should stay narrow.

## Pipeline integration

### Phase A — single-reflection, eta-aware

```
read TOML
├── [geometry] absent?  → simplified, eta=0  (bit-identical to v2.2.0)
└── [geometry] present
    ├── mode="simplified"  → eta forced to 0; legacy LUT pattern
    └── mode="oblique"
        ├── build CrystalMount from [crystal]
        ├── compute_omega_eta(mount, hkl, keV) → (η₁, η₂, θ)
        ├── validate: |config.eta − η_i| ≤ 1e-6 rad for some i ∈ {1, 2}
        ├── pick the matching ω-solution
        └── generate_kernel(theta=θ, eta=config.eta, ...)
```

LUT generator and analytic backend both get `eta` keyword (default 0.0). At
`eta=0` math collapses to v2.2.0 numerics. **One LUT per (θ, η, keV, optics)
— independent of hkl** (paper's key insight: groups share the LUT).

Forward model:
- **MC LUT backend:** oblique geometry is baked into the LUT (LUT generator
  uses `R_lab_to_image(eta, theta)` when sampling rays). Forward queries the
  LUT transparently — no per-frame change in `forward()` beyond loading the
  (η, θ)-matched LUT instead of the legacy hkl-matched one.
- **Analytic backend:** the closed-form evaluator gets `(eta, theta)` as
  parameters. `forward()` reads them from the loaded kernel's metadata and
  forwards them to the analytic eval.

`q_hkl` stays crystal-frame (no change either way). The lab→image-detector
transform lives **inside the LUT generator / analytic eval** — not in
`forward()` itself. At eta=0 both paths collapse to v2.2.0 numerics;
bit-identical regression test gates it.

Identify: pass `eta` into the geometry. Scan axes (phi, chi, two_dtheta, z)
unchanged. Within-reflection mosa-scan unchanged by oblique geometry.

Module globals: add `fm.eta` (default 0.0) to the set of globals reset per-
invocation (already part of the worker-pool hygiene work). No `fm.R_image`
needed — the rotation is consumed inside the LUT/analytic eval, not exposed
at the forward-model level.

### Phase B — multi-reflection orchestration

Bootstrap with `[[reflections]]`:
```
validate all entries share (η, θ) → reject mixed groups with a per-hkl breakdown
generate ONE LUT for the shared (η, θ, keV)        # no per-hkl duplication
record per-entry omega + hkl in kernel metadata
```

Forward / identify with `[[reflections]]`:
```
load shared LUT once
Hg, q_hkl_unused = Find_Hg(...)   # crystal-frame; SHARED across reflections
for entry in [[reflections]]:
    fm.q_hkl  = normalize(entry.hkl)   # crystal-frame
    fm.omega  = entry.omega            # per-reflection
    fm.R_image = R_lab_to_image(η, θ)  # shared
    write per-reflection /N.1 scan group to master HDF5
```

**Efficiency point:** `Find_Hg` is invariant under reflection within an
(η, θ) group; one call serves the whole group. This is the multi-reflection
equivalent of the Phase-1 hoist (`forward_throughput_arc`).

### HDF5 layout for multi-reflection

Concrete layout for an N-reflection run with M scan modes per reflection (in
practice M=2: the strain-mosa-scan + the perfect-crystal reference):

```
master.h5
├── /reflection_001/
│   ├── 1.1/             ← strain-mosa scan, reflection #1 (BLISS-style)
│   ├── 2.1/             ← perfect-crystal scan, reflection #1
│   └── attrs            ← hkl, omega; (shared eta/theta/mount duplicated for self-describing files)
├── /reflection_002/
│   ├── 1.1/
│   ├── 2.1/
│   └── attrs
├── ...
└── /attrs               ← shared: eta, theta, mount, lattice, crystal_mode, dfxm_geo version
```

- Each `/reflection_NNN/` is a self-describing BLISS-shaped subgroup; tools
  (silx, darfix, darling) that open `/reflection_001/` get an unmodified
  single-reflection master view.
- Per-reflection detector data continues to live in a LIMA-style per-scan
  file (`reflection_NNN_lima.h5` or similar) with an external link from the
  reflection subgroup. Existing per-scan-detector pattern is unchanged.
- For **single-reflection configs** (no `[[reflections]]`), the layout is
  **unchanged from v2.2.0** — `/1.1`, `/2.1` at the top level. No
  `/reflection_NNN/` indirection. This keeps the v2.2.0 bit-identical contract.
- One run = one master file with N reflection subgroups. Matches darkmod's
  multi-reflection data layout and what's needed for paper §5.2 regression.
- Open: `dfxm-migrate-h5` (the v1.1.0 → v1.2.0 migrator) gets a v2.4.0 axis to
  optionally re-pack a sequence of single-reflection masters into one
  multi-reflection master. Useful for retrofitting old runs but cheap, can be
  cut from the v2.4.0 scope if needed.

## Error handling

Every new failure mode produces a clear, actionable message with a fix-it
command. No silent physics bugs.

### Config-load failures

| Condition | Behaviour |
|---|---|
| `[crystal]` without `[geometry]` | ValueError: "[crystal] block requires [geometry] mode to be set explicitly." |
| `mode="oblique"` without `eta` | ValueError: "[geometry] mode='oblique' requires [geometry] eta (radians)." |
| `mode="simplified"` with `eta ≠ 0` | stderr WARN: "simplified mode forces eta=0; ignoring [geometry] eta=…". |
| mount Miller indices not mutually orthogonal | ValueError with dot products. |
| mount Miller indices not integer | ValueError. |
| `lattice != "cubic"` (v2.3.0) | ValueError pointing at [[followups-cif-crystal-structures]]. |
| Mutually exclusive `[reflection]` + `[[reflections]]` | ValueError. |
| `[[reflections]]` in `simplified` mode | ValueError: "[[reflections]] requires mode='oblique' (η must be constant across reflections)." |

### Bootstrap-time failures

| Condition | Behaviour |
|---|---|
| No real ω-solution for `(hkl, mount, keV)` | ValueError: "Laue unsatisfiable for hkl=…, mount=…, keV=…. Try higher keV or different mount. Use 'dfxm-find-reflections'." |
| Config `eta` doesn't match either of (η₁, η₂) within tol | ValueError with both computed values + `dfxm-find-reflections` hint. **No auto-correct** — refuse, don't silently adopt the computed value. |
| `[[reflections]]` entries don't share (η, θ) | ValueError with per-hkl breakdown + `dfxm-find-reflections` hint. |

### Lookup-time failures

| Condition | Behaviour |
|---|---|
| No LUT matching (θ, η, keV) in oblique mode | KeyError with `dfxm-bootstrap` re-run command. |
| LUT metadata mismatch | KeyError with disk-vs-config diff. |
| v2.2.0-era LUT (no eta key) loaded into oblique mode | KeyError suggesting re-bootstrap. (Still loadable in simplified mode.) |

### Numeric guards

- Soft WARN if `|eta| > π/4` (outside the physical oblique regime, but valid math).
- No bounds check on eta — `R_x(eta)` handles any real value.
- Float32 detector + non-zero η: existing float32 tests + new η ≠ 0 fixture
  verify intensity tails stay above quantization for typical η.

## Testing strategy

Three concentric rings.

### Ring 1 — Regression (bit-identical to v2.2.0)

Hardest constraint. Gates everything else.

- `test_v220_configs_bit_identical_simplified.py` — replay every v2.2.0
  `configs/*.toml` (no `[geometry]` block) through forward and identify;
  byte-compare HDF5 against v2.2.0-tip goldens.
- `test_v220_kernels_loadable.py` — load each v2.2.0-era kernel npz under the
  new lookup; verify it resolves and feeds forward without metadata complaints.
- `test_R_lab_to_image_eta_zero_identity.py` — `R_lab_to_image(eta=0, θ)` is
  bit-identical to v2.2.0's implicit rotation, sampled at 50 θ values.
- `test_simplified_mode_explicit_matches_implicit.py` — explicit
  `[geometry] mode="simplified"` matches an omitted `[geometry]` block bitwise.

### Ring 2 — Correctness against the paper

Science gates. Phase A ships when the first four pass; Phase B when the rest.

- `test_compute_omega_eta_matches_paper_table_A2.py` — reproduce paper
  Table A.2 numerically (Al, mount=(100,010,001), keV=19.1, θ ≤ 16.25°).
  Assert (ω₁, ω₂, η₁, η₂, θ) match to `1e-3 rad`. **Canonical paper-parity test.**
- `test_find_reflections_matches_table_A2_grouping.py` — `find_reflections`
  returns the same group structure (4 reflections at η=20.233°, θ=15.417°;
  4 at the second η; …).
- `test_mc_vs_analytic_oblique_parity.py` — at 3 non-zero (η, θ) configs,
  MC LUT (200M rays) vs analytic closed-form agree to RMS ≤ 5e-4 in
  normalized intensity. Borrow `tools/mc_vs_analytic.py` infrastructure.
- `test_oblique_single_reflection_reproduces_paper_figure3B.py` — reproduce
  **paper Figure 3B** (single detector image, described in caption + §6.1).
  Setup pinned by the paper:

  | Parameter | Value | Source |
  |---|---|---|
  | Crystal | Al, lattice `a = 4.0493 Å` | §6.1 |
  | Mount | `(100)//x̂_l, (010)//ŷ_l, (001)//ẑ_l` | §6.1 |
  | Reflection | one of `(1̄1̄3), (1̄13), (113), (11̄3)` (Phase A picks one) | §6.1, Table A.2 |
  | Bragg angle | θ = 15.417° = 0.2691 rad | §6.1, Table A.2 |
  | Azimuth | η = 20.233° = 0.3531 rad | §6.1, Table A.2 |
  | Beam energy | 19.1 keV | §6.2 |
  | Dislocation | single straight edge: b ∥ [11̄0] (\|b\| = 2.86 Å); n ∥ [111̄]; t ∥ [112] | §6.1, after Borgi et al. 2024 / Poulsen et al. 2021 |
  | Sample | 25 µm³, voxel 37.878 nm, grid 265×265×27 | §6.1 |
  | Goniometer image setpoint | φ = -0.42 mrad, χ = 0.46 mrad, 2θ = 0.067 mrad | Fig 3 caption |
  | CRL | 69 lenses, separation 1600 µm, apex radius 50 µm; FWHM acceptance 0.556 mrad | §6.2 |
  | Geometry distances | d₁ = 37.826 cm, d₂ = 650.899 cm (M = 15.1) | §6.2 |
  | Beam profile | Gaussian in ẑ_l, FWHM = 236 nm; flat in x̂_l, ŷ_l | §6.2 |
  | Energy bandwidth | Δk/k std = 6×10⁻⁵ | §6.2 |
  | Vertical divergence | FWHM 0.027 mrad; horizontal 0 | §6.2 |
  | Detector | 272×272 pixels, 0.75 µm/pixel, 16-bit unsigned, 9×9 PSF (Gaussian σ=1 px) | §6.2 |
  | Noise | shot (Poisson) + thermal (Normal: µ=99.453, σ=2.317) | §6.2 |

  Compare resulting image to a vetted golden (generated once at impl time
  from the same parameters, verified by hand against the paper's Figure 3B).
  Tolerance: per-pixel RMS ≤ 5e-3 of the maximum intensity, plus a structural
  similarity gate (the dislocation contrast pattern must be visually
  recognizable to a human reviewer in failure-mode reporting).

  **Phase A ships when this passes.**
- `test_multi_reflection_one_lut_shared.py` (Phase B) — `[[reflections]]`
  config bootstraps exactly ONE LUT for the group; forward consumes it
  for all reflections.
- `test_multi_reflection_one_hg_shared.py` (Phase B) — `Find_Hg` called
  exactly once per multi-reflection run (asserted by call-count instrumentation).
- `test_multi_reflection_paper_figure.py` (Phase B) — reproduce a multi-
  reflection paper figure (e.g. the 4 reflections in Table A.2 group #1).
- `test_multi_reflection_darkmod_cross_check.py` (Phase B) — pick one
  (η, θ) group, simulate via dfxm_geo and darkmod's `TruncatedPentaGauss`;
  cross-check per-reflection image stacks within RMS ≤ 5e-3.

### Ring 3 — Integration / smoke

- `test_oblique_config_smoke_5x5.py` — tiny config (5×5 scan, 64² pixels,
  ndis=2) runs bootstrap → forward → identify in < 30 s.
- `test_multi_reflection_config_smoke_4_reflections.py` (Phase B) — same
  scale, 4 reflections, verify HDF5 master contains 4 reflection groups
  with correct `omega` attrs.
- `test_dfxm_find_reflections_cli.py` (Phase B) — exec the new CLI, assert
  Table-A.2-style row format.
- `test_fanout_oblique_compat.py` — `scripts/fanout.py` accepts oblique
  configs without fanout-side changes (config-level fan-out is orthogonal).

### Negative tests

One per error-handling row above. `pytest.raises` against the exact message
stub. Keeps error messages from rotting.

### Performance regression guard

- `test_forward_throughput_arc_unaffected.py` — replay Phase 1 + Phase 2
  benchmarks at eta=0. Same `Find_Hg` time (~224 ms) and per-scan time as
  v2.2.0 within 5%. Catches accidental slowdowns from eta plumbing.

## Implementation order (per phase)

Implementation plan will be drafted via `writing-plans` skill after this spec
is approved. High-level order:

**Phase A:**
1. New `crystal/oblique.py` module: `CrystalMount`, `R_lab_to_image`,
   `compute_omega_eta`. Pure-math tests against Table A.2 first.
2. MC LUT generator gains `eta` kw; existing eta=0 path verified bit-identical.
3. Analytic backend gains `eta` per Appendix F; MC↔analytic parity test.
4. Bootstrap CLI: new TOML keys + filename pattern + back-compat shim.
   `_lookup_kernel_path` extended; v2.2.0 kernels still loadable.
5. Forward model: `eta` flows through; eta=0 bit-identical golden gate.
6. Identify modes: `eta` flows through; eta=0 bit-identical golden gate.
7. Paper figure reproduction test green. Tag v2.3.0.

**Phase B:**
1. `find_reflections` activated; `dfxm-find-reflections` CLI shipped.
2. TOML schema `[[reflections]]` added; load-time `(η, θ)` group validation.
3. Bootstrap: shared LUT for the group.
4. Forward / identify: iterate, shared `Find_Hg`, per-reflection HDF5 groups.
5. Multi-reflection paper figure reproduction + darkmod cross-check green.
   Tag v2.4.0.

## Open questions resolved during brainstorm

- **Backends:** both (MC LUT + analytic).
- **Geometry-mode coverage:** both (simplified retained for back-compat;
  removal deferred to v3.0.0).
- **Multi-reflection meaning:** both single-reflection configs (current) and
  `[[reflections]]` (new). The default stays single.
- **Identify coverage:** yes — forward and identify both gain oblique +
  multi-reflection.
- **Lattice generalization:** config-driven cubic now; `.cif` deferred to v3.0.0.
- **Axis scope:** `eta` only this arc; goniometer `μ` deferred.
- **eta input model:** "both, validated" — user provides explicit eta, bootstrap
  cross-checks against `compute_omega_eta`.
- **LUT cache key:** (θ, η, keV, optics) — not per-hkl. Groups share the LUT.

## Out-of-scope follow-ups (link forward, not part of this spec)

- `.cif` parsing — [[followups-cif-crystal-structures]].
- Persistent worker pool — [[session-handoff-2026-05-27-phase2-float32-fanout]].
- ML dataset generation across (mount, hkl, eta) sweep — [[perf-arc-ml-training-data-goal]].
- Goniometer μ motor — v3.0.0 or later.
- Drop `simplified` geometry mode — v3.0.0.
