# M3 — Multi-reflection sweeps (v2.6.0) — design

**Status:** draft, scaffolded autonomously 2026-06-10; **one open physics
decision for Sina (§3)**, everything else proceeds with the defaults below.
**Supersedes:** the Phase B section of
`2026-05-28-multi-reflection-oblique-angle-design.md` (Phase A shipped as
v2.3.0; this document is Phase B re-baselined on v2.5.1 and the June-2026
roadmap §4).
**Baseline:** v2.5.1 (`30fabe2`), branch `feature/m3-multi-reflection-sweeps`.
**Plan of record:** `roadmap202606audit.md` §4 (Goal 3).

## 1. Goal

One TOML with a `[[reflections]]` list (e.g. 111 + 200 + 220 on Al) runs
forward and identify end-to-end, one resolution kernel per unique
(θ, η, keV) geometry, with per-reflection g·b visibility labels in the
identify HDF5, and the sweep generators + fanout gaining a reflection axis.
This is the roadmap's Definition of Done, verbatim.

## 2. What changed since the 2026-05-28 spec was written

The old Phase B section assumed v2.2.0-era architecture. Re-surveyed at
v2.5.1 (2026-06-10):

- **`reflection_theta_if_oblique` is gone** (deleted by the v2.5.0
  ForwardContext refactor). Geometry now flows through
  `run_theta(config)` → `fm.build_forward_context(theta, res, hkl)` →
  frozen `ForwardContext` (`forward_model.py:88`). A per-reflection loop is
  now just "build N contexts", with no global-state juggling. This makes
  Phase B *substantially cheaper* than the old spec assumed.
- **`q_hkl` only enters at the projection step** —
  `precompute_forward_static` computes `base_qc = (Us @ Hg @ q_hkl)`
  (`forward_model.py:697`). The Hg field itself (`find_hg_population`,
  `find_hg_scene`) never sees Miller indices. Under the recommended ω
  approach (§3), Hg is reflection-invariant and shareable across a run.
- **Kernel infrastructure is already multi-reflection-shaped**: oblique
  kernels are keyed by (θ, η, keV) — not hkl — in both the filename pattern
  and `_lookup_kernel_path` (`forward_model.py:494-553`), and the npz
  metadata already carries `eta`/`omega`/`geometry_mode`/mount
  (`kernel.py:300-335`). What's missing is only the *loop* (bootstrap one
  kernel per unique geometry) and the *manifest*.
- **`find_reflections` is implemented and paper-tested** (Table A.2 group
  reproduction, `tests/test_oblique_find_reflections.py`) but wired to
  nothing — no CLI, no config loader (`oblique.py:241-288`).
- **The fanout pool needs zero changes**: it is axis-agnostic (consumes
  TOML files), and `pipeline._KERNEL_CTX_CACHE` (keyed by kernel path)
  already amortizes per-(θ,η,keV) kernel loads across a warm worker.
- The roadmap reframed the goal from "common-volume imaging bundles" to
  "**sweeps**": the `[[reflections]]` list may span multiple (θ, η) groups
  (111+200+220 do *not* share a Bragg angle); kernel sharing is
  opportunistic per group rather than a validation requirement. The old
  spec's "reject mixed groups" rule is superseded (§5).

## 3. THE open decision — ω handling (for Sina)

How much of the per-reflection crystal reorientation ω enters the image
formation? This was the exact question the 2026-05-29 Phase-B brainstorm
paused on (the "end goal" question, never resumed; the roadmap's step-3
"physics review step" is the same item). Options, costs measured against
the v2.5.1 architecture:

| | What changes per reflection | Hg shared? | Cost | Faithfulness |
|---|---|---|---|---|
| **A** | `q_hkl` only; ω recorded in provenance | ✅ | trivial | Q direction right per reflection; image does not reorient with the crystal |
| **B′ (recommended)** | `q_hkl` + ω applied at the projection step (`base_qc = Us_eff @ Hg @ q_hkl` with `Us_eff = R_z(ω) @ Us`), shared voxel↔pixel mapping | ✅ (Hg untouched; only the cheap projection re-runs) | small — one rotation at `precompute_forward_static` level | Faithful per-reflection Q; image contrast reorients correctly to first order; voxel grid shared (good for ML labels and any future reconstruction) |
| **full-ω** | Ray grid itself rotates (`rl`, beam path) per reflection | ❌ (Hg per-reflection) | large — re-derives the biggest compute stage N×, and is essentially re-implementing darkmod's 3D projection on CPU | Paper-faithful image appearance incl. projection geometry |

**Recommendation: B′.** Rationale: (a) the roadmap frames M3 as *sweeps*
for identify libraries and g·b labels — B′ gives exactly-right per-pixel Q
and visibility physics, which is what the labels need; (b) M1 just spent an
arc getting Hg from 61-70 % down to ~13 % of identify — full-ω multiplies
it back by N_reflections; (c) darkmod already exists for paper-faithful
multi-reflection imaging ([[fig3-repro-darkmod-is-the-paper-model]]).
In the 2026-05-29 brainstorm Sina initially picked full-ω, but the
discussion stopped on whether that goal even belongs in dfxm_geo; the
roadmap written afterwards points at sweeps. **Default if no answer: B′,
with the per-reflection adjustment isolated in one function
(`_reflection_projection(ctx, reflection)`) so a later full-ω upgrade is a
contained change, not a rewrite.**

Everything in §4-§9 is independent of this choice except the orchestrator
physics step, which the implementation plan marks as decision-blocked.

## 4. Non-goals

- **B2 reconstruction** (per-pixel COM → Q → least-squares F per voxel,
  paper §5): separate arc, consumes M3's ≥3-reflection datasets.
- **Non-{111} Burgers / slip systems**: Goal 4 (v3.0.0). Sweeping q_hkl to
  200/220 changes the diffraction vector, **not** the slip systems — the
  FCC {111} machinery stays valid for any reflection. Keep this boundary
  explicit (roadmap §4 note).
- **Full-ω faithful imaging** unless Sina picks it in §3.
- Goniometer μ as a free axis; `.cif`; dropping `simplified` mode — v3.0.0.

## 5. Config schema

### Explicit list

```toml
[reciprocal]
keV = 17.0                  # shared across all reflections
# hkl = ...                 # FORBIDDEN when [[reflections]] present (ValueError)
backend = "mc"              # everything else unchanged

[geometry]
mode = "oblique"            # [[reflections]] requires oblique (unchanged rule)
# eta = ...                 # optional when [[reflections]] present (see below)

[crystal]                   # mount keys as in v2.3.0
lattice = "cubic"
a       = 4.0495e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[[reflections]]
hkl = [1, 1, 1]
[[reflections]]
hkl = [2, 0, 0]
eta = 0.4012                # optional per-entry override, radians
omega_solution = 2          # optional: 1 (default) or 2
[[reflections]]
hkl = [2, 2, 0]
```

Resolution rules, per entry:

1. `compute_omega_eta(mount, hkl, keV)` → (ω₁, η₁, θ) / (ω₂, η₂, θ).
   No real solution → ValueError with the `dfxm-find-reflections` hint.
2. `eta` absent → use η of the chosen `omega_solution` (default solution 1).
   `eta` present → must match one solution within 1e-6 rad ("both,
   validated", unchanged from Phase A); the matching solution fixes ω.
3. A top-level `[geometry] eta` with `[[reflections]]` acts as a *default*
   for entries that omit `eta` and is validated per-entry — entries whose
   solutions don't include it fail loudly (no silent reassignment).
4. Entries dedup-grouped by (θ, η) within 1e-6 rad for kernel sharing.
   Mixed groups are **allowed** (roadmap supersedes the old reject rule);
   the resolved grouping is logged and recorded in the kernel manifest.

Resolved entries become frozen `ReflectionRun(hkl, keV, eta, theta, omega,
kernel_path)` records on a new `config.reflections: list[ReflectionRun]`
(empty list = single-reflection config, behavior unchanged).

### Auto-find group (brainstorm decision 2: "both")

```toml
[reflections_auto]
eta_target = 0.3531         # radians; group selector
theta_max  = 0.2837         # optional, default paper 16.25°
hkl_max    = 5              # optional
```

Expands via `find_reflections(mount, keV, eta_target=…)` into the same
`ReflectionRun` list (one shared kernel by construction). Mutually
exclusive with `[[reflections]]`.

### Error table (additions to the Phase A table)

| Condition | Behaviour |
|---|---|
| `[[reflections]]` with `[reciprocal] hkl` | ValueError: mutually exclusive |
| `[[reflections]]` with `mode != "oblique"` | ValueError (sweep simplified-mode reflections via the generators instead) |
| `[[reflections]]` and `[reflections_auto]` both present | ValueError |
| Empty `[[reflections]]` | ValueError |
| Entry with no real ω solution | ValueError + per-hkl breakdown + CLI hint |
| `eta` matching neither solution | ValueError with both computed values (no auto-correct) |
| `[reflections_auto]` matching zero reflections | ValueError + suggestion to run `dfxm-find-reflections` |

## 6. Kernel management

- `dfxm-bootstrap` loops the resolved `ReflectionRun` list, dedups by
  (θ, η, keV) within tolerance, and generates/looks-up **one kernel per
  unique geometry** (existing oblique filename pattern, unchanged).
- Writes `kernel_manifest.toml` next to the output kernels: one row per
  reflection → (hkl, ω, η, θ, kernel filename, sha256, group id). Forward/
  identify do *not* require the manifest (lookup stays by-(θ,η,keV)); it is
  provenance + the human-readable record of the grouping.
- `--if-missing` semantics extend naturally: skip groups whose kernel
  already resolves.

## 7. Orchestration

Per-reflection loop wraps at the `run_simulation` / `run_identification`
level (survey confirmed this is the natural seam): for each
`ReflectionRun`, resolve kernel → `build_forward_context` → iterate scans →
write. Phase-1 implementation does the simple N-pass loop (correctness
first); a follow-up perf item hoists the Hg stage across reflections
(valid only under A/B′ — Hg is reflection-invariant; the fused-engine call
sites are `find_hg_scene` in the three identify iterators and the
`Hg_provider` closures in `_run_simulation_inner`).

### HDF5 layout — decision #2 (default chosen, flag if you disagree)

Two candidates surveyed:

1. **Old-spec layout**: one master with `/reflection_NNN/1.1` subgroups.
   Requires `MasterWriter.add_scan` to grow a group-prefix parameter, and
   top-level tools (darfix; darling is already external-link-blind) won't
   find scans at the root.
2. **One master per reflection + super-master** *(chosen default)*: each
   reflection writes a completely standard single-reflection master into
   `reflection_NNN/` (so silx/darfix/darling work on it unchanged, today),
   plus a thin top-level `dfxm_geo_multi.h5` carrying ExternalLinks
   `/reflection_NNN → reflection_NNN/dfxm_geo.h5::/` and the shared attrs
   (mount, keV, per-reflection hkl/ω/η/θ table, g·b matrix). Zero changes
   to the existing writers; the multi-reflection layer is purely additive.

Rationale for 2 over the brainstorm-approved 1: the writer architecture is
one-master-per-call (`write_simulation_h5`/`write_identification_h5`), and
option 2 preserves every existing tool contract for free. Single-reflection
configs remain byte-identical to v2.5.1 either way.

### Per-reflection provenance

Each reflection's master gains `hkl`, `omega`, `eta`, `theta`,
`reflection_index`, `n_reflections`, `multi_master` attrs alongside the
existing geometry attrs. The super-master records the full resolved table.

## 8. g·b visibility labels

- Factor the inline criterion (`pipeline._passes_invisibility`,
  `pipeline.py:1575`) into a public helper
  `crystal/burgers.py: gb_visibility(q_hkl, b) -> float` (the normalized
  |cos∠(G,b)|) + threshold wrapper; the existing call sites delegate
  (bit-identical refactor).
- Identify HDF5, multi-reflection runs: write per-scan
  `gb_cos` (float) and `gb_visible` (bool, against
  `invisibility_threshold_deg`) per dislocation per reflection — the
  cross-reflection visibility pattern is the scientific payload (and the
  Goal-5 tutorial figure).
- `exclude_invisibility` semantics in multi-reflection runs (new feature,
  no back-compat constraint): a sweep config is excluded only if invisible
  in **all** reflections; otherwise it is kept and labeled. Single-
  reflection behavior unchanged.
- Physics regression test: an edge dislocation with b ⊥ g for one
  reflection of the list and b·g ≠ 0 for another must show contrast ratio
  ≫ 1 between the two reflection images (the classic invisibility
  criterion, asserted at smoke scale).

## 9. CLI + sweep tooling

- **`dfxm-find-reflections`** (also closes roadmap G1.3): new
  `[project.scripts]` entry → `crystal/oblique.py:cli_main`. Reads a config
  TOML (mount + keV [+ optional eta-target/θ-range/hkl-max flags]), prints
  the Table-A.2-style table: hkl, θ, η₁, ω₁, η₂, ω₂, group id. Acceptance
  test = paper Table A.2 reproduction (already unit-tested at function
  level; the CLI test execs the entry point). Mirror the entry point in the
  conda-forge recipe **at release time** (CLAUDE.md recipe-sync rule —
  Windows launcher lesson).
- **`gen_sweep_configs.py` / `gen_identify_sweep_configs.py`**: gain
  `--hkl-list` (e.g. `--hkl-list "1,1,1;2,0,0;2,2,0"`) and `--keV`;
  reflection becomes one more Cartesian axis in the emitted single-
  reflection TOMLs (filenames gain an `hkl{h}{k}{l}` token). This is the
  *cross-config* sweep path (works in simplified mode too, today, since
  each emitted config is an ordinary single-reflection config). The
  generators also accept `--reflections-toml` to emit bundled
  `[[reflections]]` configs once the in-config path lands.
- **`fanout.py`**: no changes (verified axis-agnostic). The pool's
  per-worker `_KERNEL_CTX_CACHE` covers per-(θ,η,keV) kernel amortization.

## 10. Testing

Back-compat ring (gating): configs without `[[reflections]]` /
`[reflections_auto]` are **bit-identical to v2.5.1** — existing goldens +
full suite must stay green with the failure set unchanged
([[preexisting-test-failures-2026-05-28]] discipline: compare failure SET,
not count; known stale: `test_render_readme_examples_smoke`).

Unit ring: schema parsing + every error-table row (`pytest.raises` on
message stubs); (θ,η) grouping/dedup incl. tolerance edges;
`ReflectionRun` resolution with eta defaults/overrides/solution choice;
`[reflections_auto]` expansion equals `find_reflections` output;
`gb_visibility` refactor bit-identity.

Integration ring (smoke scale, [[feedback-smoke-test-scale-down]]: 5×5
scans, 64² pixels, ndis ≤ 2): 2-reflection forward run → super-master +
2 standard masters, attrs correct; same-group pair bootstraps exactly ONE
kernel (count files); mixed-group triple bootstraps the right number;
identify 2-reflection run with gb labels present; g·b invisibility physics
test (§8); `dfxm-find-reflections` CLI table format; manifest content.

Deferred (explicitly): darkmod multi-reflection cross-check (old spec's
`test_multi_reflection_darkmod_cross_check`) — only meaningful under
full-ω; revisit after the §3 decision.

**No full-scale runs in this arc until the LSF cluster row lands** (M1 DoD
pending); all benchmarks/quantitative sweeps wait.

## 11. Open items for Sina (summary)

1. **§3 ω handling** — A / B′ (recommended, default) / full-ω.
2. **§7 HDF5 layout** — option 2 chosen as default (deviates from the
   2026-05-29 brainstorm's option-1 sketch, rationale given); veto if the
   single-file master matters to you.
3. Minor: `exclude_invisibility` all-reflections semantics (§8 default).

## 12. Out-of-scope follow-ups (link forward)

- B2 reconstruction arc (consumes ≥3-reflection datasets from this work).
- Hg hoist across reflections (perf, after correctness; only under A/B′).
- `dfxm-migrate-h5` re-pack of old single-reflection masters into a
  super-master (cheap, cut from scope unless asked).
- Goal-5 tutorial figure: g·b invisibility across a reflection sweep
  (`04_oblique_and_reflections.ipynb`).
