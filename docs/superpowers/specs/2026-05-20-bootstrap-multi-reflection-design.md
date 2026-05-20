---
title: Bootstrap multi-reflection + Bragg validity (sub-project A)
status: approved
date: 2026-05-20
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - session_handoff_2026-05-20.md (auto-memory, Q1–Q5 captured there)
  - src/dfxm_geo/reciprocal_space/kernel.py (current generate_kernel + cli_main)
  - src/dfxm_geo/direct_space/forward_model.py (pkl_fn target, line 57)
  - configs/default.toml ([reciprocal] schema source)
  - tests/test_kernel_cli.py (existing test home for kernel + cli_main)
---

# Bootstrap multi-reflection + Bragg validity (sub-project A)

## Purpose

Today `dfxm-bootstrap` hardcodes the Al (-1, 1, -1) reflection at 17 keV.
The Bragg angle is computed by `_default_theta_al_111(keV=17)` inside
`kernel.py` and consumed as the `theta` default to `generate_kernel`. To
support multi-reflection simulations (the next pipeline-features
sub-project, D), bootstrap must accept arbitrary `(hkl, keV)` from TOML,
validate the Bragg geometry, and write a kernel npz whose filename
encodes the reflection so downstream forward / identify can look it up.

This is sub-project A of the pipeline-features arc; sub-project D
generalises forward / identify to multi-reflection lookup against
filenames written by this spec. F (later) flips the default config to
a "simple" reflection. Both depend on the work specified here.

Scope: extend `kernel.py` and `cli_main`, update `pkl_fn` in
`forward_model.py` to the new default-reflection filename, extend
`configs/default.toml` with the new keys. ~150–200 LOC including tests.

## Decisions (Q1–Q5)

Captured via brainstorming (`superpowers:brainstorming`) on 2026-05-20;
all five were explicitly answered before approach selection.

- **Q1 — Material scope: Al only.** Lattice parameter
  `a = 4.0495e-10 m` stays hardcoded. Only `(h, k, l)` + `keV` are
  exposed in TOML. Other materials are out of scope for this sub-project.
- **Q2 — Kernel filename: encode params.** Bootstrap writes
  `Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz`. Multiple kernels coexist
  on disk; D will look them up by name.
- **Q3 — Absent keys: soft default with WARN.** When `[reciprocal]`
  lacks `hkl`/`keV`, default to (-1, 1, -1) @ 17 keV and emit a warning
  to stderr. Soft (not hard) was picked for iteration speed; the
  no-back-compat constraint ([[dfxm-no-backcompat-constraint]]) means
  flipping to hard-error is a later cheap change if warranted.
- **Q4 — Validity check: minimal hard checks + range warnings.**
  Hard `ValueError` on `hkl == (0,0,0)`, `keV ≤ 0`, `sin θ > 1`. Soft
  WARN on `θ < 5°` or `θ > 85°`. Always print the computed θ in degrees
  on stdout for sanity.
- **Q5 — TOML schema.** `hkl = [-1, 1, -1]` (array) + `keV = 17.0`
  (scalar), inside the existing `[reciprocal]` block.

## Approach: "Surgical" (a)

Single change point: `src/dfxm_geo/reciprocal_space/kernel.py`.

Approach (b), a dedicated `Reflection` dataclass, was deferred — its
benefits depend on D's requirements being visible, which they aren't yet.

## Architecture

Three additions inside `kernel.py`:

1. **`_validate_reflection(hkl: tuple[int, int, int], keV: float, a: float) -> float`**
   New helper. Computes θ from `d_hkl = a / sqrt(h² + k² + l²)` and
   `λ = hc / E` (with `hc = 1.239841984e-9 m·keV`), applies the hard
   checks and soft warnings from §"Error handling", returns θ in radians.

2. **`_build_kernel_filename(hkl: tuple[int, int, int], keV: float, date: str) -> str`**
   New pure helper. Returns the per-reflection basename:
   `f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz"`. `:g` drops trailing
   zeros (`17.0` → `17`); negative indices render naturally
   (`-1` → `h-1`). Pure → unit-testable in isolation.

3. **`cli_main` changes** — reads new `hkl` + `keV` keys from
   `[reciprocal]`, pops them before unpacking, calls
   `_validate_reflection` for θ, calls `_build_kernel_filename` for the
   output basename, passes `theta=θ` + `output_path=<full path>` to
   `generate_kernel`. The signature-introspection unknown-key check is
   extended to allow `hkl` and `keV` (they are cli_main-scope, not
   `generate_kernel` kwargs) — concretely:

   ```python
   valid_recip_keys = (valid_params - {"date", "output_path"}) | {"hkl", "keV"}
   ```

`_default_theta_al_111` stays as the Q3 soft-default fallback when
`hkl`/`keV` are absent. `generate_kernel` itself is unchanged — it still
takes `theta` and is reflection-agnostic.

Side effect outside `kernel.py`: `pkl_fn` in
`src/dfxm_geo/direct_space/forward_model.py:57` becomes the new
filename pattern instantiated for (-1, 1, -1) @ 17 keV, post-bootstrap
regen. Generalising forward to look up arbitrary hkl is sub-project D.

## Data flow

```
[reciprocal] block in TOML
    │
    ├── hkl present? keV present?
    │     ├── both          → pop, θ = _validate_reflection(hkl, keV, a=4.0495e-10)
    │     ├── exactly one   → ValueError("must provide both `hkl` and `keV`, or neither")
    │     └── neither       → WARN stderr; θ = _default_theta_al_111(17);
    │                         hkl = (-1, 1, -1), keV = 17 (used for filename only)
    │
    ├── theta ALSO in TOML?  → ValueError("cannot specify both `theta` and `hkl`+`keV`; pick one")
    │
    ├── Build output_path:
    │     if --output given           → use it (existing behavior)
    │     else                        → <pkl_fpath>/_build_kernel_filename(hkl, keV, date)
    │
    ├── Remaining [reciprocal] keys (after popping hkl/keV)
    │     → existing unknown-key check against generate_kernel signature, unchanged
    │
    └── generate_kernel(theta=θ, output_path=<built>, **remaining_kwargs)
            → MC run, writes npz with bundled scalar params (theta included)
            → returns Path
```

Key invariants:

- `hkl` + `keV` never reach `generate_kernel`. cli_main-scope only.
- `θ` is the boundary value — the only physics quantity crossing into
  the MC core.
- The npz bundles `theta`, not `hkl`. Adding `hkl` to the npz is
  sub-project D's job.

## Filename pattern

`Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz`

- `{h}`, `{k}`, `{l}`: integer hkl values, str-cast. Negatives render as
  `-N` (e.g. (-1, 1, -1) → `h-1_k1_l-1`).
- `{keV}`: formatted with `:g` — trailing zeros dropped. `17.0` → `17`,
  `17.5` → `17.5`.
- `{date}`: reuses the existing `YYYYmmdd_HHMM` stamp from
  `datetime.now().strftime("%Y%m%d_%H%M")`. Single stamp; the existing
  variable already encodes both date and time.

Example: `Resq_i_h-1_k1_l-1_17keV_20260520_2100.npz`.

Filename-safety constraints (enforced in `_validate_reflection`):

- hkl components are `int` (hard check). Floats are rejected — they
  are non-physical and would render ugly in filenames.
- keV is `> 0` (hard check; also a physics constraint).
- No filesystem-unsafe characters even for large negative indices.
  Verified on Windows + Linux.

## Error handling

Hard errors raise `ValueError`. Soft warnings go to `sys.stderr` via
`print(..., file=sys.stderr)`, matching the existing `cli_main` style
(no `logging` import — YAGNI).

### Hard errors in `_validate_reflection` (structural → physics order)

- `len(hkl) != 3` → `"hkl must have 3 components, got {n}."`
- any non-int component →
  `"hkl components must be int, got {hkl}."`
- `hkl == (0, 0, 0)` →
  `"hkl=(0,0,0) is not a valid reflection (no diffraction)."`
- `keV <= 0` → `"keV must be > 0, got {keV}."`
- `sin θ > 1` →
  `"Bragg condition unsatisfiable: λ={lam_A:.4f} Å, 2·d_hkl={two_d_A:.4f} Å, sin θ = {sin_theta:.4f} > 1 for hkl={hkl} at {keV} keV. Pick a lower-order reflection or higher beam energy."`

### Hard errors in `cli_main` (input coherence)

- only one of `hkl`/`keV` present →
  `"must provide both \`hkl\` and \`keV\`, or neither."`
- `theta` + (`hkl` or `keV`) →
  `"cannot specify both \`theta\` and \`hkl\`+\`keV\`; pick one."`

### Soft warnings in `_validate_reflection`

- `θ < 5°` →
  `"warning: θ = {θ_deg:.2f}° is very low (< 5°); reflection unusual but valid."`
- `θ > 85°` →
  `"warning: θ = {θ_deg:.2f}° near back-reflection (> 85°); reflection unusual but valid."`

### Soft warning in `cli_main`

- `hkl`/`keV` both absent →
  `"warning: [reciprocal] has no \`hkl\`/\`keV\`; defaulting to Al (-1, 1, -1) @ 17 keV."`

### Success-path echo (always)

On successful validation, before the MC run, print to stdout:

```
reflection: hkl=(-1, 1, -1), keV=17 → θ = 9.4517°
```

`keV` uses `:g` formatting (matching the filename, so `17.0` echoes as
`17` and `17.5` as `17.5`). θ in degrees is printed to four decimal
places. Followed by the existing `wrote {path}` line after
`generate_kernel` returns.

Error messages deliberately include computed intermediates (λ, 2d,
sin θ) so the user sees *why* a reflection is invalid, not just *that*
it is.

## Testing

Home: extend `tests/test_kernel_cli.py`. ~15–18 new tests, ~+150–200
LOC test code.

### `TestValidateReflection` — pure-function unit tests

- Hard error (one test each): `len(hkl) != 3`, non-int component,
  `hkl=(0,0,0)`, `keV <= 0`, sin θ > 1 (e.g. hkl=(10,10,10) @ 1 keV).
- Soft warnings (via `capsys`): θ < 5°, θ > 85°.
- Returns θ bit-equal to `_default_theta_al_111(17)` for
  (-1, 1, -1) @ 17 keV.
- Returns expected θ for one cross-check reflection (e.g. (2, 0, 0) @
  17 keV vs hand-computed value).

### `TestBuildKernelFilename` — pure-function unit tests

- Basic: `(-1, 1, -1), 17.0, "20260520_2100"` →
  `"Resq_i_h-1_k1_l-1_17keV_20260520_2100.npz"`.
- `:g` format: `17.0` → `"17keV"`, `17.5` → `"17.5keV"`, `17` (int) →
  `"17keV"`.
- Mixed signs: `(2, -1, 3), 8, ...` → `"...h2_k-1_l3_8keV..."`.

### `TestCliMainMultiReflection` — integration via `cli_main`

Monkey-patches `generate_kernel` to skip the MC entirely; verifies
wiring, θ, filename, exit codes only.

- Happy path: TOML with `hkl=[-1,1,-1] keV=17` → exit 0, kernel
  written at expected path, computed θ echoed on stdout.
- No `hkl`/`keV` → exit 0, WARN on stderr, default-pattern filename.
- `hkl` only (no `keV`) → exit 1 + clear stderr message.
- `keV` only (no `hkl`) → exit 1 + clear stderr message.
- `theta` + `hkl` together → exit 1 + clear stderr message.
- `hkl=(0,0,0)` → exit 1 (ValueError surfaced cleanly, not as raw
  traceback).
- Unsatisfiable Bragg → exit 1 with λ/2d/sin θ in stderr.

### `TestPklFnRegression`

Read `dfxm_geo.direct_space.forward_model.pkl_fn`, regex-match
`r"Resq_i_h-1_k1_l-1_17keV_\d{8}_\d{4}\.npz"`. Catches accidental
reverts.

### Default config update

Extend the existing `TestDefaultConfigReciprocalBlock` to assert
`configs/default.toml` has the new `hkl = [-1, 1, -1]` and `keV = 17.0`
keys.

### Performance discipline

All cli_main integration tests monkey-patch `generate_kernel` to skip
the ~50 s MC run; pure-function tests run in well under 0.1 s each.
Total added wall-clock to the suite: < 1 s.

## Out of scope (deferred to later sub-projects)

- Multi-reflection lookup in forward / identify (sub-project D).
- Adding `hkl` to the kernel npz metadata (sub-project D).
- Materials other than Al (lattice parameter table). Sub-project D may
  also defer this; revisit when concrete need arises.
- Default-config flip to "simple" reflection (sub-project F, v2.0.0).
- Seeding `Find_Hg` to make forward output bit-deterministic — adjacent
  follow-up captured in the session handoff, optional fold-in.

## Operational follow-ups

After this sub-project lands and bootstraps a new (-1, 1, -1) @ 17 keV
kernel under the new filename pattern:

- Run `dfxm-bootstrap --config configs/default.toml` once on the laptop
  and once on the cluster to produce the new
  `Resq_i_h-1_k1_l-1_17keV_<date>.npz` files.
- Update `pkl_fn` in `forward_model.py:57` to match the produced
  filename. Same operational pattern as the PR #6 post-bootstrap regen
  step.

## Version target

Lands as part of the v1.2.0 release alongside sub-projects D, B, C, E.
F (default config flip) is a separate v2.0.0-flavored breaking change.
PyPI publish is held until v1.2.0 ships.
