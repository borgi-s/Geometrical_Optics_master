# Identify fan-out (cluster) + colleague-ready notebook — design

Date: 2026-06-08
Branch: `feature/identify-fanout-and-notebook` (worktree, forked from `origin/main` = v2.3.1)

## Goal

Two shippable outcomes, in order:

1. **Make `dfxm-identify` fan out on the cluster.** Today the forward pipeline
   fans out (`scripts/fanout.py`, `lsf/fanout.bsub`) but identify does not, and
   the identify array job is silently broken.
2. **Polish the ML identification tutorial notebook** so it runs top-to-bottom
   on a fresh machine and can be sent to the ESRF_DTU ML collaborator.

## Out of scope (explicit)

- `.cif` / non-cubic crystal support (v3.0.0 arc — separate).
- An oblique-angle section in the notebook (belongs with the v2.4.0 work).
- Anything touching the in-flight `feature/forward-context-refactor` (v2.4.0).
- Pushing / opening a PR / tagging a release — local feature branch only until
  the user says otherwise.

## The gap (evidence)

- `cli_main_identify` (`src/dfxm_geo/pipeline.py:2241`) accepts only
  `--config/--output/--mode`. There is **no per-task seed override**; the RNG
  seed comes solely from the config's `noise.rng_seed`
  (`NoiseConfig.rng_seed`, `pipeline.py:571`; consumed at `pipeline.py:1833`).
- `lsf/identify_array.bsub:61` runs `dfxm-identify --config … --output …`
  across **100 array tasks** and its comment (line 57) claims each task draws a
  non-overlapping slice "offset by LSB_JOBINDEX" — but it passes no seed. **All
  100 tasks currently produce identical samples.**
- `scripts/fanout.py` is **forward-only**: `_FORWARD_PREFIX` (line 35) imports
  `cli_main`, and `_default_runner` (line 98) hardcodes `--no-postprocess`
  (a forward-only flag). There is no way to fan out identify in-node.

## Work-stream 1 — identify fan-out

### 1a. `--seed` on `dfxm-identify`
Add `--seed INT` (default `None`) to `cli_main_identify`. When provided, it
overrides the seed that flows into `run_identification`'s RNG
(`noise.rng_seed`) via `dataclasses.replace`, then re-validates. Leave
`dfxm-forward` unchanged — its fan-out already works via per-shard config
seeds.

Determinism contract (must be tested): same `--seed` → byte-identical output;
different `--seed` → different draws. The seed override must reach the same
`np.random.default_rng(rng_seed).spawn(...)` split documented at
`pipeline.py:1977`.

### 1b. `fanout.py --mode {forward,identify}`
- Add `--mode` (default `forward`, preserving current behaviour).
- Parameterize the child-interpreter prefix: identify → `cli_main_identify`.
- The `--no-postprocess` flag is **forward-only** — append it only in forward
  mode (identify has no such flag).
- Keep the injected `runner` seam, `worker_env` BLAS/numba pinning, and
  per-config batch-resilience (rc=-1 on runner exception) intact.
- Per-config distinct seeds for in-node identify fan-out come from the shard
  configs themselves (each `*.toml` carries its own `noise.rng_seed`), exactly
  as the forward sweep already works. `--seed` is the mechanism for the LSF
  *array* job (one config, many tasks); fan-out is the mechanism for *many
  configs*. Both must work after this arc.

### 1c. LSF templates
- `lsf/identify_array.bsub`: actually wire `--seed "${LSB_JOBINDEX}"` so each
  array task draws a distinct, deterministic slice (matching the comment's
  promise). Clarify the per-task `n_samples` ↔ total-samples relationship.
- `lsf/fanout.bsub`: add a `MODE` knob (default `forward`) passed through as
  `--mode`, so in-node identify fan-out is one `bsub <` away. Document it.

### 1d. Tests (TDD — write first)
- Extend `tests/test_fanout.py`: identify mode builds the
  `cli_main_identify` command and omits `--no-postprocess`; forward mode is
  unchanged (regression). Use the existing `runner` injection seam — no real
  subprocesses.
- A `cli_main_identify --seed` determinism test (small `multi` config, 5×5
  grid per the smoke-scale-down rule): same seed reproducible, different seed
  differs.
- Full suite stays green; `mypy src/dfxm_geo/` reports 0 errors; pre-commit
  hooks pass.

## Work-stream 2 — colleague-ready notebook

Target: `examples/identification_ml_tutorial/dfxm_identify_ml_tutorial.ipynb`
runs top-to-bottom on a clean machine and is honest about the (now working)
fan-out path.

- **Preflight cell**: assert `dfxm_geo` importable + print version; create
  `kernel/` before use; fail with a clear install hint if the package is
  missing.
- **Portable kernelspec**: don't pin a machine-specific kernel name.
- **Regenerate `out_demo/` locally at a small grid** (5×5, smoke-scale-down)
  so HDF5 external-link paths resolve on the runner's machine; harden
  `read_image` to fall back to the per-scan file if a master external link
  fails to resolve.
- **Rewrite the "scale to 100k" section** to use the now-working identify
  fan-out: `python scripts/fanout.py --mode identify …` for in-node, and the
  fixed `lsf/identify_array.bsub` for array jobs. Remove the current "honest
  gap / no fan-out launcher yet" caveat.
- **Lint-clean** so pre-commit (ruff/ruff-format/nbstripout) passes; outputs
  are stripped on commit (standard hygiene).
- **Verify headless**: `jupyter nbconvert --to notebook --execute` exits 0 on
  the worktree venv.

## Verification gate (whole arc)

- `python -m pytest -q` green (worktree venv).
- `mypy src/dfxm_geo/` → 0 errors.
- pre-commit hooks pass on every commit (no `--no-verify`).
- Notebook executes headless to completion.
- Parallel spec-review + code-quality-review on work-stream 1 before it's
  considered done.

## Process

Subagent-driven, in-session (not a dynamic Workflow): a Sonnet TDD implementer
for work-stream 1 → two parallel Sonnet reviewers (spec + code-quality) →
a Sonnet implementer for the notebook. Haiku for the trivial LSF/text edits.
Opus (orchestrator) does integration, the verification gate, and final review.
