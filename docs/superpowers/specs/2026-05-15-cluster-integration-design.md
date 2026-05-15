---
title: Cluster integration design (v1.0)
status: approved
date: 2026-05-15
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md
  - GitHub Release v0.9.0 (https://github.com/borgi-s/Geometrical_Optics_master/releases/tag/v0.9.0)
  - cleanup_session_state.md (auto-memory) Round 22
  - DTU HPC docs (https://www.hpc.dtu.dk/) — LSF scheduler
  - ESRF SLURM docs (https://gitlab.esrf.fr/silx/training/jupyter-slurm) — SLURM scheduler
---

# Cluster integration design (v1.0)

## Purpose

This document specifies the v1.0 cluster integration push: the work
required to take the v0.9.0 baseline (`main` HEAD `9fca38d`) to a
release that lets a fresh-clone user run production-scale `dfxm-forward`
and `dfxm-identify` simulations on DTU HPC (LSF) and ESRF (SLURM)
clusters with no tribal knowledge. v1.0 ships when the work in this
spec lands and triggers the second Zenodo deposit.

Decisions captured via interview ("grill") on 2026-05-15; each
subsection cross-references the question that produced it (Q1–Q6).

## Correction to the finalization spec

The cleanup-finalization spec
(`docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md`)
asserts "Linux SLURM clusters (DTU HPC + ESRF)." That's wrong on DTU:
**DTU HPC uses LSF** (Load Sharing Facility, `bsub` / `#BSUB`
directives, per `hpc.dtu.dk`), not SLURM. ESRF uses SLURM. This spec
plans for both schedulers, and the finalization spec gets a one-line
amendment in the same PR that lands cluster integration.

## Context

After 22 rounds of cleanup, v0.9.0 is published. What is missing for
production cluster runs:

- **No `environment.yml`.** Cluster install is undocumented; users have
  to translate `pyproject.toml` to a conda spec by hand.
- **No stage 0.** Fresh checkouts have no `Resq_i_*.pkl` kernel pickle.
  Today, `dfxm-forward` would import successfully (the auto-load
  silently no-ops on a missing pickle) and then raise the cryptic
  `RuntimeError: forward_model state is not initialized` from inside
  `forward()`. There's no surfaced instruction to regenerate the kernel.
- **No batch-job templates.** Users must invent `bsub` / `sbatch`
  scripts from scratch.
- **No cluster-runs documentation.** README is laptop-centric.
- **No example images in README.** The "what does this thing produce?"
  question has no visual answer in the landing page.

## Goals (recap from finalize spec, Q1)

In priority order:

1. **(A) Onboarding** — fresh clone → conda env create → `pip install -e .`
   → `dfxm-bootstrap` → `dfxm-forward --config x.toml`. No pickle copy
   step, no implicit prerequisites.
2. **(C) Maintainability** — the cluster paths are themselves
   maintainable: one canonical bootstrap entry point, one stage-0
   function, one set of templates per scheduler.
3. **(D) Portability** — runs work identically on DTU LSF, ESRF SLURM,
   and laptop. CI keeps the laptop path validated.

## Components

### 1. `environment.yml` (conda-forge only)

A new top-level file:

```yaml
name: dfxm-geo
channels:
  - conda-forge
dependencies:
  - python>=3.11
  - pip
  - numpy>=1.23,<3
  - scipy>=1.10,<2
  - numba>=0.56
  - matplotlib>=3.6
  - seaborn>=0.12
  - fabio>=2023.4
  - joblib>=1.3
  - tqdm>=4.66.3
  # runtime extras included by default — production install is complete
  - xraylib>=4.1        # [beamstop-wire]
  - plotly>=5           # [identification]
  - psutil>=5           # [memory-aware]
  - pip:
    - -e .
```

**Decisions (Q3b):**
- Channel: `conda-forge` only. Has every dep including `xraylib`,
  `fabio`, `numba`, `scipy`. Avoids the `defaults` channel licensing
  question that affects DTU/ESRF institutional licensing.
- Pinning: `>=` ranges mirroring `pyproject.toml`. `pyproject.toml` is
  the source of truth for compatibility; `environment.yml` just
  bootstraps the conda-side deps that pip can't easily install.
  Reproducibility-critical paper runs can generate a `conda-lock` file
  separately (out of scope for v1.0).
- Runtime extras (`[beamstop-wire]`, `[identification]`,
  `[memory-aware]`) included by default — the cluster install is meant
  to be production-complete; a SLURM job shouldn't suddenly fail
  because xraylib wasn't installed.
- `[dev]` tools (pytest, ruff, mypy, jupyterlab) split into a separate
  `environment-dev.yml` for development setups. Keeps the runtime env
  lean.

### 2. `dfxm-bootstrap` console script (Q2a, Q2b)

New entry point in `[project.scripts]`:

```toml
dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"
```

`dfxm_geo.reciprocal_space.kernel.cli_main(argv=None)`:
- Loads `--config <toml>` and parses the `[reciprocal]` block (TOML
  schema TBD in implementation; mirrors what stage 0 will read).
- Dispatches to `generate_kernel(**kernel_kwargs)`.
- Writes the resulting pickle to the path the matching `dfxm-forward
  --config x.toml` will look for (canonical
  `reciprocal_space/pkl_files/Resq_i_<auto-derived-suffix>.pkl`, or a
  user-overridden `--output <path>`).
- Flags: `--config <toml>` (required), `--output <path>` (optional),
  `--force` (overwrite existing pickle without prompting).

Underlying `python -m dfxm_geo.reciprocal_space.kernel` continues to
work as the lower-level mechanism — `dfxm-bootstrap` is the
recommended user-facing entry point.

### 3. Pipeline stage 0 (Q1, Q6)

Inside `dfxm_geo.pipeline.run_simulation` and
`dfxm_geo.pipeline.run_identification`, before any forward call:

```python
def _ensure_kernel_loaded(config) -> None:
    """Pre-flight: verify the reciprocal-space kernel is loaded.

    If the canonical pickle is missing on disk, raises FileNotFoundError
    with the bootstrap instruction. If the pickle exists but
    forward_model's import-time auto-load somehow didn't run, calls
    `_load_default_kernel(...)` to populate state.
    """
    import dfxm_geo.direct_space.forward_model as fm
    if fm.Resq_i is not None:
        return  # auto-load already populated state at import
    pkl_path = Path(fm.pkl_fpath) / fm.pkl_fn
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Reciprocal-space kernel pickle not found at {pkl_path}.\n"
            f"Run 'dfxm-bootstrap --config {<your toml>}' to generate it "
            "(takes ~50 s for default Nrays=1e8). See docs/cluster-runs.md "
            "for the full cluster workflow."
        )
    fm._load_default_kernel(str(pkl_path))
```

**Decisions:**
- (Q6 a) Lives in the pipeline functions (`run_simulation`,
  `run_identification`, `run_postprocess`). One source of truth; covers
  CLI users, ad-hoc notebook callers, and future tools that import
  these functions.
- (Q1 b) Fail-loud with a clear bootstrap instruction. Not silent
  regen — silent regen could trigger a 1-hour kernel regen on a typo,
  with the user thinking it's "doing the simulation."

### 4. `forward_model` import-time fate (Q3a)

**No change.** `dfxm_geo.direct_space.forward_model` continues to call
`_load_default_kernel()` at module import — silently no-ops if the
pickle is missing, populates module state if present. This preserves
the convenience for notebook users with the pickle on disk:
`import fm; fm.forward(Hg)` continues to work without ceremony.

The pipeline's stage 0 owns correctness; the import-time auto-load is
an opportunistic optimisation that makes the common case fast.

### 5. Batch-job templates (Q4)

New top-level directories `lsf/` (DTU) and `slurm/` (ESRF). Four
templates, two per scheduler:

```
lsf/
  forward_single.bsub      # one full-resolution forward simulation
  identify_array.bsub      # ML-training-data array job (10 samples per task)
slurm/
  forward_single.sbatch    # one full-resolution forward simulation
  identify_array.sbatch    # ML-training-data array job (10 samples per task)
```

**Decisions:**
- (Q4 a) Four templates total (single + array × LSF + SLURM).
- DTU is Sina's primary cluster; LSF templates get closer polish (more
  realistic defaults, more hand-tuned comments). SLURM templates use
  ESRF-typical defaults with a "verify with `sinfo`" callout.
- Sensible defaults inside each template: LSF uses DTU's `hpc` queue,
  24 h walltime, 4 GB memory; SLURM uses ESRF's typical default
  partition. An "EDIT THESE" header block at the top of each template
  flags the lines users must change (account/project, walltime,
  partition/queue overrides).
- Array batching: 10 samples per array task → ~100 array tasks for the
  default `n_samples=1000` ML-training run. LSF syntax:
  `#BSUB -J "ident[1-100]"`; SLURM syntax: `#SBATCH --array=1-100`.

### 6. `docs/cluster-runs.md`

New documentation file. Sections:

1. **Two-step workflow** — `dfxm-bootstrap` once, `dfxm-forward` /
   `dfxm-identify` many times. Why the two-step (kernel regen is
   slow; explicit beats silent regen for expensive operations).
2. **DTU HPC (LSF) walkthrough** — module load, conda env activate,
   bootstrap, submit `lsf/forward_single.bsub`, monitor with `bjobs`,
   collect output.
3. **ESRF (SLURM) walkthrough** — same flow with `sbatch`,
   `squeue`, `--array=...` for ML-training arrays.
4. **Output handling** — `--output` resolves relative to the cluster
   scratch dir; per-job output naming conventions; rsync-back patterns.
5. **Memory + walltime sizing** — per-mode rough estimates from the
   benchmarks (Round 12/13 numba-JIT'd Fd_find baselines + a
   pessimistic `forward()` cost).
6. **DTU vs ESRF specifics callout** — partition naming, time format
   differences, account flags.

### 7. README updates (Q5 a)

Two new top-level sections added to the existing README:

- **"Running on a cluster"** — short summary (3–5 sentences) that
  links to `cluster-runs.md` for the full walkthrough; lists the
  template paths (`lsf/`, `slurm/`).
- **"Examples"** — 3–5 inline images: a representative simulated
  DFXM image stack frame, an identification multi-mode result, the
  coordinate-frame diagram. Inspiration target:
  [FABLE-3DXRD/xrd_simulator](https://github.com/FABLE-3DXRD/xrd_simulator)'s
  hero-image-near-the-top README pattern.

Existing sections (quickstart, configuration, deferred follow-up
items) stay largely intact. (Q5 — option a, incremental, not full
rewrite — chosen over a top-down restructure to avoid churn-risk on
the working sections.)

### 8. Auto-generated example images (Q5 d)

New `scripts/render_readme_examples.py`:

- Runs `dfxm-forward --config <small variant of default>` against a
  scaled-down config (e.g., `Npixels=170`, `phi_steps=chi_steps=11`)
  to keep wall time under ~30 s.
- Saves selected output frames + the postprocess COM/chi_shift figure
  to `docs/img/example_*.png`.
- Documented as the regen step in `docs/cluster-runs.md` and the
  README; CI does NOT run it (would slow the test matrix and tie
  rendered images to floating-point output).

### 9. `pyproject.toml` version bump (orthogonal cleanup)

Current `pyproject.toml:7` says `version = "0.1.0"` — stale relative to
the just-tagged v0.9.0 release. As part of the v1.0 cluster
integration work, bump to `version = "1.0.0"` in the same commit that
introduces the bootstrap entry point. Git tags (not the pyproject
field) are what Zenodo and citations consume, so backfilling 0.9.0 is
not necessary.

## Decision log (Q1–Q6 trace)

| Q | Decision | Section |
|---|---|---|
| Q1 | Stage 0 = fail-loud with bootstrap instruction | §3 |
| Q2a | New `dfxm-bootstrap` console script | §2 |
| Q2b | TOML-aware bootstrap | §2 |
| Q3a | Keep `forward_model` import-time auto-load (no change) | §4 |
| Q3b | conda-forge only; `>=` ranges; runtime extras included; dev split | §1 |
| Q4 | 4 templates (lsf+slurm × single+array); DTU primary; sensible defaults + EDIT THESE block; generic placeholders rejected in favor of cluster-typed defaults | §5 |
| Q5 (a) | Incremental README (add cluster + examples sections, no full rewrite) | §7 |
| Q5 (d) | Auto-generated images via `render_readme_examples.py` | §8 |
| Q6 | Stage 0 lives in pipeline functions, not CLI entry points | §3 |

## Out of scope (non-goals for v1.0)

- **Replacing the `Resq_i` pickle.** Filed as
  `[[followups-kernel-pickle-alternatives]]` (6 candidate directions,
  not gating). v1.0 ships with the existing pickle + stage-0 regen UX.
- **GPU / numba CUDA acceleration.** Current numba CPU JIT is already
  4–5× faster than pre-cleanup Python loops.
- **MPI / multi-node distributed compute.** Job arrays are the
  parallelism model.
- **Replacing the kernel regeneration with a smaller fixture for
  CI tests.** Round 7 pickle-dependent tests stay deferred until a
  fixture kernel exists; not gating.
- **Reflection runtime-configurable refactor.** Targets v1.1 — does
  not block v1.0.
- **conda-lock files for exactly-pinned reproducibility.** v1.0 ships
  with `>=` ranges in environment.yml; lock-file generation is
  optional add for paper runs.
- **Container (Apptainer/Singularity) recipe.** Considered in finalize
  spec Q2 (option c) and rejected — `environment.yml` covers DTU + ESRF.
- **`dfxm-archive` / two-stage staging dir.** Users rsync manually if
  needed.

## Path to v1.0 release

After this spec is approved and the writing-plans skill produces an
implementation plan:

1. Polish round commits land in dependency order (env file → bootstrap
   CLI → stage 0 → templates → docs → README → render script).
2. CI must stay green throughout (lesson from Round 21.5: explicit
   per-commit verification, no "CI is running" hand-waves).
3. Mark `pyproject.toml` version `1.0.0`.
4. Tag `v1.0.0` on the merge commit; `gh release create v1.0.0` with
   release notes summarising the cluster surface.
5. Push tag → second Zenodo deposit (assumes user has linked
   Zenodo↔GitHub between v0.9.0 and v1.0.0).

## Related memory

- `[[cleanup-session-state]]` — overall cleanup status; this work is
  Round 23+.
- `[[followups-kernel-pickle-alternatives]]` — the deferred
  exploration of replacing the 128 MB pickle.
- `[[followups-pipeline-identification-oom]]` — environmental memory
  pressure on Sina's laptop; cluster runs eliminate this concern.
- `[[followups-readme-examples]]` — the source follow-up that asked
  for example images; this spec absorbs it.
- `[[nsub-default]]` — Nsub=1 is Sina's typical real-run choice; the
  cluster templates default to Nsub=2 (publication quality) but the
  EDIT THESE block calls out Nsub=1 as the faster option.
