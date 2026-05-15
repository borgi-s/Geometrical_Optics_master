---
title: Cleanup finalization design
status: approved
date: 2026-05-15
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - docs/superpowers/plans/2026-05-12-codebase-cleanup.md
  - cleanup_session_state.md (auto-memory) Round 0 .. Round 19
  - PR #1 (draft on origin/cleanup/main-modernization at HEAD `9e954eb`)
---

# Cleanup finalization design

## Purpose

This document captures the operational decisions for closing out the
DFXM `Geometrical_Optics_master` cleanup arc — what lands before the
merge to `main`, how the merge is performed, what the v0.9.0 release
looks like, what gates v1.0, and how the project is positioned for the
long term. Decisions were elicited via an interview ("grill") on
2026-05-15; each subsection cross-references the question that produced
it (Q1 .. Q12) so a future reader can trace rationale.

This is the canonical handoff for the finalization phase. Implementation
plans for each work item live separately under `docs/superpowers/plans/`.

## Context

After 19 rounds of subagent-driven cleanup work (`Round 0` Phase 0
scaffold through `Round 19` Beam_Stop no-op closeout), the cleanup
branch `cleanup/main-modernization` is:

- **140 commits ahead of `main`**, origin in sync, PR #1 draft +
  mergeable, all CI green (Python 3.11 + 3.12 tests, pre-commit hooks).
- All 5 collaborator branches (CDD_inc, dislocation_identification,
  ESRF_DTU, Purdue_Paper, Beam_Stop) are absorbed into the cleanup —
  Phase 11 of the original plan is functionally complete.
- The bit-exact parity to `legacy/init_forward.py` was proven once
  (Round 15) and is documented in `cleanup_session_state.md`.
- Round 18 + polish round added the Purdue sample-remount port, the
  `Find_Hg(remount_name=...)` whitelist guard, and converted the
  `exit("Not enough values for ...")` calls in `resolution.py` into
  `ValueError`s.
- The architecture is package-shaped (`src/dfxm_geo/`), CLI-driven
  (`dfxm-forward`, `dfxm-identify`), TOML-configured, mypy-clean across
  25 source files, ruff-clean, and Zenodo-deposit-ready (`.zenodo.json`
  staged but not deposited).

What remains is the *finalize* phase: a small polish round, a merge,
the first release tag, the first Zenodo deposit, and the post-merge
roadmap.

## Goals (Q1)

In priority order:

1. **(A) Onboarding** — a new collaborator can clone the repo and run
   `dfxm-forward --config x.toml` to reproduce paper figures, with no
   tribal DTU-laptop knowledge.
2. **(C) Maintainability** — future-Sina (or a successor) can extend
   the codebase with new physics without piling on tech debt.
3. **(D) Portability** — production-scale runs work on Linux clusters,
   not just the development laptop.

Goal (B) "citable artifact" (Zenodo DOI) is a means to (A), not an
independent priority — addressed under §"Release & Zenodo arc."

## Production environment (Q1.5, Q2)

- **Production target:** Linux SLURM clusters (DTU HPC + ESRF).
- **Install path:** conda env from a checked-in `environment.yml`,
  followed by `pip install -e .[beamstop-wire,identification]`. Both
  target clusters expose conda via modules; numba + xraylib + fabio
  resolve cleanly via conda-forge wheels.
- **Development environment unchanged:** Sina's existing Windows
  laptop venv at `C:\Users\borgi\Documents\GM-reworked\.venv\` keeps
  working. Cluster integration is additive, not migratory.

## Workload model (Q4)

The pipeline must support all current modes; the dominant near-term
workload is **mode C — ML training-data generation** for the
dislocation-identification follow-on work (extension of Borgi 2025,
J. Appl. Cryst. 58, 813–821).

| Mode | Example | Scale | Orchestration |
|------|---------|-------|---------------|
| Single config full forward | one `(dis, ndis)`, full Nsub=2, 510×510 detector, 61×61 rocking | ~3700 images, ~1 h wall | one SLURM job, single node, multi-thread |
| Parameter sweep | N variants per `configs/variants/dis_*.toml` | N × single-config | SLURM array, one task per variant |
| **ML training-data generation** | `dfxm-identify` multi-mode, 1000+ samples × planes × Burgers vectors | embarrassingly parallel | SLURM array, **batch K=10 samples per task → ~100 array tasks** (Q4.5 default) |
| z-scan stack (ESRF_DTU port) | many z-layers × planes × Burgers vectors × rocking | hours-to-days | single SLURM job with checkpointing (future) |

## Output handling on the cluster (Q5)

- The CLI flag `--output <abs path>` is the single point of policy.
  No magic auto-detection of `$SCRATCH`. Cluster users set
  `--output $SCRATCH/$SLURM_JOB_ID/run/` or similar in their job
  script.
- A new `docs/cluster-runs.md` ships a worked example: SLURM script,
  conda module load, output path conventions, where the kernel pickle
  comes from.

## Pipeline shape: stage 0 = ensure kernel (Q3)

Today, `dfxm_geo.direct_space.forward_model` auto-loads the
`Resq_i_*.pkl` reciprocal-space kernel at module import iff the file
exists at the canonical path. This breaks fresh-clone scenarios on a
cluster: import succeeds but `Find_Hg` fails.

**The pipeline gains a stage 0:** `_ensure_kernel(config)`. Behaviour:

- If the canonical pickle exists and matches the config's reciprocal-
  space parameters (per its `_vars.txt` sidecar), reuse it.
- Otherwise, regenerate via
  `dfxm_geo.reciprocal_space.kernel.generate_kernel(...)` (already
  exists, ~50 s on the laptop, deterministic for a given seed).
- The current import-time auto-load in `forward_model` either becomes
  a no-op or is moved out of import-time — it is no longer load-bearing
  once stage 0 owns the kernel lifecycle.

This is part of **cluster integration**, not the polish round. It
gates v1.0.

## Polish round (pre-merge)

Three concrete work items, in order, before marking PR #1 ready-for-
review and merging to `main`:

### 1. Drop all 5 deprecation shims (Q7, decision a)

Files to delete:
- `functions.py` (re-exports `Fd_find`, `multi_dislocs_parallel`,
  `rotatedU`, `fast_inverse2`, `load_or_generate_Hg`, `check_folder`)
- `image_processor.py` (re-exports `save_images_parallel`,
  `load_images`, `calc_moments`, `inv_polefigure_colors`, etc.)
- `direct_space/forward_model.py` (re-exports the whole
  `dfxm_geo.direct_space.forward_model` surface — ~40 names)
- `reciprocal_space/recspace_res.py` (re-exports `reciprocal_res_func`)
- `reciprocal_space/{generate_Resq_i,exposure_time}.py` (analogous
  re-export shims)

Consequence: `legacy/init_forward.py` and `legacy/init_forward_purdue.py`
will stop importing. This is consistent with the "frozen reference for
paper-figure reproduction or audit, NOT maintained" framing already in
`legacy/init_forward_purdue.py`'s docstring (Round 18 Task 11). Future
parity re-runs check out a pre-cleanup commit hash via `git checkout
<tag> -- init_forward.py direct_space/forward_model.py image_processor.py`
into a worktree.

The Round 15 bit-exact parity is already proven and documented; ad-hoc
re-runs of legacy scripts are not load-bearing.

### 2. Fill in Zenodo metadata (Q8 prerequisite)

In `.zenodo.json`:
- ORCIDs for Henning Friis Poulsen, Carsten Detlefs, Grethe Winther —
  fetched via WebSearch / their respective institutional pages /
  ORCID.org direct lookup. (Carsten Detlefs is at ESRF; the other two
  at DTU. Affiliations have not changed, per Sina.)
- Affiliations: fill DTU for Poulsen + Winther directly (public
  knowledge; no need to ask).
- Carlsen and Ræder: keep as name-only entries (per Q8); no ORCID, no
  affiliation chase.

### 3. README example images — explicitly out of scope here

Per Q10a, README example images are deferred to the post-merge v1.0
cluster docs push — they ride along with the README rewrite that
cluster integration triggers anyway. Including them in the polish round
is scope creep without changing the answer to "is the cleanup done?"

## Merge strategy (Q6, Q10b, Q10c)

- **Timing:** after the polish round above, mark PR #1 ready-for-review
  and merge.
- **Strategy:** **merge commit**, preserving all 140 commits as
  bisectable history on `main`. (`git log --first-parent main` shows
  the cleanup arc; `git log main` shows every commit.)
- **Reviewer:** **self-merge.** 19 rounds of subagent-driven
  development with two-stage review per task already provided more
  thorough review than a single external pass would. Optional FYI
  ping (NOT a blocking review) to Khaled with the diff link.
- **Post-merge branch lifecycle:** delete
  `cleanup/main-modernization` from local + origin. The merge commit
  on `main` preserves all history; the branch reference is then
  redundant. The historical anchor tags (`pre-cleanup-2026-05-12`,
  `baseline-smoke-tests`, `offline-checkpoint-2026-05-12`,
  `checkpoint-2026-05-12-phase5-complete`) stay.

## Release & Zenodo arc (Q8)

- **First release: v0.9.0** at the merge commit. Triggers the first
  Zenodo deposit (via the GitHub↔Zenodo integration; `.zenodo.json`
  metadata is already staged in the cleanup branch).
- **First DOI** issues at v0.9.0 deposit. Citable from this point on.
- **v1.0.0** follows when cluster integration lands (see post-merge
  plan below). Second Zenodo deposit; both records linked under the
  same concept DOI.
- **Why v0.9 first, not v1.0:** until stage 0 (kernel ensure-or-
  regenerate) lands, fresh clones cannot end-to-end run without
  manually copying the `Resq_i_*.pkl` pickle to disk first. That's
  pre-1.0 territory in semver-honest terms. v1.0 means "fresh clone →
  conda env create → pip install -e . → dfxm-forward works."

## Post-merge plan

In order (Q9):

1. **v1.0 — cluster integration push** (gates v1.0 release):
   - `environment.yml` for the conda env on DTU HPC + ESRF cluster
   - Stage 0 refactor in `dfxm_geo.pipeline` (`_ensure_kernel(config)`
     before `Find_Hg`; cleanup of `forward_model` import-time auto-load)
   - `docs/cluster-runs.md` with a worked SLURM array example
   - `slurm/` directory with template job + array scripts (single
     forward, parameter sweep, ML-training-array)
   - README rewrite — include cluster-runs cross-reference, install
     instructions for both laptop and cluster, **and the deferred
     README example images from Q10a**
   - Tag v1.0.0 → second Zenodo deposit
2. **v1.1 — reflection runtime-configurable refactor** (Q9 (b),
   targets v1.1, NOT gating):
   - Allow `(h, k, l)` to be specified at runtime in the TOML config;
     recompute detector ray grid `rl`, `Theta`, `Us` lazily instead of
     at import time.
   - Unblocks Purdue's `[020]` reflection (currently inaccessible) and
     arbitrary other reflections.
   - Touches `forward_model`'s import-time geometry construction;
     ripples through `pipeline.py`. Significant — gets its own spec +
     plan when v1.0 lands.
3. **Science work runs alongside** (Q9 (c)) from the moment v1.0
   lands — ML training data generation for the dislocation-
   identification follow-on becomes the substrate user.

## Long-term lifecycle (Q11)

**(a) + (d): single point of failure, with honest archival framing in
README.**

- The repo is maintained by Sina Borgi while at DTU; no formal
  succession plan, no nominated maintainer-of-last-resort.
- When Sina pivots / leaves DTU, the repo goes dormant. The Zenodo
  DOI + GitHub repo remain citable; future researchers fork from
  Zenodo if they need a starting point.
- The README will include a one-paragraph note setting these
  expectations explicitly: "This is active research software
  maintained by Sina Borgi while at DTU; forks welcome; archived as a
  Zenodo deposit at <DOI>." Honest signaling beats implied promises
  of ongoing maintenance.
- Existing handoff artifacts (`docs/architecture.md`,
  `docs/physics.md`, `docs/reproducibility.md`,
  `docs/superpowers/specs/*`, the Zenodo deposit) ARE the succession
  package for any future researcher who picks the work up.

## Explicit non-goals

To prevent scope creep, the following are explicitly NOT in scope for
the finalize phase or for v0.9 / v1.0:

- **Renaming the repo** to `dfxm-geo` or similar. The git URL stays as
  `borgi-s/Geometrical_Optics_master` for citation continuity.
- **PyPI publication.** GitHub + Zenodo is sufficient for academic
  distribution. Adding PyPI is a v2 question if external user demand
  emerges.
- **ReadTheDocs / GitHub Pages.** Markdown docs in `docs/` are
  sufficient. Hosting them is a v2 question.
- **GPU / numba CUDA acceleration.** The current numba CPU JIT
  (Round 12 / Round 13) is already 4–5× faster than the pre-cleanup
  Python loops. GPU work would benefit but is far out of scope.
- **MPI / distributed compute.** SLURM array jobs are the parallelism
  model; multi-node MPI is overkill for the workload shape.
- **Two-stage staging dir + archival rsync** (Q5 option d). Users can
  do this manually if they need it.
- **Auto-detected per-cluster output paths** (Q5 option c). Magic is a
  debugging trap; the CLI flag stays explicit.
- **Replacing the kernel pickle with a smaller fixture for tests.**
  Round 7 pickle-dependent tests stay deferred until a real fixture
  kernel exists (separate work item, not gating).
- **A successor plan** (Q11 option c). If a successor emerges, write a
  handoff doc then. Not now.

## Open follow-ups (filed in memory, NOT gating finalization)

- `[[followups-readme-examples]]` — example images / illustrations
  inspired by `github.com/FABLE-3DXRD/xrd_simulator`. Lands with the
  v1.0 README rewrite.
- `[[followups-pipeline-identification-oom]]` — 8 tests in
  `test_pipeline_identification.py` OOM on Sina's laptop under whole-
  suite runs. Pre-existing memory pressure on the 11.79M-column `rl`
  grid; environmental on the laptop, fine on cluster. Not blocking.
- `[[followups-cdd-inc-port]]` — wire-chord physics deviation
  (factor-of-2 vs full chord) preserved verbatim from CDD_inc; audit
  before reusing wire-mode for non-this-paper data.
- `[[followups-purdue-paper-port]]` — both Round 18 follow-ups were
  addressed in the polish round on 2026-05-15 (commits `d411a3f`,
  `7010c97`); this memory file remains as historical context.

## Decision log (Q1 .. Q12 trace)

| Q | Decision | Source line |
|---|---|---|
| Q1 | Goals A + C + D in priority order | §Goals |
| Q1.5 | Production environment = Linux cluster | §Production environment |
| Q2 | Install via conda env (`environment.yml`) + `pip install -e .` | §Production environment |
| Q2.5 | Target clusters DTU HPC + ESRF | §Production environment |
| Q3 | Kernel pickle = stage 0 of pipeline (regenerate or reuse) | §Pipeline shape |
| Q4 | Workload mode E (all); ML training (C) dominant near-term | §Workload model |
| Q4.5 | SLURM batching = 10 samples per array task → 100 tasks | §Workload model |
| Q5 | Output via CLI `--output`; docs ship `cluster-runs.md` | §Output handling |
| Q6 | Polish round → merge with merge commit (preserve 140 commits) | §Merge strategy |
| Q7 | Drop all 5 deprecation shims | §Polish round item 1 |
| Q8 | v0.9.0 on merge → first Zenodo; v1.0.0 after cluster | §Release & Zenodo arc |
| Q8.5 | ORCID lookups for Poulsen / Detlefs / Winther; skip Carlsen / Ræder | §Polish round item 2 |
| Q9 | Post-merge: cluster (gates v1.0); reflection (v1.1); science alongside | §Post-merge plan |
| Q10a | README images deferred to v1.0 cluster docs push | §Polish round item 3 |
| Q10b | Delete `cleanup/main-modernization` after merge | §Merge strategy |
| Q10c | Self-merge; optional FYI to Khaled | §Merge strategy |
| Q11 | Single point of failure (Sina) + honest README archival framing | §Long-term lifecycle |
| Q12 | Write this spec doc and commit it | (this document) |
