# M5 arc 1: tutorial notebooks 01–05, references page, notebook CI

**Date:** 2026-06-12
**Status:** DRAFT — awaiting Sina's review.
**Baseline:** local `main = 2ccd9cf` (M4 Stage 4.1 merged, push pending).
Branch `feature/m5-tutorial-notebooks` in worktree `wt-m5-tutorial` with its
own venv — the main tree and shared `.venv` belong to the concurrent
M4 Stage 4.2 session and are not touched by this arc.

## 1. Goal and scope

Roadmap §6 (Goal 5, "Final tutorial / walkthrough"), first arc. M1–M3 are
closed, so notebooks 01–05 are unblocked. Style per roadmap: **minimal
text, figure-led, citation-anchored** — each section ≤3 sentences of prose,
then code + figure.

**In scope**

1. Notebook series `examples/01_…05_*.ipynb` (§3).
2. `docs/references.md` — full BibTeX for every cited source (§4).
3. `.github/workflows/notebooks.yml` — CI executes notebooks 01–03 (§5).
4. `examples/README.md` updated as the notebook index.

**Out of scope (deferred)**

- `06_other_crystals.ipynb` — gated on M4 (CIF), in flight elsewhere.
- MkDocs Material site + GitHub Pages — second M5 arc, after 06 exists.
- README roadmap-section swap — waits for Sina's own README ideas
  (flagged 2026-06-12).
- Any version bump, tag, or PyPI/conda action.

## 2. Hard constraint: concurrent M4 session

M4 Stage 4.2 is being developed in the main tree
(`feature/m4-stage42-cif-ingestion`) by a separate session. This arc MUST
NOT edit files that arc is touching or will touch:

- `pyproject.toml` (4.2 adds the `[cif]` extra; `ipykernel`/`jupyterlab`
  are already in `dev`, so no dependency edits are needed here — `nbmake`
  is installed ad hoc in the workflow).
- `.github/workflows/ci.yml` (notebook CI gets its own workflow file).
- `src/dfxm_geo/**` — this arc adds no production code. If a notebook
  exposes a real bug, it is reported, not fixed on this branch.
- The main working tree itself; all work happens in `wt-m5-tutorial`.

## 3. Notebook series

Conventions (follow `examples/identification_ml_tutorial/`):

- Committed **stripped** (no outputs); a small `preview.png`-style figure
  per notebook gives the figure-led browsing experience on GitHub.
  Previews live in `examples/img/<nn>_<name>_preview.png`.
- Notebooks are flat files `examples/<nn>_<name>.ipynb`; scratch output
  goes to a gitignored per-notebook out dir.
- Smoke-scale geometry throughout (small grids, low ray counts — the
  established sizing rule); first cell runs `dfxm-bootstrap --if-missing`
  style kernel setup so a fresh clone works.
- Every physics claim links to `docs/references.md` anchors.

| # | File | Content | Cites |
|---|---|---|---|
| 01 | `01_quickstart.ipynb` | empty-TOML → first image in ~5 lines; the two-stage model in one diagram (reuse `docs/img/`) | Borgi 2024 §2 |
| 02 | `02_reciprocal_space.ipynb` | kernel point cloud; MC vs analytic backend comparison figure | Poulsen 2021; Borgi 2024 §3 |
| 03 | `03_dislocations_and_contrast.ipynb` | edge/screw/mixed contrast; weak-beam; COM ≈ −qi reproduction (the README figure, now executable) | Borgi 2024 Figs. 3–5; Hirth & Lothe |
| 04 | `04_oblique_and_reflections.ipynb` | oblique mount; `dfxm-find-reflections` table; g·b invisibility across a reflection sweep | Borgi 2024 App. A |
| 05 | `05_identification_at_scale.ipynb` | slim re-cut of the ML tutorial; fan-out throughput plot built from the committed numbers in `docs/cluster-profiling.md` (no live cluster run) | Borgi 2025 |

## 4. References page

`docs/references.md`: full BibTeX + one-line role for each of —
Borgi 2024 (IUCrJ, forward model), Borgi 2025 (identification),
Poulsen 2021 (original MATLAB GO model), Poulsen 2017 (DFXM optics),
Hirth & Lothe (dislocation displacement fields), and the darkmod
comparison source (Carlsen et al.). Stable anchors per entry so notebooks
can deep-link.

## 5. Notebook CI

New standalone `.github/workflows/notebooks.yml` (never touches `ci.yml`):

- ubuntu-latest, one Python version (3.11), `pip install -e ".[dev]" nbmake`.
- Bootstrap a low-ray smoke kernel (CI has no `pkl_files/` — the known CI
  kernel gap), then `python -m pytest --nbmake examples/01_*.ipynb
  examples/02_*.ipynb examples/03_*.ipynb`.
- 04/05 stay CI-exempt per roadmap ("CI executes notebooks 01–03"):
  04 needs reflection-sweep runtime, 05 is scale-themed.
- Budget: each notebook ≤ ~3 min on CI after JIT; trigger on PRs/pushes
  touching `examples/**`, `src/**`, or the workflow itself.

## 6. Gates / definition of done (this arc)

- [ ] `pytest --nbmake` green on 01–03 in the fresh worktree venv.
- [ ] 04 and 05 execute end-to-end clean locally (same venv).
- [ ] Full suite + mypy unchanged from the `2ccd9cf` baseline
      (898 passed / mypy 0 — no production code edits).
- [ ] Five preview figures committed; `examples/README.md` indexes the
      series; every notebook links `docs/references.md`.
- [ ] No diff in `pyproject.toml`, `.github/workflows/ci.yml`, `src/**`.

Roadmap §6's own DoD (six executed notebooks on a public site) closes in
the second M5 arc, after M4 lands.

## 7. Risks

- **Merge adjacency with M4 4.2** — mitigated by the §2 no-touch list;
  remaining overlap (`examples/README.md`, `.gitignore`) is trivial.
- **Kernel availability** — notebooks self-bootstrap at smoke ray counts;
  CI does the same explicitly.
- **CI runtime drift** — smoke sizing is pinned in the notebooks
  themselves (no env-var dual-path complexity in arc 1; add one later
  only if CI times demand it).
- **Citation accuracy** — exact BibTeX fields verified against the
  publisher records during implementation, not recalled from memory.
