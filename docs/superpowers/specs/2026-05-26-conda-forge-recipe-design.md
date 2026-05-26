# conda-forge recipe for `dfxm-geo` — design

**Date:** 2026-05-26
**Status:** approved (brainstorming) → ready for implementation
**Author:** Sina Borgi (with Claude Code)

## Goal

Distribute `dfxm-geo` through conda-forge so users can
`conda install -c conda-forge dfxm-geo`. The package is pure Python, so it
ships as a single `noarch: python` build; conda-forge's own CI does all
cross-platform building. Our deliverable is a correct recipe that passes the
conda-forge lint + build bots in a `staged-recipes` PR.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Distribution target | conda-forge feedstock |
| Recipe format | rattler-build v1 `recipe.yaml` |
| Source | PyPI sdist (`dfxm_geo-2.1.0.tar.gz`) |
| Optional extras | `run_constrained` (optional, version-pinned if present) |

## Source pin

- PyPI project: `dfxm-geo`; sdist filename (PEP 625): `dfxm_geo-2.1.0.tar.gz`
- **sha256:** `8dc4385ead462b2716c47b1c154b89ebc6e12f14b3ba42257b5e10c1ec993083`
- v2.1.0 is live on PyPI (confirmed 2026-05-26), so the staged-recipes build
  bot can fetch the tarball.

## Recipe content

`noarch: python`, build script
`python -m pip install . -vv --no-deps --no-build-isolation`.

**host:** python >=3.11, pip, setuptools >=68, wheel

**run:** python >=3.11, numpy >=1.23,<3, scipy >=1.10,<2, numba >=0.56,
matplotlib-base >=3.6, seaborn-base >=0.12, fabio >=2023.4, h5py >=3,
joblib >=1.3, tqdm >=4.66.3

- Uses the `-base` variants of matplotlib and seaborn (conda-forge norm —
  avoids dragging in Qt / statsmodels the package doesn't import).

**run_constrained** (optional extras, not installed by default):
xraylib >=4.1 (beamstop-wire), psutil >=5 (memory-aware), plotly >=5
(identification).

## Tests (baked into the recipe)

- Python imports: `dfxm_geo` + the 6 subpackages (`analysis`, `crystal`,
  `direct_space`, `io`, `reciprocal_space`, `viz`), with `pip_check: true`.
- Script smoke: `dfxm-forward --help`, `dfxm-identify --help`,
  `dfxm-bootstrap --help` — confirms console entry points resolve.

## File layout

- **Reference copy in this repo:** `packaging/conda/recipe.yaml` — documents
  the conda build alongside the code. Not consumed by any build here; kept in
  sync manually on version bumps (the conda-forge feedstock is the source of
  truth once it exists).
- **Submission copy:** `recipes/dfxm-geo/recipe.yaml` in a fork of
  `conda-forge/staged-recipes`. Identical content.

## Validation strategy

This machine has broken conda HTTP (curl works, conda/rattler resolves may
not), so a full local `rattler-build` is not guaranteed. Local validation is
limited to:

1. Recipe schema sanity (YAML well-formed, fields present).
2. sha256 matches the live PyPI sdist (already confirmed above).
3. pip-based import smoke against the installed package (already green via the
   existing test suite).

**Authoritative validation is conda-forge's multi-platform CI in the
staged-recipes PR.** We do not claim a green local conda build.

## Submission & maintenance (out of scope for the code change here)

1. Fork `conda-forge/staged-recipes`, add `recipes/dfxm-geo/recipe.yaml`,
   open a PR. **User pushes the fork / opens the PR** (external repo).
2. After merge, conda-forge auto-creates `conda-forge/dfxm-geo-feedstock`.
3. The regro autotick bot watches PyPI and opens version-bump PRs on the
   feedstock automatically for future releases.

## Out of scope

- No changes to the existing PyPI publish workflow.
- No automated conda upload from this repo's CI (conda-forge owns building).
- maintainers list: `borgi-s` (the only maintainer).
