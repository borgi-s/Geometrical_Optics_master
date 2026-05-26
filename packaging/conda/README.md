# conda-forge packaging

`recipe.yaml` is the conda-forge recipe for `dfxm-geo`, in the rattler-build
v1 schema. It builds a single `noarch: python` package from the PyPI sdist.

This directory is a **reference copy**. Once the conda-forge feedstock exists,
that feedstock is the source of truth; keep this file in sync manually on
version bumps (or just delete it and treat the feedstock as canonical).

## Submitting to conda-forge (one-time)

The recipe is built and shipped by conda-forge's CI, not by this repo. To get
`dfxm-geo` onto conda-forge:

1. Fork [`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes).
2. Copy this file to `recipes/dfxm-geo/recipe.yaml` in the fork.
3. Open a PR against `conda-forge/staged-recipes`. The lint + build bots
   validate it on linux/osx/win.
4. After a maintainer merges, conda-forge auto-creates
   `conda-forge/dfxm-geo-feedstock`. Builds and uploads happen there.

After that, `conda install -c conda-forge dfxm-geo` works.

## Version bumps

The regro **autotick bot** watches PyPI and opens a version-bump PR on the
feedstock automatically for each new release. Manual steps are only needed if
dependency pins or the recipe structure change.

## Notes

- The sdist `sha256` in `recipe.yaml` is the PyPI artifact's digest. On a
  version bump it must be updated (the autotick bot does this for you).
- `matplotlib-base` / `seaborn-base` are used deliberately to avoid pulling in
  Qt / statsmodels that the package never imports.
- Optional extras (`xraylib`, `psutil`, `plotly`) are listed under
  `run_constraints`: not installed by default, but version-pinned if a user
  installs them for beamstop-wire / memory-aware / identification features.
