# conda-forge feedstock update — dfxm-geo 3.0.0

Step-by-step for pushing **3.0.0** to conda-forge. The authoritative recipe lives
in the feedstock repo (`conda-forge/dfxm-geo-feedstock`, file
`recipe/recipe.yaml`); the in-repo `packaging/conda/recipe.yaml` is the
maintainer's reference copy and is already updated to 3.0.0.

**Why a hand-edit is needed this release:** the autotick bot only bumps `version`
+ `sha256`. It does NOT re-derive entry points or dependencies from
`pyproject.toml`. Two things changed since the feedstock's last sync and MUST be
applied by hand:

1. **New CLI:** `dfxm-find-reflections` — add it to `build.python.entry_points`
   (the recipe must list all **7** scripts, or its Windows `.exe` launcher is
   broken — exits 1 with no output).
2. **New optional extra:** `gemmi` (the `[cif]` CIF-input extra) — add
   `gemmi >=0.7` to `run_constraints`. Keep it OUT of `run` so the package stays
   `noarch: python` (gemmi is a compiled dep; a hard run-dep would break noarch).

No other dependency or `requires-python` changes vs. the feedstock's current
pins, so `run`/`host` and `python_min` (3.11) are unchanged.

---

## Step 0 — Prerequisite: 3.0.0 must be on PyPI first

conda-forge builds from the **PyPI sdist**, not from git. Nothing happens on
conda-forge until 3.0.0 is published to PyPI.

- The `v3.0.0` tag has been pushed; the GitHub Actions `publish.yml` run builds +
  uploads to TestPyPI automatically, then **waits for your manual approval** on
  the `pypi` GitHub Environment.
- Go to **GitHub → Actions → the `v3.0.0` "Publish" run → Review deployments →
  approve `pypi`.** This publishes `dfxm-geo 3.0.0` to PyPI.
- Confirm it is live: <https://pypi.org/project/dfxm-geo/3.0.0/>.

## Step 1 — Get the sha256 of the 3.0.0 sdist

```bash
curl -sL https://pypi.org/packages/source/d/dfxm-geo/dfxm_geo-3.0.0.tar.gz | sha256sum
```

(If you let the bot do the bump in Step 2, it fills this for you and you can skip
computing it.)

## Step 2 — Open / get the feedstock PR

**Option A (recommended — let the bot start it):** within a few hours of the PyPI
publish, the **regro autotick bot** opens a PR on `conda-forge/dfxm-geo-feedstock`
titled "dfxm-geo v3.0.0" that bumps `version`, resets `build.number` to 0, and
fills `sha256`. Check the feedstock's Pull Requests tab.

**Option B (manual):** fork/branch the feedstock and edit `recipe/recipe.yaml`
yourself, setting `version: "3.0.0"`, `build.number: 0`, and the `sha256` from
Step 1.

## Step 3 — Apply the two hand-edits to `recipe/recipe.yaml`

On the PR branch, make the recipe match the updated reference
(`packaging/conda/recipe.yaml` in this repo). Concretely:

**a) entry_points — add `dfxm-find-reflections`:**

```yaml
build:
  python:
    entry_points:
      - dfxm-forward = dfxm_geo.pipeline:cli_main
      - dfxm-identify = dfxm_geo.pipeline:cli_main_identify
      - dfxm-bootstrap = dfxm_geo.reciprocal_space.kernel:cli_main
      - dfxm-find-reflections = dfxm_geo.find_reflections_cmd:cli_main   # <-- ADD
      - dfxm-init = dfxm_geo.init_cmd:cli_main
      - dfxm-migrate-output = dfxm_geo.io.migrate:cli_main_npy_to_h5
      - dfxm-migrate-h5 = dfxm_geo.io.migrate:cli_main_h5_to_h5
```

**b) run_constraints — add `gemmi`:**

```yaml
requirements:
  run_constraints:
    - gemmi >=0.7       # cif (CIF crystal-structure input)   # <-- ADD
    - xraylib >=4.1     # beamstop-wire
    - psutil >=5        # memory-aware
    - plotly >=5        # identification
```

**c)** (optional, nice-to-have) add `dfxm-find-reflections --help` to the
`tests: - script:` block, and refresh the `about.description` — both are already
in `packaging/conda/recipe.yaml` if you want to copy the whole file over.

> Easiest path: copy this repo's `packaging/conda/recipe.yaml` over the
> feedstock's `recipe/recipe.yaml`, then restore the bot's `sha256` line (or
> paste the Step-1 hash).

## Step 4 — Verify and merge

- The feedstock CI (Azure) re-renders and builds on linux/osx/win. Wait for all
  checks green. The **Windows** build is the one that exercises the `.exe`
  launchers — if `dfxm-find-reflections` were missing, this is where it would
  surface.
- Sanity-check the build log lists all 7 `*.exe` launchers.
- Merge the PR. conda-forge publishes `dfxm-geo 3.0.0` to the `conda-forge`
  channel within ~30–60 min.

## Step 5 — Confirm

```bash
conda search -c conda-forge "dfxm-geo>=3.0.0"
# optional CIF support (kept out of run deps on purpose):
conda install -c conda-forge dfxm-geo gemmi
```

---

### Recap of the 3.0.0 deltas (vs the feedstock's previous recipe)
- `version` 2.x → **3.0.0**, fresh `sha256`, `build.number` 0.
- entry_points: **+ dfxm-find-reflections** (now 7 total).
- run_constraints: **+ gemmi >=0.7** (cif extra; optional, noarch preserved).
- run/host deps and `python_min` (3.11): unchanged.
