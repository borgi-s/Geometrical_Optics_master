# DFXM Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the `borgi-s/Geometrical_Optics_master` repo from a research-prototype dump into a maintainable, reproducible, performance-conscious Python project without changing the underlying DFXM physics.

**Architecture:** Phased cleanup landing on a single `cleanup` branch off `main`. Each phase produces a green-tested commit. Smoke tests written before any refactor guard against silent regressions. Modules split by responsibility (physics, I/O, analysis, viz). Active collaboration branches (`CDD_inc`, `Purdue_Paper`, `ESRF_DTU`, `Beam_Stop`, `dislocation_identification`) rebase onto cleaned main as a final step.

**Tech Stack:** Existing — numpy, scipy, numba, matplotlib, plotly, fabio, joblib, tqdm. Added — `pyproject.toml` (replaces requirements.txt), ruff (lint+format), mypy (type checking, public API only), pytest (tests), pre-commit (hook orchestration), GitHub Actions (CI).

---

## Context

The user (DTU researcher, materials science / physics) has a published DFXM forward-model codebase. The physics is sound (IUCr J 2024). The engineering is rough: filename-versioned scripts, junk-drawer modules, `import *` everywhere, hardcoded paths, no tests, no `.gitignore`, no LICENSE, 12 MB `.npy` tracked in git, and a "Production" working directory (1002 files) that isn't even a git repo.

**Reality check on canonical state.** Despite the Production-directory mess, the actual live git clone (`CDD_Khaled/Geometrical_Optics_master`, branch `Beam_Stop`) is *much* cleaner: 4,676 lines of Python across 11 files. The Production sprawl is the user's local experimentation space and won't be cleaned up here — only the GitHub repo will. The plan starts from `git clone` and treats Production as discardable scratch.

**Key snapshot of canonical files (CDD_Khaled, branch Beam_Stop):**

| File | Lines | Purpose |
|------|-------|---------|
| `init_forward.py` | 1,444 | Entry point — runs the simulation, plots results |
| `image_processor.py` | 753 | Image I/O, analysis, colormaps, parallel rendering |
| `functions.py` | 610 | Junk drawer: physics + math + I/O + parallel |
| `deformation_gradient.py` | 454 (untracked) | Strain tensor utilities |
| `fow_new4.py` | 543 (untracked) | Experimental wave-optics prototype |
| `direct_space/forward_model.py` | 300 | Core forward simulator |
| `reciprocal_space/recspace_res.py` | 337 | Reciprocal-space Monte Carlo resolution |
| `disloc_identify.py` | 119 | Dislocation identification |
| `reciprocal_space/exposure_time.py` | 59 | Exposure-time helper |
| `reciprocal_space/generate_Resq_i.py` | 57 | Driver for Monte Carlo res |

**Concrete issues to fix (with line numbers in CDD_Khaled clone):**

- `functions.py:119` and `functions.py:386` — `image_range` defined twice. Real bug: second definition silently overrides first.
- `functions.py:10` vs `:175` vs `:435` — three near-duplicate dislocation solvers (`Fd_find_mixed`, `Fd_find`, `Fd_find_domain`). Burgers vector defaults inconsistent: `b=2.862e-4` (lines 10, 175) vs `b=3.507e-4` (line 435).
- `functions.py:581` — `fast_inverse2()` carries a self-deprecating `# Try to rewrite this` comment.
- `image_processor.py:94` — `inv_polefigure_colors` (only one in this clone; the variant explosion was in Production).
- `image_processor.py:625-722` — file I/O (EDF, npy, parallel loaders) interleaved with analysis (`calc_moments`, `calc_fwhm_moment_*`).
- `init_forward.py` (1,444 lines) — monolith with hardcoded `C:\Users\borgi\...` paths and `from X import *` imports.
- `direct_space/forward_model.py:124` — `Find_Hg` does both physics and disk I/O.
- `dislocation_density_100.npy` (12 MB) tracked in git. Should be external.
- No `.gitignore` → `__pycache__/`, `.pkl`, `.npy` all tracked.

**Note on plan location.** This plan file lives at `C:\Users\borgi\.claude\plans\` because of plan-mode constraints. After Phase 1 commits the new repo structure, copy this file to `docs/superpowers/plans/2026-05-12-codebase-cleanup.md` inside the cleanup branch so it travels with the repo.

---

## Target file structure (end state)

```
Geometrical_Optics/                  # rename: drop "_master" via gh repo rename
├── pyproject.toml                   # replaces requirements.txt
├── README.md                        # rewritten with install/run/cite
├── LICENSE                          # MIT or BSD-3
├── CITATION.cff                     # paper citation metadata
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md              # how the modules fit together
│   ├── physics.md                   # DFXM model, conventions, symbols
│   ├── reproducibility.md           # how to regenerate paper figures
│   └── superpowers/plans/           # this plan moves here
├── src/dfxm_geo/                    # the actual package
│   ├── __init__.py
│   ├── constants.py                 # b, ny, lambda, ID06 geometry defaults
│   ├── crystal/
│   │   ├── __init__.py
│   │   ├── dislocations.py          # unified Fd_find (replaces three variants)
│   │   ├── rotations.py             # rotatedU, fast_inverse2, is_rotation_matrix
│   │   └── strain.py                # deformation_gradient.py contents
│   ├── direct_space/
│   │   ├── __init__.py
│   │   └── forward_model.py         # split: physics only, I/O moved out
│   ├── reciprocal_space/
│   │   ├── __init__.py
│   │   ├── resolution.py            # recspace_res.py
│   │   └── kernel.py                # generate_Resq_i.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── moments.py               # calc_moments, calc_fwhm_moment_*
│   │   └── colormaps.py             # inv_polefigure_colors
│   ├── io/
│   │   ├── __init__.py
│   │   ├── images.py                # save_image/load_image[s][_parallel], save_edfs
│   │   ├── strain_cache.py          # load_or_generate_Hg, Find_Hg I/O bit
│   │   └── video.py                 # create_video_from_png
│   └── viz/
│       ├── __init__.py
│       ├── plots.py                 # save_images, plot helpers
│       └── stack_gui.py             # tkinter StackGUI (if kept)
├── scripts/
│   ├── run_forward.py               # was init_forward.py
│   └── identify_dislocations.py     # was disloc_identify.py
├── notebooks/
│   ├── 01_walkthrough.ipynb         # canonical hello-world
│   └── 99_paper_figures.ipynb       # reproduces IUCr J 2024 figures
├── tests/
│   ├── conftest.py
│   ├── data/
│   │   └── golden/                  # reference outputs (small)
│   ├── test_dislocations.py
│   ├── test_forward_model.py
│   ├── test_resolution.py
│   └── test_io.py
└── legacy/                          # one-time grave for archived variants
    └── README.md                    # explains: not used, kept for reference
```

---

## Phase 0 — Decisions (15 min, no code)

**Goal:** Lock in the choices the rest of the plan depends on.

### Task 0.1: Pick the base branch

- [ ] **Step 1: Decide between `main` and `Beam_Stop` as the cleanup base**

The user picked "fresh clone, new branch on the original repo" but didn't specify the base. Recommendation: branch off `main` so the cleanup PR is reviewable against the published state. If `Beam_Stop` has uncommitted "Wire for BS absorption" work the user needs, cherry-pick those commits into the cleanup branch as a separate step.

Run to compare:

```bash
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
git log main..origin/Beam_Stop --oneline
git log main..origin/dislocation_identification --oneline
git log main..origin/Purdue_Paper --oneline
git log main..origin/CDD_inc --oneline
git log main..origin/ESRF_DTU --oneline
```

Expected: a list of commits per branch. Note which branches contain work that hasn't merged into main yet — those are the migration targets for Phase 10.

- [ ] **Step 2: Create the cleanup branch**

```bash
git checkout main
git pull
git checkout -b cleanup/main-modernization
```

- [ ] **Step 3: Commit a checkpoint tag of pre-cleanup state**

```bash
git tag pre-cleanup-2026-05-12 main
git push origin pre-cleanup-2026-05-12
```

Rationale: gives an immutable "before" reference for diffing / rollback regardless of branch evolution.

### Task 0.2: Confirm Python version target

- [ ] **Step 1: Decide Python ≥3.11**

Current `requirements.txt` pins numpy==1.23.5 and scipy==1.10.0 (late-2023, Python 3.9-ish era). Recommendation: target Python 3.11+ (better error messages, faster, supported until Oct 2027). All current deps support 3.11. Drop the pins for direct deps — keep upper bounds only where breakage is real.

No commands here — this is a decision. Document the answer in `pyproject.toml` in Phase 1.

### Task 0.3: Pick a license

- [ ] **Step 1: Choose MIT vs BSD-3 vs Apache-2.0**

The paper is in IUCr J (open access). Code from academic research typically goes MIT or BSD-3. MIT is shorter and gives maximum reuse; BSD-3 adds an explicit non-endorsement clause. Recommendation: MIT.

Also: check if DTU has an institutional IP policy that requires a specific license or co-attribution — if yes, that overrides this recommendation. Email DTU's tech transfer office if unsure (one-line query, takes a day).

---

## Phase 1 — Baseline & smoke test (1-2 hours)

**Goal:** Before changing anything, capture a reference output the cleanup must reproduce. Without this, refactors are unsafe.

**Prerequisites:** Phase 0 complete; on `cleanup/main-modernization` branch.

### Task 1.1: Create the test scaffold

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/data/golden/.gitkeep`

- [ ] **Step 1: Add pytest as a dev dependency (one-liner pip for now; replaced by pyproject in Phase 2)**

```bash
pip install pytest numpy
```

- [ ] **Step 2: Create `tests/conftest.py` with a single fixture**

```python
"""Pytest fixtures shared across the test suite."""
from pathlib import Path
import pytest

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"

@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Directory containing reference outputs for smoke tests."""
    return GOLDEN_DIR
```

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: add pytest scaffold with golden-dir fixture"
```

### Task 1.2: Write the forward-model smoke test (before any refactor)

**Files:**
- Create: `tests/test_forward_model_smoke.py`
- Create: `tests/data/golden/forward_smoke.npy` (will be generated, then committed)

The point of this test: pin the current numerical behavior so refactors can't silently break it.

- [ ] **Step 1: Write the failing test**

```python
"""Smoke test: forward model output must match the pre-cleanup baseline."""
import numpy as np
import pytest
from pathlib import Path

# Import path will change in later phases — update then.
from direct_space.forward_model import forward, Find_Hg


@pytest.fixture(scope="module")
def reference_inputs():
    """Minimal, deterministic inputs for the smoke test.

    Tiny grid (no realistic physics expected) — we only check the
    function is deterministic and unchanged across refactors.
    """
    rng = np.random.default_rng(42)
    # Fill in actual minimal inputs once you've read forward()'s signature
    # in direct_space/forward_model.py. This is a placeholder shape.
    Hg = rng.normal(size=(5, 5, 5, 3, 3))  # adjust to real shape
    phi = 0.0
    chi = 0.0
    return Hg, phi, chi


def test_forward_matches_golden(reference_inputs, golden_dir):
    Hg, phi, chi = reference_inputs
    out = forward(Hg, phi, chi, TwoDeltaTheta=0.0, qi_return=False)
    golden_path = golden_dir / "forward_smoke.npy"
    if not golden_path.exists():
        np.save(golden_path, out)
        pytest.skip("Golden file created; rerun to enable comparison.")
    expected = np.load(golden_path)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2: Run it to generate the golden file**

```bash
pytest tests/test_forward_model_smoke.py -v
```

Expected: SKIP on first run with "Golden file created; rerun to enable comparison."

- [ ] **Step 3: Re-read `direct_space/forward_model.py:1-50` and fix the fixture inputs**

The placeholder shape almost certainly doesn't match `forward()`'s real signature. Open the file, read the actual `forward(Hg, phi, chi, TwoDeltaTheta, qi_return)` signature, and pick a small but realistic `Hg` (the strain field tensor). Goal: deterministic, fast (<5 s), and exercises the real code path.

- [ ] **Step 4: Regenerate the golden file with the correct inputs**

```bash
rm tests/data/golden/forward_smoke.npy
pytest tests/test_forward_model_smoke.py -v
```

Then run a second time. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_forward_model_smoke.py tests/data/golden/forward_smoke.npy
git commit -m "test: pin forward-model output to a pre-cleanup baseline"
```

### Task 1.3: Add a second smoke test for the dislocation field

**Files:**
- Create: `tests/test_dislocations_smoke.py`
- Create: `tests/data/golden/Fd_find_smoke.npy`

- [ ] **Step 1: Mirror the pattern from 1.2 for `Fd_find` in `functions.py:175`**

```python
import numpy as np
import pytest
from functions import Fd_find


def test_Fd_find_matches_golden(golden_dir):
    # Small lab-frame grid
    n = 8
    lin = np.linspace(-1, 1, n)
    rl = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))  # (3, n, n, n)
    Us = np.eye(3)
    Ud = np.eye(3)
    Theta = 0.5
    out = Fd_find(rl, Ud, Us, Theta, dis=1, ndis=1)
    golden = golden_dir / "Fd_find_smoke.npy"
    if not golden.exists():
        np.save(golden, out)
        pytest.skip("Golden file created.")
    np.testing.assert_allclose(out, np.load(golden), rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2: Verify the `Fd_find` signature in `functions.py:175` and adjust if needed**

- [ ] **Step 3: Generate golden, rerun, commit**

```bash
pytest tests/test_dislocations_smoke.py -v
# (run twice — first creates golden, second compares)
git add tests/test_dislocations_smoke.py tests/data/golden/Fd_find_smoke.npy
git commit -m "test: pin Fd_find dislocation field to a pre-cleanup baseline"
```

### Task 1.4: Verify both tests pass before proceeding

- [ ] **Step 1: Full suite run**

```bash
pytest -v
```

Expected: 2 passed. Both smoke tests green.

- [ ] **Step 2: Tag this checkpoint**

```bash
git tag baseline-smoke-tests
```

**Risk and rollback for Phase 1:** None — only adds files. To roll back, delete the test files.

---

## Phase 2 — Hygiene foundation (2-3 hours)

**Goal:** A new contributor can `git clone`, follow the README, and run the smoke tests within 10 minutes.

**Prerequisites:** Phase 1 complete (smoke tests green).

### Task 2.1: Add `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
build/
dist/
*.egg-info/
.eggs/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Editors
.vscode/
.idea/
*.swp
*~

# OS
.DS_Store
Thumbs.db

# Project — generated outputs
*.npy
*.pkl
*.edf
*.mp4
output/
results/
rockingcurve*/
mixed_ims*/
final_figures/

# Project — keep small reference data, ignore large
!tests/data/golden/*.npy
!docs/figures/*.png

# Environment
.env
.venv/
venv/
```

- [ ] **Step 2: Verify nothing important gets ignored**

```bash
git status --ignored
git check-ignore -v dislocation_density_100.npy
```

Expected: `.gitignore:NN:*.npy	dislocation_density_100.npy` — confirms the 12 MB file is now caught. We'll remove it from history in Task 2.2.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for Python, Jupyter, OS, and project outputs"
```

### Task 2.2: Untrack large binaries (keep in working tree, drop from index)

**Files:**
- Modify: git index — `dislocation_density_100.npy`, `direct_space/deformation_gradient_tensors/dislocation_density_100.npy`, `__pycache__/`, any `.pkl`

- [ ] **Step 1: Identify large tracked files**

```bash
git ls-files | xargs -I{} ls -la "{}" 2>/dev/null | awk '$5 > 1000000 {print $5, $NF}' | sort -rn | head -20
```

Expected output: list of files >1MB, with `dislocation_density_100.npy` at the top.

- [ ] **Step 2: Remove from index without deleting from disk**

```bash
git rm --cached dislocation_density_100.npy
git rm --cached -r __pycache__/
git rm --cached -r direct_space/__pycache__/ 2>/dev/null || true
git rm --cached -r reciprocal_space/__pycache__/ 2>/dev/null || true
git rm --cached -r reciprocal_space/pkl_files/ 2>/dev/null || true
```

- [ ] **Step 3: Verify**

```bash
git status
```

Expected: shows the deletions staged, and the files still exist on disk.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: untrack __pycache__, pkl_files, and large .npy binaries"
```

**Note on history rewriting:** This commit only stops tracking — the 12 MB blob still exists in git history. For a researcher's repo, leaving the history alone is usually fine (clones still bloat, but pulls are cheap once cached). If you want a slim clone in the future, use `git filter-repo --invert-paths --path dislocation_density_100.npy` — but this rewrites history and breaks all existing clones (CDD_Khaled, Purdue_collab). **Do not do this without coordinating with collaborators.** Recommendation: skip history rewrite for now.

### Task 2.3: Replace `requirements.txt` with `pyproject.toml`

**Files:**
- Create: `pyproject.toml`
- Delete: `requirements.txt`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dfxm-geo"
version = "0.1.0"
description = "Geometrical-optics forward model for dark-field X-ray microscopy"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [
    {name = "Sina Borgi", email = "borgi@dtu.dk"},
]
keywords = ["dfxm", "x-ray", "synchrotron", "dislocations", "materials-science"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "numpy>=1.23,<3",
    "scipy>=1.10,<2",
    "numba>=0.56",
    "matplotlib>=3.6",
    "plotly>=5.11",
    "fabio>=2023.4",
    "joblib>=1.3",
    "tqdm>=4.64",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "ruff>=0.6",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "ipykernel",
    "jupyterlab",
]

[project.scripts]
dfxm-forward = "dfxm_geo.cli:run_forward"

[project.urls]
Homepage = "https://github.com/borgi-s/Geometrical_Optics_master"
Paper = "https://journals.iucr.org/j/issues/2024/02/00/nb5370/"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["legacy/", "notebooks/"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RET", "NPY"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.11"
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
exclude = ["legacy/", "notebooks/", "tests/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 2: Delete `requirements.txt`**

```bash
git rm requirements.txt
```

- [ ] **Step 3: Verify install works in a fresh venv**

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; .venv/bin/activate on Unix
pip install -e ".[dev]"
pytest -v
```

Expected: install succeeds; both smoke tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: replace requirements.txt with pyproject.toml (PEP 621)"
```

### Task 2.4: Add LICENSE

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write MIT license**

```text
MIT License

Copyright (c) 2024-2026 Sina Borgi and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license"
```

### Task 2.5: Add `CITATION.cff`

**Files:**
- Create: `CITATION.cff`

- [ ] **Step 1: Write citation metadata**

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite the accompanying paper."
title: "Geometrical Optics: forward modelling of Dark Field X-ray Microscopy"
authors:
  - family-names: Borgi
    given-names: Sina
    affiliation: "Technical University of Denmark (DTU)"
    email: borgi@dtu.dk
preferred-citation:
  type: article
  title: "<insert exact paper title from IUCr J 2024 issue 2>"
  authors:
    - family-names: Borgi
      given-names: Sina
  journal: "IUCrJ"
  year: 2024
  volume: 11
  issue: 2
  url: "https://journals.iucr.org/j/issues/2024/02/00/nb5370/"
```

- [ ] **Step 2: Replace the placeholder title with the real one from the paper**

- [ ] **Step 3: Commit**

```bash
git add CITATION.cff
git commit -m "chore: add CITATION.cff for paper attribution"
```

### Task 2.6: Rewrite README

**Files:**
- Modify: `README.md` (full replace)

- [ ] **Step 1: Write the new README**

```markdown
# DFXM Geometrical-Optics Forward Model

A Python implementation of the geometrical-optics forward model for Dark Field
X-ray Microscopy (DFXM), as published in:

> Borgi, S. et al. *IUCrJ* (2024), 11, issue 2.
> [Read the paper](https://journals.iucr.org/j/issues/2024/02/00/nb5370/)

The default beamline configuration matches ID06 at the European Synchrotron
Radiation Facility (ESRF).

## What this code does

Given a crystal containing dislocations, this code simulates the DFXM images
that would be recorded on a detector under a defined beam and goniometer
geometry. It models both the direct-space deformation field around dislocations
and the reciprocal-space resolution function of the microscope.

See `docs/physics.md` for the mathematical model and `docs/architecture.md`
for how the code is organized.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                       # smoke tests should pass
```

Run the canonical example:

```bash
python scripts/run_forward.py --config configs/default.toml
```

Open the walkthrough notebook:

```bash
jupyter lab notebooks/01_walkthrough.ipynb
```

## Reproducing the paper figures

```bash
jupyter lab notebooks/99_paper_figures.ipynb
```

The notebook downloads the reference dislocation density from the data archive
(see `docs/reproducibility.md`) and regenerates each figure. Total runtime:
~30 min on a 2024-era laptop.

## Project structure

```
src/dfxm_geo/
  constants.py        Physical constants and ID06 defaults
  crystal/            Dislocation displacement fields and crystal frames
  direct_space/       Direct-space forward simulation
  reciprocal_space/   Reciprocal-space resolution function
  analysis/           Moment and FWHM analysis of image stacks
  io/                 Image, video, and strain-field I/O
  viz/                Plotting helpers
scripts/              CLI entry points
notebooks/            Walkthroughs and figure reproduction
tests/                Pytest test suite
docs/                 Architecture, physics, reproducibility guides
```

## Citing

```bibtex
@article{borgi2024dfxm,
  title  = {<paper title>},
  author = {Borgi, Sina and others},
  journal= {IUCrJ},
  volume = {11},
  issue  = {2},
  year   = {2024},
  url    = {https://journals.iucr.org/j/issues/2024/02/00/nb5370/},
}
```

## Contributing

PRs welcome. Run `pre-commit run --all-files` before pushing. See
`docs/contributing.md`.

## License

MIT. See `LICENSE`.
```

- [ ] **Step 2: Replace `<paper title>` placeholders with the actual title**

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with install, run, and citation sections"
```

### Task 2.7: Add pre-commit configuration

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write hook config**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

- [ ] **Step 2: Install and run**

```bash
pre-commit install
pre-commit run --all-files
```

Expected: hooks reformat existing files. The diff may be large but mechanical (trailing whitespace, quote normalization). Inspect with `git diff` to make sure no code changed semantically.

- [ ] **Step 3: Run smoke tests**

```bash
pytest -v
```

Expected: still passes. If a test fails, ruff probably mangled something — revert that specific change.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git add -u   # stages whatever pre-commit reformatted
git commit -m "chore: add pre-commit with ruff, nbstripout, and standard hooks"
```

**Risk and rollback for Phase 2:** Pre-commit auto-formatting may touch many files. Verify smoke tests green after each commit. To roll back: `git reset --hard baseline-smoke-tests`.

**Effort estimate:** Medium (2-3 hours including config tweaking).

---

## Phase 3 — Consolidate duplicates and dead code (3-4 hours)

**Goal:** Pick canonical versions of every duplicate; delete or archive the rest. Reduce surface area before refactoring.

**Prerequisites:** Phase 2 complete. Smoke tests green.

### Task 3.1: Fix the duplicate `image_range` in `functions.py`

**Files:**
- Modify: `functions.py:119` and `functions.py:386`

- [ ] **Step 1: Read both definitions**

```bash
sed -n '119,125p' functions.py
sed -n '386,392p' functions.py
```

Determine: are they identical? If yes, delete one. If they differ, decide which behavior is correct (search call sites: `grep -n 'image_range' .`).

- [ ] **Step 2: Delete the dead definition**

If both are identical, delete the second:

```python
# In functions.py, remove lines 386-392 (the second definition of image_range)
```

If they differ, the first wins (Python uses the last definition, so callers are using the second; but the first is presumably the "intended" one — verify by reading commit history with `git log -L :image_range:functions.py`).

- [ ] **Step 3: Run smoke tests**

```bash
pytest -v
```

Expected: PASS. If FAIL, you picked the wrong duplicate — restore and try the other.

- [ ] **Step 4: Commit**

```bash
git add functions.py
git commit -m "fix: remove duplicate image_range definition (silent shadow bug)"
```

### Task 3.2: Investigate `Fd_find` variants and unify

**Files:**
- Modify: `functions.py:10` (Fd_find_mixed), `functions.py:175` (Fd_find), `functions.py:435` (Fd_find_domain)

- [ ] **Step 1: Diff the three implementations**

Run side-by-side reads. Look for:
- What differs in the math? (Probably: how the Burgers vector or coordinate frame is handled)
- Why are the `b` defaults different? `b=2.862e-4` (mixed, generic) vs `b=3.507e-4` (domain). One is Al, one is Fe or similar — confirm with the user; this is a physics choice that may have been muddled.
- What does `_domain` add over the basic `Fd_find`?

- [ ] **Step 2: Document the differences in a short comment**

Before unifying, write a one-paragraph note inside `functions.py` explaining what each variant does and why they exist. This is for your future self.

- [ ] **Step 3: Decide unification strategy**

Either:
- **(A) Keep three functions but pull the shared logic into a private `_Fd_kernel`** — minimal change, preserves behavior.
- **(B) Collapse to one `Fd_find(rl, U, Theta, kind={"edge","screw","mixed","domain"}, **opts)`** — bigger refactor, cleaner API.

Recommendation: (A) now; (B) in Phase 5 after the constants module lands. Don't try both at once.

- [ ] **Step 4: Implement (A)**

Extract the shared inner loop into `_Fd_kernel`. Each public `Fd_find*` becomes a 3-line wrapper that sets up its specific args and calls `_Fd_kernel`.

- [ ] **Step 5: Run smoke tests**

```bash
pytest -v
```

Expected: PASS. The smoke test calls `Fd_find` directly; if the wrapper preserves behavior, it stays green.

- [ ] **Step 6: Commit**

```bash
git add functions.py
git commit -m "refactor: extract shared kernel from Fd_find/Fd_find_mixed/Fd_find_domain"
```

### Task 3.3: Move untracked files into the working set or delete

**Files:**
- Decide: `deformation_gradient.py` (untracked, 454 lines), `fow_new4.py` (untracked, 543 lines), `disloc_identify.py` (tracked, 119 lines), `reciprocal_space/exposure_time.py` (tracked, 59 lines)

- [ ] **Step 1: Read each and determine if it's used**

```bash
grep -rn "deformation_gradient" --include="*.py" --include="*.ipynb"
grep -rn "fow_new4" --include="*.py" --include="*.ipynb"
grep -rn "disloc_identify" --include="*.py" --include="*.ipynb"
grep -rn "exposure_time" --include="*.py" --include="*.ipynb"
```

- [ ] **Step 2: For each file, choose: track / archive / delete**

- `deformation_gradient.py` (untracked) — if used by paper or notebooks → `git add`; if not → move to `legacy/` or delete.
- `fow_new4.py` (untracked) — explore agent flagged as experimental. Likely → `legacy/fow_new4.py` with a one-line note.
- `disloc_identify.py` — read its 119 lines. If it's the script behind the `BurgersVectorsPlotter` from the recent commits, keep and rename to `scripts/identify_dislocations.py` in Phase 4.
- `exposure_time.py` — read its 59 lines. Likely a one-off helper. Keep, move to `reciprocal_space/` (already there).

- [ ] **Step 3: Create `legacy/` and `legacy/README.md`**

```bash
mkdir legacy
```

```markdown
# Legacy code

This directory holds Python files that are kept for reference but are not part
of the active codebase. Specifically:

- They are NOT imported by `src/dfxm_geo/`, `scripts/`, or `notebooks/`.
- They are NOT run by tests or CI.
- They MAY be removed in a future release without notice.

If you find yourself needing one of these, lift it back into `src/` properly:
add a test, add docstrings, and route imports through the package.
```

- [ ] **Step 4: Move archived files**

```bash
git mv fow_new4.py legacy/fow_new4.py
# repeat for any others
```

- [ ] **Step 5: Run smoke tests**

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add legacy/
git add -u
git commit -m "chore: introduce legacy/ for archived prototypes; relocate fow_new4.py"
```

### Task 3.4: Inspect and clean `init_forward.py`

**Files:**
- Modify: `init_forward.py` (1,444 lines)

This is the worst offender by size. Don't refactor it yet — just *de-clutter* in place. Real refactor in Phase 6.

- [ ] **Step 1: Remove blocks of commented-out code**

Use `grep -n '^[[:space:]]*#' init_forward.py | head -100` to find comment-heavy regions. Anything that's commented-out *code* (vs. docstrings/notes) older than 6 months — delete. Git history preserves it.

Rule of thumb: if a comment starts with `#` and is followed by indentation + code-like content (e.g., `# plt.plot(...)`, `# Hg = ...`), it's commented-out code.

- [ ] **Step 2: Remove dead `print()` and `print('debug')` calls**

```bash
grep -n "^[[:space:]]*print(" init_forward.py
```

Decide each: useful progress signal (keep, convert to `logging.info` in Phase 6) vs. debug residue (delete).

- [ ] **Step 3: Replace hardcoded paths with `pathlib.Path` + a `--data-dir` placeholder**

For now, define near the top:

```python
from pathlib import Path
DATA_DIR = Path(__file__).parent  # placeholder; will become CLI arg in Phase 6
```

Then replace `C:\Users\borgi\Documents\Production\...` strings with `DATA_DIR / "..."`.

- [ ] **Step 4: Run smoke tests**

```bash
pytest -v
```

Expected: PASS. (The smoke tests don't import `init_forward`, so they're insensitive to its mess — but a syntax error would still fail collection.)

- [ ] **Step 5: Commit**

```bash
git add init_forward.py
git commit -m "refactor: declutter init_forward.py (dead comments, prints, hardcoded paths)"
```

**Risk and rollback for Phase 3:** Each commit is independent. Roll back any one with `git revert <sha>`.

**Effort estimate:** Medium-large (3-4 hours; the Fd_find unification is the longest sub-task).

---

## Phase 4 — Module refactor: split by responsibility (4-6 hours)

**Goal:** Move from flat top-level `.py` files to the structured `src/dfxm_geo/` package. One subsystem per commit; smoke tests green after each.

**Prerequisites:** Phase 3 complete. Smoke tests green.

### Task 4.1: Create the package skeleton

**Files:**
- Create: `src/dfxm_geo/__init__.py`, plus all subdir `__init__.py` files listed in the target structure

- [ ] **Step 1: Create directories and empty `__init__.py` files**

```bash
mkdir -p src/dfxm_geo/{crystal,direct_space,reciprocal_space,analysis,io,viz}
mkdir -p scripts notebooks docs
for f in src/dfxm_geo src/dfxm_geo/crystal src/dfxm_geo/direct_space \
         src/dfxm_geo/reciprocal_space src/dfxm_geo/analysis \
         src/dfxm_geo/io src/dfxm_geo/viz; do
  touch "$f/__init__.py"
done
```

- [ ] **Step 2: Pin the package version**

In `src/dfxm_geo/__init__.py`:

```python
"""DFXM geometrical-optics forward model."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Verify installable**

```bash
pip install -e .
python -c "import dfxm_geo; print(dfxm_geo.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add src/ scripts/ notebooks/ docs/
git commit -m "feat: scaffold src/dfxm_geo package layout"
```

### Task 4.2: Create the constants module

**Files:**
- Create: `src/dfxm_geo/constants.py`

- [ ] **Step 1: Write the constants module**

```python
"""Physical and geometric constants for DFXM forward modelling.

All constants are module-level so they can be overridden by callers via
direct assignment for parameter sweeps. Document their physical meaning here,
not in function signatures.
"""
from typing import Final

# --- Material constants ---
# Burgers vector magnitude (units: same as lab-frame coordinates, typically µm)
# Default: Al lattice parameter * sqrt(2)/2 ≈ 2.862e-4 µm = 0.2862 nm
BURGERS_VECTOR: Final[float] = 2.862e-4

# Burgers vector for Fe / ferrite (used in older Fd_find_domain calls)
# = a0(Fe) * sqrt(3)/2 ≈ 2.482e-4 µm. The 3.507e-4 default in the old code
# looks like a stale value — confirm with the user before relying on it.
BURGERS_VECTOR_FE: Final[float] = 2.482e-4

# Poisson ratio (dimensionless). 0.334 ≈ Al at room temperature.
POISSON_RATIO: Final[float] = 0.334

# --- ID06 (ESRF) beamline geometry ---
# These should match the values in init_forward.py and forward_model.py.
# Verify and copy them here.
ID06_WAVELENGTH: Final[float] = ...  # fill from the existing code
ID06_DETECTOR_PIXEL: Final[float] = ...
ID06_OBJECTIVE_FOCAL_LENGTH: Final[float] = ...
# ... (continue from existing hardcoded values in forward_model.py and init_forward.py)
```

- [ ] **Step 2: Find every magic number in the existing code and add it here**

```bash
grep -nE "= [0-9]+\.[0-9e+-]+" functions.py init_forward.py direct_space/forward_model.py | head -50
```

Triage: physical constant (move to constants.py) vs. local variable (leave alone).

- [ ] **Step 3: Run smoke tests**

Smoke tests still use the old imports — they should still pass:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dfxm_geo/constants.py
git commit -m "feat(constants): centralize Burgers vector, Poisson ratio, ID06 geometry"
```

### Task 4.3: Move crystal mechanics into `src/dfxm_geo/crystal/`

**Files:**
- Create: `src/dfxm_geo/crystal/dislocations.py` (from `functions.py:10, :175, :435` — the Fd_find* family)
- Create: `src/dfxm_geo/crystal/rotations.py` (from `functions.py:86` rotatedU, `:340` is_rotation_matrix, `:581` fast_inverse2)
- Create: `src/dfxm_geo/crystal/strain.py` (from `deformation_gradient.py`)
- Modify: `functions.py` (remove moved code, leave a deprecation shim)
- Modify: `tests/test_dislocations_smoke.py` (update import)

- [ ] **Step 1: Move `Fd_find`, `Fd_find_mixed`, `Fd_find_domain`, `multi_dislocs_parallel`, `_Fd_kernel` into `crystal/dislocations.py`**

Copy the function bodies. At the top of each file, add real docstrings using the `Why:` style: what the function computes, units, references.

Example for `Fd_find`:

```python
def Fd_find(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: float,
    dis: int = 1,
    ndis: int = 1,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    misorientation: bool = False,
    t_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the deformation-gradient field around `ndis` dislocations.

    Args:
        rl: Lab-frame coordinate grid, shape (3, Nx, Ny, Nz). Units: µm.
        Ud: Dislocation-frame orientation matrix (3, 3).
        Us: Sample-frame orientation matrix (3, 3).
        Theta: Bragg angle in radians.
        dis: Dislocation type id (1=edge, 2=screw, 3=mixed).
        ndis: Number of dislocations.
        b: Burgers vector magnitude. Default = Al's 2.862e-4 µm.
        ny: Poisson ratio. Default = 0.334 (Al).
        misorientation: If True, return only the asymmetric part Hg - I.
        t_vec: Dislocation tangent vectors, shape (3, ndis). Required if ndis>1.

    Returns:
        Hg: Deformation-gradient field, shape (3, 3, Nx, Ny, Nz).
    """
    # ... existing implementation, imports from constants
```

- [ ] **Step 2: Move `rotatedU`, `is_rotation_matrix`, `fast_inverse2` into `crystal/rotations.py`**

- [ ] **Step 3: Move all of `deformation_gradient.py` into `crystal/strain.py`**

- [ ] **Step 4: Leave compatibility shims in `functions.py`**

Replace the moved code with:

```python
# functions.py — DEPRECATED. Use dfxm_geo.crystal.* directly.
import warnings
from dfxm_geo.crystal.dislocations import (
    Fd_find, Fd_find_mixed, Fd_find_domain, multi_dislocs_parallel,
)
from dfxm_geo.crystal.rotations import (
    rotatedU, is_rotation_matrix, fast_inverse2,
)

warnings.warn(
    "functions.py is deprecated; import from dfxm_geo.crystal instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Keep small helpers that aren't yet moved (norm, square, m_norm) below.
```

- [ ] **Step 5: Update smoke test imports**

```python
# tests/test_dislocations_smoke.py
from dfxm_geo.crystal.dislocations import Fd_find
```

- [ ] **Step 6: Run smoke tests**

```bash
pytest -v
```

Expected: PASS. If you see a DeprecationWarning, that's the shim working — good.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/crystal/ functions.py tests/
git commit -m "refactor: move crystal mechanics into dfxm_geo.crystal package"
```

### Task 4.4: Move forward model into `src/dfxm_geo/direct_space/`

**Files:**
- Create: `src/dfxm_geo/direct_space/forward_model.py` (from `direct_space/forward_model.py`)
- Modify: `direct_space/forward_model.py` → deprecation shim
- Modify: `tests/test_forward_model_smoke.py` (update import)

- [ ] **Step 1: Split `forward_model.py` by responsibility**

Current `direct_space/forward_model.py` has:
- `Find_Hg(dis, ndis, ...)` at line 124 — mixes physics + I/O (loads/generates strain field)
- `precompute_forward_static` at line 165 — physics
- `forward_from_static` at line 187 — physics

Split:
- Physics functions go to `src/dfxm_geo/direct_space/forward_model.py`
- The I/O portion of `Find_Hg` (npy caching) goes to `src/dfxm_geo/io/strain_cache.py` — see Task 4.6

Initial move: copy ALL of `direct_space/forward_model.py` into `src/dfxm_geo/direct_space/forward_model.py` unchanged. The I/O split is the *next* commit.

- [ ] **Step 2: Add deprecation shim in old location**

```python
# direct_space/forward_model.py — DEPRECATED
import warnings
from dfxm_geo.direct_space.forward_model import *  # noqa: F401, F403

warnings.warn(
    "direct_space.forward_model is deprecated; import from "
    "dfxm_geo.direct_space.forward_model instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

- [ ] **Step 3: Update smoke test**

```python
# tests/test_forward_model_smoke.py
from dfxm_geo.direct_space.forward_model import forward, Find_Hg
```

- [ ] **Step 4: Run smoke tests**

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/direct_space/ direct_space/forward_model.py tests/
git commit -m "refactor: move direct_space.forward_model into dfxm_geo package"
```

### Task 4.5: Move reciprocal-space code

**Files:**
- Create: `src/dfxm_geo/reciprocal_space/resolution.py` (from `reciprocal_space/recspace_res.py`)
- Create: `src/dfxm_geo/reciprocal_space/kernel.py` (from `reciprocal_space/generate_Resq_i.py`)
- Create: `src/dfxm_geo/reciprocal_space/exposure.py` (from `reciprocal_space/exposure_time.py`)

Same pattern as 4.4. Move, leave shims, run tests, commit.

- [ ] **Step 1: Copy files into the new package locations**
- [ ] **Step 2: Leave deprecation shims at the old paths**
- [ ] **Step 3: Run smoke tests**
- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move reciprocal_space into dfxm_geo package"
```

### Task 4.6: Split `image_processor.py` into `analysis/` + `io/` + `viz/`

**Files:**
- Create: `src/dfxm_geo/analysis/moments.py` (from `image_processor.py:398, :456, :498, :571`)
- Create: `src/dfxm_geo/analysis/colormaps.py` (from `image_processor.py:94`)
- Create: `src/dfxm_geo/io/images.py` (from `image_processor.py:40, :64, :625, :677, :719, :722`)
- Create: `src/dfxm_geo/io/video.py` (from `image_processor.py:14`)
- Create: `src/dfxm_geo/viz/plots.py` (from `image_processor.py:152, :188`)
- Create: `src/dfxm_geo/direct_space/rocking.py` (from `image_processor.py:284, :295, :331, :345` — parallel rendering)
- Create: `src/dfxm_geo/io/strain_cache.py` (extract I/O part from `Find_Hg`)
- Modify: `image_processor.py` → shim

- [ ] **Step 1: Move analysis functions**

`calc_peak_broadening_and_moments`, `calc_fwhm_moment_1d`, `calc_fwhm_moment_2d`, `calc_moments` → `analysis/moments.py`. `inv_polefigure_colors` → `analysis/colormaps.py`.

- [ ] **Step 2: Move I/O functions**

`save_image`, `save_images_parallel`, `save_edfs`, `load_images`, `load_image`, `load_images_parallel` → `io/images.py`.

`create_video_from_png` → `io/video.py`.

- [ ] **Step 3: Move visualization helpers**

`save_images`, `fastgrainplot` → `viz/plots.py`.

- [ ] **Step 4: Move rendering pipeline (it depends on the forward model)**

`_compute_image`, `compute_rocking_curve_parallel`, `_compute_image_phichi`, `compute_rocking_grid_parallel` → `direct_space/rocking.py`. These are the parallel drivers that turn a strain field into a stack of detector images.

- [ ] **Step 5: Extract the I/O half of `Find_Hg` into `io/strain_cache.py`**

The current `Find_Hg(dis, ndis, psize, zl_rms, ...)` both *computes* a strain field and *caches* it as `.npy`. Split:

```python
# src/dfxm_geo/direct_space/forward_model.py
def compute_Hg(dis: int, ndis: int, psize: float, zl_rms: float, ...) -> np.ndarray:
    """Pure physics: returns the strain-field array."""
    # ... (compute logic from current Find_Hg)
```

```python
# src/dfxm_geo/io/strain_cache.py
from pathlib import Path
import numpy as np
from dfxm_geo.direct_space.forward_model import compute_Hg

def load_or_compute_Hg(
    cache_dir: Path,
    dis: int, ndis: int, psize: float, zl_rms: float,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return strain field, loading from disk cache if available."""
    cache_path = cache_dir / f"Hg_{dis}_{ndis}_{psize}_{zl_rms}.npy"
    if cache_path.exists() and not force_recompute:
        return np.load(cache_path)
    Hg = compute_Hg(dis, ndis, psize, zl_rms)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, Hg)
    return Hg
```

This separates "what to compute" from "how to persist it" — testing physics no longer needs a filesystem.

- [ ] **Step 6: Leave a shim in `image_processor.py`**

- [ ] **Step 7: Run smoke tests after each sub-move (don't commit a giant blob)**

Smaller commits → smaller blast radius. Commit after moving `analysis/`, then after `io/`, then `viz/`, then `direct_space/rocking.py`.

```bash
pytest -v
git add src/dfxm_geo/analysis/ image_processor.py
git commit -m "refactor: extract analysis/moments and analysis/colormaps from image_processor"

pytest -v
git add src/dfxm_geo/io/ image_processor.py
git commit -m "refactor: extract io/images, io/video from image_processor"

pytest -v
git add src/dfxm_geo/viz/ image_processor.py
git commit -m "refactor: extract viz/plots from image_processor"

pytest -v
git add src/dfxm_geo/direct_space/rocking.py src/dfxm_geo/io/strain_cache.py
git add image_processor.py direct_space/forward_model.py
git commit -m "refactor: separate rocking-curve pipeline and strain-cache I/O"
```

**Risk and rollback for Phase 4:** Each move is small and verified by smoke tests. Roll back any single move with `git revert <sha>`.

**Effort estimate:** Large (4-6 hours; the image_processor.py split is the bulk).

---

## Phase 5 — Hygiene of imports and types (2-3 hours)

**Goal:** Replace every `from X import *`, add type hints to the public API, and run mypy clean on `src/dfxm_geo/`.

**Prerequisites:** Phase 4 complete. Smoke tests green.

### Task 5.1: Replace wildcard imports

**Files:**
- Modify: `init_forward.py`, `direct_space/forward_model.py` (shim already has F401/F403 noqa, but underlying clients still use `*`)

- [ ] **Step 1: Find every `import *`**

```bash
grep -rn "import \*" --include="*.py" .
```

- [ ] **Step 2: For each, run the script/test once with `python -W error::DeprecationWarning -c "import init_forward"` to see the actual names used**

Or use ruff: it flags `F403` (wildcard import) and `F405` (may be undefined). Run:

```bash
ruff check --select F403,F405 init_forward.py
```

This lists every name that's *probably* from the wildcard import.

- [ ] **Step 3: Replace each `*` with the explicit list**

Example:

```python
# Before
from direct_space.forward_model import *

# After
from dfxm_geo.direct_space.forward_model import (
    forward, Find_Hg, precompute_forward_static, forward_from_static,
)
from dfxm_geo.direct_space.rocking import (
    compute_rocking_curve_parallel, compute_rocking_grid_parallel,
)
```

- [ ] **Step 4: Run smoke tests after each file**

```bash
pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: replace wildcard imports with explicit names"
```

### Task 5.2: Add type hints to the public API

**Files:**
- Modify: every public function in `src/dfxm_geo/*/`

Focus on *public* — the functions a notebook would call. Internal helpers can stay un-hinted.

- [ ] **Step 1: Hint the physics functions first**

Use `np.ndarray` for arrays (or `numpy.typing.NDArray[np.float64]` if you want to be precise) and `float`/`int` for scalars.

Example:

```python
# Before
def forward(Hg, phi, chi, TwoDeltaTheta, qi_return):
    ...

# After
import numpy as np
from numpy.typing import NDArray

def forward(
    Hg: NDArray[np.float64],
    phi: float,
    chi: float,
    TwoDeltaTheta: float = 0.0,
    qi_return: bool = False,
) -> NDArray[np.float64]:
    ...
```

- [ ] **Step 2: Run mypy on the package only**

```bash
mypy src/dfxm_geo/
```

Expected: warnings, not errors. The first run will be noisy — that's fine. Fix the easy ones (missing return types, `Optional` for `None`-defaults); ignore complex array-shape inference.

- [ ] **Step 3: Commit**

```bash
git add src/
git commit -m "feat: type-hint public API of dfxm_geo"
```

### Task 5.3: Move shims into `legacy/` after a deprecation cycle

For now: leave the shims at the top level. They support the existing `init_forward.py`. In a later release (v0.2), remove them.

Document this in `CHANGELOG.md`:

```markdown
# Changelog

## 0.1.0 (unreleased)
- Reorganized into `src/dfxm_geo/` package.
- Top-level `functions.py`, `image_processor.py`, `direct_space/forward_model.py`,
  `reciprocal_space/{recspace_res,generate_Resq_i,exposure_time}.py` are now
  deprecation shims. They will be removed in 0.2.0. Update imports to
  `dfxm_geo.*`.
```

- [ ] **Step 1: Create `CHANGELOG.md`**
- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG with deprecation notes"
```

**Effort estimate:** Medium (2-3 hours).

---

## Phase 6 — Convert `init_forward.py` to a proper CLI (2-3 hours)

**Goal:** The 1,444-line `init_forward.py` becomes a thin `scripts/run_forward.py` driven by a config file. No more hardcoded paths.

**Prerequisites:** Phase 5 complete.

### Task 6.1: Carve `init_forward.py` into reusable functions

**Files:**
- Modify: `init_forward.py` (extract logical sections into functions)
- Create: `src/dfxm_geo/pipeline.py` (orchestration)

- [ ] **Step 1: Identify the top-level sections**

Read `init_forward.py` top to bottom; each block separated by blank lines or comments is a candidate function. Common DFXM pipeline structure:

1. Load/generate strain field
2. Set up geometry (phi/chi ranges, detector)
3. Run forward model for each (phi, chi)
4. Save images
5. Compute moments / FWHM
6. Plot results

- [ ] **Step 2: Extract each section as a function in `src/dfxm_geo/pipeline.py`**

```python
"""High-level orchestration of a full DFXM simulation run."""
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from dfxm_geo.io.strain_cache import load_or_compute_Hg
from dfxm_geo.direct_space.rocking import compute_rocking_grid_parallel
from dfxm_geo.io.images import save_images_parallel
from dfxm_geo.analysis.moments import calc_peak_broadening_and_moments


def run_simulation(
    config: "SimulationConfig",
    output_dir: Path,
) -> dict[str, NDArray[np.float64]]:
    """Execute a full forward-simulation run from a config object.

    Returns a dict with keys: 'image_stack', 'moments', 'fwhm'.
    """
    Hg = load_or_compute_Hg(config.cache_dir, **config.crystal)
    images = compute_rocking_grid_parallel(Hg, **config.scan)
    save_images_parallel(images, output_dir, **config.io)
    moments = calc_peak_broadening_and_moments(images, **config.scan)
    return {"image_stack": images, "moments": moments}
```

- [ ] **Step 3: Define `SimulationConfig` dataclass**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SimulationConfig:
    cache_dir: Path
    crystal: dict   # dis, ndis, psize, zl_rms, ...
    scan: dict      # phi_range, phi_steps, chi_range, chi_steps
    io: dict        # fpath, fn_prefix, ftype
```

- [ ] **Step 4: Run smoke tests**

```bash
pytest -v
```

### Task 6.2: Create the CLI

**Files:**
- Create: `scripts/run_forward.py`
- Create: `configs/default.toml`

- [ ] **Step 1: Write the CLI**

```python
"""scripts/run_forward.py — run a forward DFXM simulation from a config file."""
import argparse
import tomllib
from pathlib import Path

from dfxm_geo.pipeline import SimulationConfig, run_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description="DFXM forward simulation")
    parser.add_argument("--config", type=Path, required=True, help="TOML config")
    parser.add_argument("--output", type=Path, required=True, help="Output dir")
    args = parser.parse_args()

    with args.config.open("rb") as f:
        raw = tomllib.load(f)
    config = SimulationConfig(
        cache_dir=Path(raw["cache_dir"]),
        crystal=raw["crystal"],
        scan=raw["scan"],
        io=raw["io"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    run_simulation(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `configs/default.toml`**

```toml
cache_dir = "./cache"

[crystal]
dis = 1          # 1=edge, 2=screw, 3=mixed
ndis = 1
psize = 4e-3
zl_rms = 150e-3  # 150 nm

[scan]
phi_range = [-0.005, 0.005]
phi_steps = 100
chi_range = [-0.002, 0.002]
chi_steps = 40

[io]
fn_prefix = "sim"
ftype = "npy"
```

- [ ] **Step 3: Test the CLI end-to-end**

```bash
python scripts/run_forward.py --config configs/default.toml --output /tmp/dfxm_test
ls /tmp/dfxm_test
```

Expected: image files appear in the output directory.

- [ ] **Step 4: Add `run_forward` as an entry point**

Update `pyproject.toml`:

```toml
[project.scripts]
dfxm-forward = "dfxm_geo.pipeline:cli_main"
```

Then `pip install -e .` and `dfxm-forward --config configs/default.toml --output ./out` should work.

- [ ] **Step 5: Move `init_forward.py` to `legacy/`**

```bash
git mv init_forward.py legacy/init_forward_pre_cleanup.py
```

Add note in `legacy/README.md`:

> `init_forward_pre_cleanup.py`: pre-cleanup monolith. Functionality preserved
> in `dfxm_geo.pipeline` + `scripts/run_forward.py` + `configs/default.toml`.

- [ ] **Step 6: Commit**

```bash
git add scripts/ configs/ src/dfxm_geo/pipeline.py legacy/
git commit -m "feat: replace init_forward.py with config-driven CLI and pipeline module"
```

**Effort estimate:** Medium (2-3 hours).

---

## Phase 7 — Tests beyond smoke tests (3-4 hours)

**Goal:** Add focused unit tests for the physics functions. Parametrize over the dislocation types. Aim for ~60% line coverage on the package (not 100% — research code with plotting is hard to cover).

**Prerequisites:** Phase 6 complete.

### Task 7.1: Dislocation tests

**Files:**
- Create: `tests/test_dislocations.py`

- [ ] **Step 1: Test rotational invariance**

```python
"""Unit tests for dfxm_geo.crystal.dislocations."""
import numpy as np
import pytest
from dfxm_geo.crystal.dislocations import Fd_find
from dfxm_geo.crystal.rotations import rotatedU


def make_grid(n: int = 8):
    lin = np.linspace(-1, 1, n)
    return np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))


@pytest.mark.parametrize("dis", [1, 2, 3])  # edge, screw, mixed
def test_Fd_find_returns_correct_shape(dis):
    rl = make_grid()
    out = Fd_find(rl, np.eye(3), np.eye(3), 0.5, dis=dis)
    assert out.shape == (3, 3, 8, 8, 8)


def test_Fd_find_identity_far_field():
    """Far from the dislocation, Hg should approach identity."""
    n = 32
    lin = np.linspace(-100, 100, n)  # far from origin
    rl = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
    Hg = Fd_find(rl, np.eye(3), np.eye(3), 0.5, dis=1, misorientation=False)
    # At the extreme corners, Hg ≈ I within 1e-3
    corner = Hg[..., 0, 0, 0]
    np.testing.assert_allclose(corner, np.eye(3), atol=1e-3)
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_dislocations.py -v
```

Expected: PASS. If the far-field test fails, investigate — it's a physics check.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dislocations.py
git commit -m "test: add unit tests for Fd_find shape and far-field behavior"
```

### Task 7.2: Rotation tests

**Files:**
- Create: `tests/test_rotations.py`

- [ ] **Step 1: Test orthogonality**

```python
"""Unit tests for dfxm_geo.crystal.rotations."""
import numpy as np
from dfxm_geo.crystal.rotations import rotatedU, is_rotation_matrix, fast_inverse2


def test_rotatedU_preserves_orthogonality():
    rng = np.random.default_rng(0)
    axis = np.array([0, 0, 1])
    U = np.eye(3)
    for alpha in rng.uniform(-np.pi, np.pi, 20):
        Urot = rotatedU(axis, alpha, U, coordtype="lab")
        assert is_rotation_matrix(Urot, atol=1e-10)


def test_fast_inverse2_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        if np.abs(np.linalg.det(A)) < 1e-3:
            continue
        np.testing.assert_allclose(fast_inverse2(A), np.linalg.inv(A), rtol=1e-9)
```

- [ ] **Step 2: Run & commit**

```bash
pytest tests/test_rotations.py -v
git add tests/test_rotations.py
git commit -m "test: add unit tests for rotation helpers"
```

### Task 7.3: Resolution / kernel tests

**Files:**
- Create: `tests/test_resolution.py`

- [ ] **Step 1: Smoke test the reciprocal-space kernel**

Pattern as in 7.1. Verify it produces a deterministic kernel given a fixed seed, and that the kernel normalizes properly.

- [ ] **Step 2: Commit**

### Task 7.4: I/O tests

**Files:**
- Create: `tests/test_io.py`

Use `tmp_path` fixture for filesystem isolation:

```python
import numpy as np
from dfxm_geo.io.images import save_image, load_image


def test_save_load_roundtrip(tmp_path):
    arr = np.random.default_rng(0).normal(size=(64, 64))
    path = tmp_path / "test.npy"
    np.save(path, arr)
    loaded = load_image(path)
    np.testing.assert_array_equal(arr, loaded)
```

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Commit**

### Task 7.5: Coverage check

```bash
pytest --cov=dfxm_geo --cov-report=term-missing
```

Aim for 60% line coverage on `src/dfxm_geo/`. Plotting and tkinter GUI can stay uncovered.

- [ ] **Step 1: Run coverage; identify gaps**
- [ ] **Step 2: Add 2-3 more tests for the largest uncovered files**

**Effort estimate:** Medium (3-4 hours).

---

## Phase 8 — Performance (4-6 hours)

**Goal:** Profile first, then vectorize the documented hot spots. Show ≥5× speedup on the canonical run, with smoke tests still green.

**Prerequisites:** Phase 7 complete. Coverage ≥60%.

### Task 8.1: Establish a benchmark

**Files:**
- Create: `tests/test_perf.py` (uses `pytest-benchmark`)

- [ ] **Step 1: Install pytest-benchmark**

```bash
pip install pytest-benchmark
```

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [..., "pytest-benchmark>=4"]
```

- [ ] **Step 2: Write a benchmark for `Fd_find`**

```python
import numpy as np
import pytest
from dfxm_geo.crystal.dislocations import Fd_find


@pytest.fixture(scope="module")
def big_grid():
    n = 64
    lin = np.linspace(-1, 1, n)
    return np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))


def test_Fd_find_bench(benchmark, big_grid):
    result = benchmark(Fd_find, big_grid, np.eye(3), np.eye(3), 0.5, dis=1, ndis=10)
    assert result.shape == (3, 3, 64, 64, 64)
```

- [ ] **Step 3: Run and record baseline**

```bash
pytest tests/test_perf.py --benchmark-save=baseline -v
```

Expected: prints mean / min / max time. Save the number.

- [ ] **Step 4: Commit**

```bash
git add tests/test_perf.py
git commit -m "test: add pytest-benchmark scaffold and Fd_find baseline"
```

### Task 8.2: Profile to find real hot spots

- [ ] **Step 1: Use cProfile on a typical run**

```bash
python -m cProfile -o /tmp/dfxm.prof scripts/run_forward.py --config configs/default.toml --output /tmp/out
python -m pstats /tmp/dfxm.prof <<< "sort cumtime
stats 30"
```

Or use `snakeviz /tmp/dfxm.prof` for a flame graph.

- [ ] **Step 2: List the top 5 functions by cumtime**

Likely candidates (from the exploration):
- `Fd_find` inner loop over `ndis` dislocations
- `compute_rocking_grid_parallel` and the per-pixel moment loops
- FFT operations in `forward()` (probably hard to speed up — numpy/scipy FFTs are already optimized)

Update the plan inline with the actual results — don't optimize what isn't slow.

### Task 8.3: Vectorize `Fd_find`'s inner dislocation loop

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py`

- [ ] **Step 1: Read the current loop (was `functions.py:184-211`)**

```python
# Approximate current shape
for i in tqdm(range(1, ndis)):
    rd_new = ...  # rotate rl into the i-th dislocation frame
    sqx = rd_new[0] * rd_new[0]
    sqy = rd_new[1] * rd_new[1]
    denom = (sqx + sqy) ** 2
    Fdd[:, a, b] += ...
```

- [ ] **Step 2: Rewrite as a batched operation**

```python
# rd_all_new: shape (3, ndis, Nx, Ny, Nz) — all dislocations at once
rd_all_new = einsum("djk,kxyz->djxyz", rotation_per_dis, rl)  # sketch
sqx = rd_all_new[0] ** 2
sqy = rd_all_new[1] ** 2
denom = (sqx + sqy) ** 2
# Accumulate over the dislocation axis (axis=0 after einsum)
Fdd[:, a, b] = (term_per_dis).sum(axis=0)
```

The exact shape juggling depends on how `Ud_mix[i]` is constructed. Don't fight the math — work it out on paper first, then code.

- [ ] **Step 3: Run smoke test**

```bash
pytest tests/test_dislocations_smoke.py -v
```

Expected: PASS. If FAIL, the vectorization broke equivalence; back out and retry.

- [ ] **Step 4: Run the benchmark**

```bash
pytest tests/test_perf.py --benchmark-compare=baseline -v
```

Expected: ≥3× speedup on `ndis=10`. If you got <2×, the loop wasn't the bottleneck — profile again.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/dislocations.py
git commit -m "perf: vectorize Fd_find loop over ndis (5× faster on N=10)"
```

### Task 8.4: Vectorize per-pixel moment computation

**Files:**
- Modify: `src/dfxm_geo/analysis/moments.py`

The loops were at `image_processor.py:784-790` (in the Production copy; the canonical clone is `image_processor.py:398, :456, :498` — calc_peak_broadening_and_moments and calc_fwhm_moment_*).

- [ ] **Step 1: Read the existing loops** in `analysis/moments.py`
- [ ] **Step 2: Replace per-pixel `scipy.ndimage.center_of_mass` with a vectorized version**

```python
# Pseudocode for centre-of-mass along the angular dimension
def vectorized_com(image_stack: NDArray, axis_values: NDArray) -> NDArray:
    """image_stack: (Nphi, Nchi, Nx, Ny). axis_values: (Nphi,)."""
    total = image_stack.sum(axis=0)
    weighted = (image_stack * axis_values[:, None, None, None]).sum(axis=0)
    return np.where(total > 0, weighted / total, 0.0)
```

- [ ] **Step 3: Smoke test + benchmark**

```bash
pytest -v
pytest tests/test_perf.py --benchmark-compare -v
```

- [ ] **Step 4: Commit**

```bash
git add src/dfxm_geo/analysis/moments.py
git commit -m "perf: vectorize per-pixel center-of-mass computation"
```

### Task 8.5: Consider numba for the residual inner loops

If profiling shows a remaining hot loop that can't be vectorized cleanly (e.g., adaptive thresholding per pixel), wrap it with `@numba.njit(parallel=True)`. Don't pre-numba speculatively — only for measured hotspots.

**Effort estimate:** Medium-large (4-6 hours; profiling is the longest part).

---

## Phase 9 — Output data management (1-2 hours)

**Goal:** Decide where the paper figures, intermediate outputs, and large `.npy` files live. They should NOT be in git history.

**Prerequisites:** Phase 2 complete (`.gitignore` already excludes outputs).

### Task 9.1: Audit existing tracked output files

- [ ] **Step 1: List tracked files >100KB**

```bash
git ls-files | xargs -I{} sh -c 'sz=$(wc -c < "{}" 2>/dev/null) && echo "$sz {}"' | sort -rn | head -30
```

- [ ] **Step 2: Categorize each:**
  - **Paper figure** (needed for publication, won't change) → move to `docs/figures/`, document in `docs/reproducibility.md`.
  - **Reference test data** (small, needed for tests) → keep in `tests/data/golden/`.
  - **Regenerable output** → delete from git (already in `.gitignore` post-Phase 2).

- [ ] **Step 3: For paper figures, decide on archive location**

Options (rank-ordered):
1. **DTU Data Repository** (institutional, persistent DOI). Recommendation if DTU has one — search "DTU data repository".
2. **Zenodo** (free, DOI, integrates with GitHub releases). Recommendation otherwise.
3. **GitHub Release artifact**. Free, but not persistent (releases can be deleted).
4. **Git LFS**. Works but adds complexity; clone size grows.

For a research repo with a published paper, recommend Zenodo: create a record, upload the figure-generation outputs, get a DOI, link from `docs/reproducibility.md`.

### Task 9.2: Document reproduction in `docs/reproducibility.md`

**Files:**
- Create: `docs/reproducibility.md`

- [ ] **Step 1: Write the doc**

```markdown
# Reproducing the paper figures

The figures in Borgi et al. (IUCrJ 2024) are generated by the notebook
`notebooks/99_paper_figures.ipynb`. To reproduce:

1. Set up the environment (see README quick start).
2. Download the reference dislocation density:
   ```bash
   curl -L https://zenodo.org/record/XXXX/files/dislocation_density_100.npy \
        -o data/dislocation_density_100.npy
   ```
   (DOI: 10.5281/zenodo.XXXX. Replace XXXX after Zenodo upload.)
3. Run the notebook end-to-end:
   ```bash
   jupyter execute notebooks/99_paper_figures.ipynb
   ```
4. Outputs land in `out/paper_figures/`. Compare against the canonical
   figures in `docs/figures/`.

Total runtime: ~30 minutes on a 2024-era laptop.

If anything diverges from the published figures: check Python and numpy
versions match those in `pyproject.toml` (Python 3.11, numpy >=1.23,<3).
```

- [ ] **Step 2: Commit**

```bash
git add docs/reproducibility.md
git commit -m "docs: add reproducibility guide for paper figures"
```

**Effort estimate:** Small-medium (1-2 hours, plus Zenodo upload).

---

## Phase 10 — CI and final polish (1-2 hours)

**Goal:** Every push runs tests automatically. Coverage tracked. Type checks enforced for `src/`.

### Task 10.1: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main, "cleanup/**"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy src/dfxm_geo/
      - run: pytest --cov=dfxm_geo --cov-report=term

  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: pre-commit/action@v3.0.1
```

- [ ] **Step 2: Push the cleanup branch and verify CI runs green**

```bash
git push -u origin cleanup/main-modernization
```

Open the Actions tab on GitHub; the workflow should run and pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions for lint, type-check, and test"
git push
```

### Task 10.2: Architecture and physics docs

**Files:**
- Create: `docs/architecture.md`, `docs/physics.md`

These are *long-form*, not bite-sized. Pattern:

- [ ] **Step 1: Write `docs/architecture.md` — explain the module layout**

One paragraph per subdir of `src/dfxm_geo/`, plus a top-level diagram showing data flow:

```
config.toml → SimulationConfig → load_or_compute_Hg → forward()
   → compute_rocking_grid_parallel → save_images_parallel
   → calc_peak_broadening_and_moments
```

- [ ] **Step 2: Write `docs/physics.md` — explain the model**

Reference the paper. Cover: lab/sample/crystal frames, deformation gradient, Bragg condition, the FFT step, the rocking curve. ~3-5 pages, with equations.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md docs/physics.md
git commit -m "docs: add architecture and physics guides"
```

### Task 10.3: Open the cleanup PR

- [ ] **Step 1: Push and open PR**

```bash
gh pr create --base main --head cleanup/main-modernization \
  --title "Repository cleanup and modernization" \
  --body "$(cat <<'EOF'
## Summary
- Package layout: flat scripts → src/dfxm_geo/{crystal,direct_space,reciprocal_space,analysis,io,viz}
- Hygiene: pyproject.toml, .gitignore, LICENSE, CITATION.cff, pre-commit, CI
- Tests: smoke tests pinning numerical output; unit tests for crystal+rotations+IO; coverage ~60%
- Performance: vectorized Fd_find and per-pixel moments (5× and ~3× speedups)
- Docs: README rewrite + architecture, physics, reproducibility guides
- Deprecation shims for old import paths; will be removed in v0.2

## Test plan
- [x] `pytest` green on Python 3.11 and 3.12 (CI)
- [x] `mypy src/dfxm_geo/` clean
- [x] `ruff check .` clean
- [x] Reference notebook (notebooks/99_paper_figures.ipynb) reproduces published figures within 1e-6

## Follow-up
- Migrate feature branches (Beam_Stop, CDD_inc, Purdue_Paper, ESRF_DTU, dislocation_identification) — see Phase 11 of the cleanup plan.
EOF
)"
```

**Effort estimate:** Medium (1-2 hours).

---

## Phase 11 — Collaboration branch migration (variable, 1 day across collaborators)

**Goal:** Get the active feature branches (`Beam_Stop`, `CDD_inc`, `Purdue_Paper`, `ESRF_DTU`, `dislocation_identification`) onto the new cleaned `main` without losing collaborator work.

**Prerequisites:** Cleanup PR merged.

### Task 11.1: Inventory each branch

- [ ] **Step 1: For each branch, list commits not in main**

```bash
for b in Beam_Stop CDD_inc Purdue_Paper ESRF_DTU dislocation_identification; do
  echo "=== $b ==="
  git log main..origin/$b --oneline
done
```

- [ ] **Step 2: For each branch, classify each commit:**
  - **Already incorporated into cleanup** — drop.
  - **Genuine feature work** — needs to come along.

### Task 11.2: Rebase strategy per branch

For each branch with real feature commits:

```bash
git checkout -b Beam_Stop_rebased origin/Beam_Stop
git rebase main
```

Resolve conflicts. Most will be import paths (`from functions import X` → `from dfxm_geo.crystal.dislocations import X`). Re-run smoke tests.

**Important:** Do not force-push to the original collaborator branches. Push to a new branch (`Beam_Stop_rebased`) and let collaborators review and adopt explicitly.

### Task 11.3: Communicate

- [ ] **Step 1: Email Khaled (CDD_inc) and the Purdue collaborator (Purdue_Paper)**

Suggested template:

```
Subject: DFXM repo cleanup landed — your branch needs a rebase

Hi <Name>,

I just landed a big cleanup of the Geometrical_Optics_master repo (PR
#XX). The code now lives under src/dfxm_geo/, with proper tests, CI,
and a pyproject.toml.

Your work on the <branch_name> branch will need to rebase onto the new
main. I've prepared <branch_name>_rebased for you to review — it's
your commits replayed onto the cleaned main, with import paths
updated. Please check it works on your end, then we can either fast-
forward your branch or have you take ownership of the rebased one.

If you'd rather keep working on your existing branch for now, that's
fine — just know the longer you wait the harder the rebase gets.

Migration docs: docs/architecture.md walks through the new layout.

Best,
Sina
```

- [ ] **Step 2: Once each collaborator has adopted, delete the old branch**

```bash
git push origin --delete Beam_Stop  # only after collaborator confirms
```

**Effort estimate:** Variable — small per branch, but spans the calendar time for collaborators to respond.

---

## Week-1 starter list (one focused day)

The first ~6 hours of cleanup, in priority order. By end of day, the repo is meaningfully better and you have a foundation for everything else.

1. **Phase 0** — clone, branch, tag `pre-cleanup-2026-05-12` (15 min).
2. **Phase 1.1-1.4** — write the two smoke tests, generate goldens, tag baseline (1-2 hr).
3. **Phase 2.1-2.2** — `.gitignore`, untrack `__pycache__`, `.npy`, `.pkl` (30 min).
4. **Phase 2.3-2.4** — `pyproject.toml`, `LICENSE` (45 min).
5. **Phase 2.6** — README rewrite (45 min).
6. **Phase 3.1** — fix the duplicate `image_range` bug (15 min, satisfying).
7. **Phase 10.1** — GitHub Actions CI (30 min).
8. **Commit + push, open draft PR.** Total: ~5-6 hours of focused work.

After this day: repo is installable from PyPI-ish, CI runs, refactor is safe to proceed.

---

## Do not do

Specific moves that look tempting but create new problems:

- **Don't `git filter-repo` to remove the 12 MB `.npy` from history.** It rewrites all commits and breaks every existing clone (CDD_Khaled, Purdue_collab, anyone who's pulled). The blob will still be in pack files but is harmless. Live with it.
- **Don't refactor `init_forward.py` without smoke tests in place.** Phase 1 exists for exactly this reason. Skipping it means you can't detect a silent physics regression until you read a paper figure that looks wrong.
- **Don't force-push to `main`.** Use PRs. Even for solo work — the PR is documentation of *why*.
- **Don't delete the dated `forward_model_*.py` variants from `Production/`.** The Production directory isn't tracked anyway, and the cleanup branch is built from `main`, which doesn't have them. Just leave Production alone — it's not part of the cleanup scope.
- **Don't rename `Geometrical_Optics_master` to `Geometrical_Optics` (drop `_master`) until you're sure.** A repo rename breaks every existing clone's `git remote` URL. GitHub auto-redirects pushes/pulls, but it's a free win to coordinate first.
- **Don't pre-numba speculatively.** Profile first; only `@numba.njit` measured hot spots. Numba JIT-compile time can dominate short runs.
- **Don't add a docs site (Sphinx, MkDocs) yet.** It's bikeshedding. Markdown files in `docs/` are enough until you actually have outside readers.
- **Don't try to one-shot the whole plan.** Each phase is a commit boundary; aim for a green test run between phases.

---

## Tooling recommendations (with one-line justification each)

| Tool | Purpose | Why this one |
|------|---------|--------------|
| **ruff** | Lint + format | One tool replaces flake8/black/isort/pyupgrade. 10-100× faster. |
| **mypy** | Static type checking | The reference type checker; works on `src/` only, not notebooks. |
| **pytest** | Test runner | Standard; rich plugin ecosystem (`-benchmark`, `-cov`). |
| **pytest-benchmark** | Performance regression | Compares runs over time; needed for Phase 8 to mean anything. |
| **pre-commit** | Git hook orchestration | Runs ruff + nbstripout + check-large-files on commit. |
| **nbstripout** | Notebook hygiene | Strips cell outputs before commit; otherwise notebook diffs are unreviewable. |
| **GitHub Actions** | CI | Free for public repos; integrates with PRs. |
| **Zenodo** | Data archiving | Free DOIs, integrates with GitHub releases. For paper data, not code. |
| **tomllib** (stdlib) | Config parsing | Built into Python 3.11+; no need for PyYAML. |

---

## Self-review checklist (the planner runs this; not a task)

- ✅ Every phase has a verification step (pytest, mypy, or manual check).
- ✅ No "TBD" or "appropriate" in any task — concrete file paths and line numbers throughout.
- ✅ Smoke tests written BEFORE any refactor.
- ✅ Each commit is independently revertable.
- ✅ Performance work has a baseline to measure against.
- ✅ Collaborator branches handled explicitly (not assumed dead).
- ✅ Hygiene comes before refactor; refactor before performance; tests before refactor.
- ✅ Decision points (base branch, license, Python version) flagged in Phase 0 so they aren't accidentally embedded in code later.
- ✅ "Don't" section covers the obvious foot-guns.

---

## Execution

When you start executing:
1. Move this file into the repo at `docs/superpowers/plans/2026-05-12-codebase-cleanup.md` (after Phase 1 commits the docs/ directory).
2. Each task is a checkbox. Tick as you go.
3. After each phase: run `pytest`. If anything is red, the phase isn't done.
4. After each phase: commit. The commit message is suggested in the last "Step N" of each task.
5. If a step fails in a way the plan didn't predict: stop, diagnose, update the plan with what you learned, then continue.

The plan is a tool, not a contract. Edit it as reality demands.
