# Cluster Integration (v1.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1.0 by making a fresh clone of the repo runnable end-to-end on DTU HPC (LSF) and ESRF (SLURM) clusters: `conda env create` → `pip install -e .` → `dfxm-bootstrap --config x.toml` → `dfxm-forward --config x.toml` with no implicit prerequisites.

**Architecture:** Add `environment.yml` (+ dev split) for conda installs; add a `dfxm-bootstrap` console script (`dfxm_geo.reciprocal_space.kernel:cli_main`) that drives `generate_kernel` from a TOML `[reciprocal]` block and writes the pickle to the canonical path stage 0 reads; upgrade pipeline `_ensure_kernel_loaded` to fall back to explicit `_load_default_kernel(...)` and surface a `FileNotFoundError` with a bootstrap instruction when the pickle is absent; ship 4 batch templates (LSF single+array, SLURM single+array); write `docs/cluster-runs.md`; add cluster + examples sections to README; add a `scripts/render_readme_examples.py` to regenerate the README images on demand; bump `pyproject.toml` to `1.0.0`; amend the one-line LSF/SLURM error in the cleanup-finalization spec.

**Tech Stack:** Python 3.11+ stdlib (`tomllib`, `argparse`, `pathlib`), `numpy`, `scipy`, `numba`, `matplotlib`, `xraylib` (runtime extra), conda/conda-forge, LSF `bsub`/`#BSUB`, SLURM `sbatch`/`#SBATCH`, pytest.

**Source spec:** `docs/superpowers/specs/2026-05-15-cluster-integration-design.md`

---

## File map

**New files**
- `environment.yml` — conda-forge runtime install (production-complete)
- `environment-dev.yml` — conda-forge runtime + dev tooling
- `configs/default.toml` (modify, see below) gains a `[reciprocal]` block
- `lsf/forward_single.bsub` — DTU LSF: one forward simulation
- `lsf/identify_array.bsub` — DTU LSF: ML-training-data array job
- `slurm/forward_single.sbatch` — ESRF SLURM: one forward simulation
- `slurm/identify_array.sbatch` — ESRF SLURM: ML-training-data array job
- `docs/cluster-runs.md` — two-step workflow + DTU/ESRF walkthroughs
- `docs/img/example_dislocs_frame.png` — generated, committed
- `docs/img/example_mosaicity.png` — generated, committed
- `docs/img/example_coordinate_frames.png` — committed (already exists in repo? if not, source from physics.md figure)
- `scripts/render_readme_examples.py` — regenerates the above (CI does NOT run it)
- `tests/test_kernel_cli.py` — unit tests for `dfxm-bootstrap`
- `tests/test_cluster_templates.py` — structural checks on the four templates

**Modified files**
- `src/dfxm_geo/reciprocal_space/kernel.py` — add `cli_main(argv=None) -> int`
- `src/dfxm_geo/reciprocal_space/resolution.py` — add optional `output_path: Path | None` to `reciprocal_res_func`, write there when provided
- `src/dfxm_geo/pipeline.py` — upgrade `_ensure_kernel_loaded` (file-existence fallback + `FileNotFoundError` with bootstrap instruction)
- `configs/default.toml` — add `[reciprocal]` block with the CDD_inc canonical defaults
- `pyproject.toml` — bump `version = "1.0.0"`; add `dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"`
- `README.md` — add "Running on a cluster" and "Examples" sections (incremental, not a rewrite)
- `tests/test_pipeline.py` — extend the existing `TestPreflight` class with two new cases (pickle-on-disk reload, missing-pickle FileNotFoundError)
- `tests/test_reciprocal_resolution.py` — one new test for the `output_path` kwarg
- `docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md` — one-line correction (LSF, not SLURM, on DTU)

**Untouched on purpose**
- `src/dfxm_geo/direct_space/forward_model.py` — import-time `_load_default_kernel()` auto-load stays as-is (spec §4)

---

## Task 1: `environment.yml` (conda-forge runtime install)

**Files:**
- Create: `environment.yml`

- [ ] **Step 1: Write the failing test**

Create `tests/test_environment_files.py`:

```python
"""Structural checks on environment.yml / environment-dev.yml.

These don't try to `conda env create` (slow, sandboxed); they just validate
the YAML shape and pin the high-leverage invariants from
docs/superpowers/specs/2026-05-15-cluster-integration-design.md §1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# PyYAML isn't a project dep; skip if unavailable in the test env.
yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    with (REPO_ROOT / name).open() as f:
        return yaml.safe_load(f)


class TestEnvironmentYml:
    def test_exists(self) -> None:
        assert (REPO_ROOT / "environment.yml").is_file()

    def test_name_and_channels(self) -> None:
        env = _load("environment.yml")
        assert env["name"] == "dfxm-geo"
        # conda-forge only — defaults channel is intentionally excluded (Q3b).
        assert env["channels"] == ["conda-forge"]

    def test_runtime_deps_present(self) -> None:
        env = _load("environment.yml")
        deps = env["dependencies"]
        # Plain-name deps appear as strings; pip block appears as a dict.
        names = {d.split(">=")[0].split("=")[0] for d in deps if isinstance(d, str)}
        # Core runtime deps from pyproject.toml.
        for required in [
            "python",
            "pip",
            "numpy",
            "scipy",
            "numba",
            "matplotlib",
            "seaborn",
            "fabio",
            "joblib",
            "tqdm",
        ]:
            assert required in names, f"{required} missing from environment.yml"
        # Runtime extras must be production-complete (Q3b).
        for extra in ["xraylib", "plotly", "psutil"]:
            assert extra in names, f"runtime extra {extra} missing from environment.yml"

    def test_pip_self_install(self) -> None:
        env = _load("environment.yml")
        pip_block = [d for d in env["dependencies"] if isinstance(d, dict) and "pip" in d]
        assert len(pip_block) == 1, "expected exactly one pip: block"
        assert "-e ." in pip_block[0]["pip"]

    def test_no_dev_tools_in_runtime_env(self) -> None:
        """pytest/ruff/mypy/pre-commit live in environment-dev.yml, not here."""
        env = _load("environment.yml")
        names = {d.split(">=")[0].split("=")[0] for d in env["dependencies"] if isinstance(d, str)}
        for dev_only in ["pytest", "ruff", "mypy", "pre-commit", "jupyterlab"]:
            assert dev_only not in names, f"{dev_only} should be in environment-dev.yml only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_environment_files.py::TestEnvironmentYml -v`
Expected: FAIL — `environment.yml` does not exist; the `test_exists` test fails with `AssertionError`.

- [ ] **Step 3: Create `environment.yml`**

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
  # Runtime extras — production-complete install (cluster jobs should not
  # fail because an extra wasn't installed). See spec §1, Q3b.
  - xraylib>=4.1        # beamstop-wire physics (T cross-section)
  - plotly>=5           # identification visualisations
  - psutil>=5           # memory-aware reciprocal-space MC
  - pip:
    - -e .
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_environment_files.py::TestEnvironmentYml -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add environment.yml tests/test_environment_files.py
git commit -m "feat: add environment.yml for conda-forge cluster install

Mirrors pyproject.toml runtime deps with >= ranges; ships runtime
extras (xraylib, plotly, psutil) by default so cluster jobs can't fail
on missing optional packages.
"
```

---

## Task 2: `environment-dev.yml` (dev tooling split)

**Files:**
- Create: `environment-dev.yml`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_environment_files.py`:

```python
class TestEnvironmentDevYml:
    def test_exists(self) -> None:
        assert (REPO_ROOT / "environment-dev.yml").is_file()

    def test_extends_runtime(self) -> None:
        """environment-dev.yml is a superset of environment.yml: runtime deps + dev tools."""
        env = _load("environment-dev.yml")
        assert env["name"] == "dfxm-geo-dev"
        assert env["channels"] == ["conda-forge"]
        names = {d.split(">=")[0].split("=")[0] for d in env["dependencies"] if isinstance(d, str)}
        # Same runtime deps as environment.yml.
        for required in ["python", "numpy", "scipy", "numba", "xraylib"]:
            assert required in names
        # Plus dev tooling.
        for dev in ["pytest", "pytest-cov", "pytest-benchmark", "ruff", "mypy", "pre-commit"]:
            assert dev in names, f"{dev} missing from environment-dev.yml"

    def test_pip_self_install_dev(self) -> None:
        env = _load("environment-dev.yml")
        pip_block = [d for d in env["dependencies"] if isinstance(d, dict) and "pip" in d]
        assert pip_block, "environment-dev.yml needs a pip: block too"
        assert any("-e ." in p for p in pip_block[0]["pip"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_environment_files.py::TestEnvironmentDevYml -v`
Expected: FAIL — `environment-dev.yml` does not exist.

- [ ] **Step 3: Create `environment-dev.yml`**

```yaml
name: dfxm-geo-dev
channels:
  - conda-forge
dependencies:
  - python>=3.11
  - pip
  # Runtime (mirrors environment.yml).
  - numpy>=1.23,<3
  - scipy>=1.10,<2
  - numba>=0.56
  - matplotlib>=3.6
  - seaborn>=0.12
  - fabio>=2023.4
  - joblib>=1.3
  - tqdm>=4.66.3
  - xraylib>=4.1
  - plotly>=5
  - psutil>=5
  # Dev tooling.
  - pytest>=8
  - pytest-cov>=4
  - pytest-benchmark>=4
  - ruff>=0.6
  - mypy>=1.10
  - pre-commit>=3.7
  - ipykernel
  - jupyterlab
  - pip:
    - -e .
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_environment_files.py -v`
Expected: PASS, 8 tests total.

- [ ] **Step 5: Commit**

```bash
git add environment-dev.yml tests/test_environment_files.py
git commit -m "feat: add environment-dev.yml for development conda env

Mirrors environment.yml runtime + adds pytest, ruff, mypy, pre-commit,
jupyterlab. Lets development setups stay on conda without polluting
the runtime env that cluster jobs activate.
"
```

---

## Task 3: Plumb `output_path` through `reciprocal_res_func`

This unlocks Task 4 — `dfxm-bootstrap` needs to write the pickle to the canonical path stage 0 reads (not just `pkl_files/Resq_i_<timestamp>.pkl` in CWD).

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py:151-176` (signature) and `:317-325` (pickle write)
- Modify: `tests/test_reciprocal_resolution.py` (add 1 test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reciprocal_resolution.py` (new test class — match the file's existing style):

```python
class TestExplicitOutputPath:
    """The `output_path` kwarg lets callers pin the pickle to a specific file
    instead of writing under `pkl_files/Resq_i_<date>.pkl` in CWD. Required
    by `dfxm-bootstrap`, which must write to the path stage 0 will read.
    """

    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        """When `output_path` is provided, pickle goes there (not to CWD/pkl_files)."""
        from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

        out = tmp_path / "kernel.pkl"
        reciprocal_res_func(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            qi1_range=5e-4,
            qi2_range=7.5e-3,
            qi3_range=7.5e-3,
            plot_figs=False,
            save_resqi=True,
            zeta_v_fwhm=5.3e-4,
            zeta_h_fwhm=0,
            NA_rms=7.31e-4 / 2.35,
            eps_rms=1.41e-4 / 2.35,
            theta=0.1566,
            phys_aper=2e-3 / 0.274,
            date="test",
            rng=np.random.default_rng(42),
            output_path=out,
        )
        assert out.is_file(), "expected pickle written to explicit output_path"
        # And the legacy default path was NOT created.
        assert not (tmp_path / "pkl_files").exists(), (
            "explicit output_path must not also create the legacy pkl_files/ dir"
        )

    def test_default_path_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no `output_path`, falls back to `pkl_files/Resq_i_<date>.pkl` in CWD."""
        from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

        monkeypatch.chdir(tmp_path)
        reciprocal_res_func(
            Nrays=1000, npoints1=20, npoints2=20, npoints3=20,
            qi1_range=5e-4, qi2_range=7.5e-3, qi3_range=7.5e-3,
            plot_figs=False, save_resqi=True,
            zeta_v_fwhm=5.3e-4, zeta_h_fwhm=0,
            NA_rms=7.31e-4 / 2.35, eps_rms=1.41e-4 / 2.35,
            theta=0.1566, phys_aper=2e-3 / 0.274,
            date="legacy", rng=np.random.default_rng(42),
        )
        assert (tmp_path / "pkl_files" / "Resq_i_legacy.pkl").is_file()
```

(Top of file — confirm imports already include `numpy as np` / `pytest` / `Path`; add if missing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_reciprocal_resolution.py::TestExplicitOutputPath -v`
Expected: FAIL — `TypeError: reciprocal_res_func() got an unexpected keyword argument 'output_path'`.

- [ ] **Step 3: Add `output_path` to `reciprocal_res_func`**

In `src/dfxm_geo/reciprocal_space/resolution.py`, extend the signature (around line 151, just after `knife_edge: bool = False,`):

```python
    output_path: Path | None = None,
```

(Add `from pathlib import Path` to the imports at the top of the file if not already present.)

Replace the save block at lines 317-325:

```python
    if save_resqi == 1:
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as output:
                pickle.dump(normResq_i, output)
            print(f"Resq_i saved to {output_path}")
        else:
            # Legacy default: write to pkl_files/Resq_i_<date>.pkl in CWD.
            # Ensure pkl_files/ exists in the CWD only when we're about to write
            # to it. Previously this was done as a module-level side effect on
            # import, which silently created a stray directory anywhere the
            # module was imported.
            check_folder("", "pkl_files")
            with open(f"pkl_files/Resq_i_{date}.pkl", "wb") as output:
                pickle.dump(normResq_i, output)
            print(f"Resq_i saved as Resq_i_{date}.pkl")
```

Also update the docstring of `reciprocal_res_func` (top of the function body) with one new line:

```
    output_path: Optional explicit path for the pickle. When provided,
        overrides the default `pkl_files/Resq_i_<date>.pkl` in CWD.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_reciprocal_resolution.py -v`
Expected: PASS, including the two new `TestExplicitOutputPath` tests and the existing baseline tests (the bit-equal golden test at `tests/data/golden/reciprocal_baseline.npz` must still pass — `output_path=None` is the default).

- [ ] **Step 5: mypy + ruff sanity check**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m mypy src/dfxm_geo/`
Expected: 0 errors.

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m ruff check src/dfxm_geo/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal): support explicit output_path in reciprocal_res_func

Unlocks dfxm-bootstrap, which must write the kernel pickle to the
canonical path that stage 0 reads, not to pkl_files/ in the caller's
CWD. Legacy default behaviour preserved when output_path is None.
"
```

---

## Task 4: Plumb `output_path` through `generate_kernel`

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py:31-104`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_cli.py`:

```python
"""Tests for `dfxm_geo.reciprocal_space.kernel.generate_kernel` and `cli_main`.

Uses Nrays=1000 + a 20**3 grid to keep each test under ~1 s.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


class TestGenerateKernelOutputPath:
    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "subdir" / "Resq_i_test.pkl"
        result_path = generate_kernel(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            output_path=out,
        )
        assert Path(result_path) == out
        assert out.is_file()
        with out.open("rb") as f:
            arr = pickle.load(f)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20, 20, 20)

    def test_writes_vars_sidecar_next_to_pickle(self, tmp_path: Path) -> None:
        """The `<stem>_vars.txt` sidecar lands next to the pickle, not in CWD."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "Resq_i_explicit.pkl"
        generate_kernel(Nrays=1000, npoints1=20, npoints2=20, npoints3=20, output_path=out)
        sidecar = tmp_path / "Resq_i_explicit_vars.txt"
        assert sidecar.is_file()
        # Sidecar contains the kwargs (sanity check on serialisation).
        text = sidecar.read_text()
        assert "Nrays" in text
        assert "qi1_range" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py::TestGenerateKernelOutputPath -v`
Expected: FAIL — `generate_kernel()` does not yet accept `output_path`.

- [ ] **Step 3: Add `output_path` to `generate_kernel`**

In `src/dfxm_geo/reciprocal_space/kernel.py`, extend the signature:

```python
def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    npoints1: int = 400,
    npoints2: int = 200,
    npoints3: int = 200,
    qi1_range: float = 5e-4,
    qi2_range: float = 0.75e-2,
    qi3_range: float = 0.75e-2,
    zeta_v_fwhm: float = 5.3e-04,
    zeta_h_fwhm: float = 0,
    NA_rms: float = 7.31e-4 / 2.35,
    eps_rms: float = 1.41e-4 / 2.35,
    theta: float = _default_theta_al_111(17),
    D: float = float(2 * np.sqrt(50e-6 * 1.6e-3)),
    d1: float = 0.274,
    beamstop: bool = True,
    bs_height: float = 25e-3,
    aperture: bool = True,
    knife_edge: bool = False,
    dphi_range: float = 0.0,
    output_path: Path | None = None,
) -> Path:
```

Update the imports at the top:

```python
from datetime import datetime
from pathlib import Path

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func
```

Replace the body of `generate_kernel` (after `date = ... ; phys_aper = D / d1`):

```python
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    # Resolve the destination. When `output_path` is provided, it is the
    # canonical path; the sidecar lives next to it as `<stem>_vars.txt`.
    # When `output_path` is None, fall back to the legacy default
    # `pkl_files/Resq_i_<date>.pkl` (relative to CWD) — preserves the
    # `python -m dfxm_geo.reciprocal_space.kernel` workflow.
    if output_path is not None:
        output_path = Path(output_path)
        vars_path = output_path.with_name(output_path.stem + "_vars.txt")
    else:
        output_path = None  # let reciprocal_res_func fall back to legacy
        vars_path = Path("pkl_files") / f"Resq_i_{date}_vars.txt"

    reciprocal_res_func(
        Nrays,
        npoints1,
        npoints2,
        npoints3,
        qi1_range,
        qi2_range,
        qi3_range,
        plot_figs=False,
        save_resqi=True,
        zeta_v_fwhm=zeta_v_fwhm,
        zeta_h_fwhm=zeta_h_fwhm,
        NA_rms=NA_rms,
        eps_rms=eps_rms,
        theta=theta,
        phys_aper=phys_aper,
        date=date,
        beamstop=beamstop,
        bs_height=bs_height,
        aperture=aperture,
        knife_edge=knife_edge,
        dphi_range=dphi_range,
        output_path=output_path,
    )

    vars_used = {
        "Nrays": Nrays,
        "npoints1": npoints1,
        "npoints2": npoints2,
        "npoints3": npoints3,
        "qi1_range": qi1_range,
        "qi2_range": qi2_range,
        "qi3_range": qi3_range,
        "zeta_v_fwhm": zeta_v_fwhm,
        "zeta_h_fwhm": zeta_h_fwhm,
        "NA_rms": NA_rms,
        "eps_rms": eps_rms,
        "theta": theta,
        "D": D,
        "d1": d1,
        "phys_aper": phys_aper,
        "beamstop": beamstop,
        "bs_height": bs_height,
        "aperture": aperture,
        "knife_edge": knife_edge,
        "dphi_range": dphi_range,
    }

    vars_path.parent.mkdir(parents=True, exist_ok=True)
    vars_path.write_text(str(vars_used))

    return output_path if output_path is not None else Path("pkl_files") / f"Resq_i_{date}.pkl"
```

Update the docstring's `Returns:` block:

```
    Returns:
        The path the pickle was written to.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py::TestGenerateKernelOutputPath -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: mypy check**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m mypy src/dfxm_geo/`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli.py
git commit -m "feat(kernel): generate_kernel accepts output_path

Returns the path it wrote. Sidecar _vars.txt lands next to the pickle.
Legacy CWD-relative default preserved for the
\`python -m dfxm_geo.reciprocal_space.kernel\` workflow.
"
```

---

## Task 5: Add `[reciprocal]` block to `configs/default.toml`

A `dfxm-bootstrap --config configs/default.toml` invocation has to know what parameters to pass to `generate_kernel`. Stage 0's `FileNotFoundError` message will quote this exact path.

**Files:**
- Modify: `configs/default.toml`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_cli.py`:

```python
class TestDefaultConfigReciprocalBlock:
    def test_default_toml_has_reciprocal_block(self) -> None:
        """configs/default.toml must include a `[reciprocal]` block that
        dfxm-bootstrap can drive without any extra args.
        """
        import tomllib

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        assert "reciprocal" in data, "configs/default.toml missing [reciprocal] block"
        recip = data["reciprocal"]
        # CDD_inc canonical recipe (spec §1 + kernel.py defaults).
        assert recip["Nrays"] == int(1e8)
        assert recip["beamstop"] is True
        assert recip["aperture"] is True
        assert recip["knife_edge"] is False
        assert recip["bs_height"] == 25e-3
        # qi ranges (units consistent with generate_kernel kwargs).
        assert recip["qi1_range"] == 5e-4
        assert recip["qi2_range"] == 0.75e-2
        assert recip["qi3_range"] == 0.75e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock -v`
Expected: FAIL — no `[reciprocal]` key in TOML.

- [ ] **Step 3: Add `[reciprocal]` block to `configs/default.toml`**

Append to the file:

```toml

[reciprocal]
# Used by `dfxm-bootstrap --config configs/default.toml` to (re)generate the
# reciprocal-space resolution kernel pickle. Defaults reproduce the CDD_inc
# canonical recipe (Al 111 reflection at 17 keV, square-aperture beamstop
# 25 mm at the BFP). Takes ~50 s wall-clock on a laptop at Nrays=1e8.
Nrays      = 100_000_000   # 1e8
npoints1   = 400
npoints2   = 200
npoints3   = 200
qi1_range  = 5e-4
qi2_range  = 7.5e-3
qi3_range  = 7.5e-3
# Beam / objective parameters.
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.110638297872340e-4  # = 7.31e-4 / 2.35
eps_rms     = 6.0e-5                # = 1.41e-4 / 2.35
# theta is auto-derived from Al 111 + 17 keV when absent (see kernel.cli_main).
D  = 0.000565685424949238   # = 2 * sqrt(50e-6 * 1.6e-3)
d1 = 0.274
# Beamstop config.
beamstop   = true
bs_height  = 25e-3
aperture   = true
knife_edge = false
dphi_range = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/default.toml tests/test_kernel_cli.py
git commit -m "feat(config): add [reciprocal] block to configs/default.toml

Lets \`dfxm-bootstrap --config configs/default.toml\` drive
kernel generation with no extra args. Defaults match the CDD_inc
canonical recipe baked into generate_kernel().
"
```

---

## Task 6: `dfxm-bootstrap` CLI (`kernel.cli_main`)

The TOML-aware entry point that fresh-clone users invoke.

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (add `cli_main`)
- Modify: `pyproject.toml` (`[project.scripts]` entry)
- Test: `tests/test_kernel_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_cli.py`:

```python
class TestCliMain:
    """Unit tests for `dfxm_geo.reciprocal_space.kernel.cli_main`.

    We mock `generate_kernel` itself (the underlying Monte Carlo is exercised
    elsewhere); the goal here is to pin the CLI surface — flags, defaults,
    overwrite-guard, and TOML parsing.
    """

    def _make_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "tiny.toml"
        cfg.write_text(
            "[reciprocal]\n"
            "Nrays = 1000\n"
            "npoints1 = 20\n"
            "npoints2 = 20\n"
            "npoints3 = 20\n"
            "qi1_range = 5e-4\n"
            "qi2_range = 7.5e-3\n"
            "qi3_range = 7.5e-3\n"
            "beamstop = true\n"
            "bs_height = 25e-3\n"
            "aperture = true\n"
            "knife_edge = false\n"
        )
        return cfg

    def test_requires_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dfxm_geo.reciprocal_space.kernel import cli_main

        with pytest.raises(SystemExit) as excinfo:
            cli_main([])
        assert excinfo.value.code == 2  # argparse usage error

    def test_invokes_generate_kernel_with_toml_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        captured: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> Path:
            captured.update(kwargs)
            out = kwargs.get("output_path")
            assert isinstance(out, Path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        out = tmp_path / "canonical.pkl"
        rc = kmod.cli_main(["--config", str(cfg), "--output", str(out)])
        assert rc == 0
        assert out.is_file()
        # TOML fields are forwarded as kwargs.
        assert captured["Nrays"] == 1000
        assert captured["npoints1"] == 20
        assert captured["beamstop"] is True
        assert captured["bs_height"] == 25e-3
        assert captured["output_path"] == out

    def test_default_output_matches_forward_model_canonical_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no --output, write to `<fm.pkl_fpath>/<fm.pkl_fn>`."""
        from dfxm_geo.reciprocal_space import kernel as kmod
        import dfxm_geo.direct_space.forward_model as fm

        cfg = self._make_config(tmp_path)
        seen: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> Path:
            seen.update(kwargs)
            out = kwargs["output_path"]
            assert isinstance(out, Path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 0
        expected = Path(fm.pkl_fpath) / fm.pkl_fn
        assert seen["output_path"] == expected

    def test_refuses_to_overwrite_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        existing = tmp_path / "existing.pkl"
        existing.write_bytes(b"prior contents")
        called = False

        def fake_generate(**kwargs: object) -> Path:
            nonlocal called
            called = True
            return Path()

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(existing)])
        assert rc != 0
        assert not called, "generate_kernel must not run when output exists and --force absent"
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "--force" in out

    def test_force_flag_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        existing = tmp_path / "existing.pkl"
        existing.write_bytes(b"prior contents")

        def fake_generate(**kwargs: object) -> Path:
            out = kwargs["output_path"]
            assert isinstance(out, Path)
            out.write_bytes(b"new contents")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(existing), "--force"])
        assert rc == 0
        assert existing.read_bytes() == b"new contents"

    def test_missing_reciprocal_block_errors_clearly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.reciprocal_space.kernel import cli_main

        bad = tmp_path / "no_recip.toml"
        bad.write_text("[scan]\nphi_range = 0.1\n")
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--config", str(bad)])
        assert excinfo.value.code != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "[reciprocal]" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py::TestCliMain -v`
Expected: FAIL — `cli_main` does not exist (`AttributeError`).

- [ ] **Step 3: Implement `cli_main`**

Append to `src/dfxm_geo/reciprocal_space/kernel.py` (just before the `if __name__ == "__main__":` guard, replacing it as well):

```python
def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-bootstrap`.

    Reads a TOML config (e.g. `configs/default.toml`), parses the
    `[reciprocal]` block, and writes the resulting reciprocal-space kernel
    pickle to the canonical path that `dfxm-forward`'s stage-0 preflight will
    read (`<fm.pkl_fpath>/<fm.pkl_fn>`), or to `--output <path>` if given.
    """
    import argparse
    import sys
    import tomllib

    import dfxm_geo.direct_space.forward_model as fm

    parser = argparse.ArgumentParser(
        prog="dfxm-bootstrap",
        description=(
            "Generate the reciprocal-space resolution kernel pickle for "
            "dfxm-forward / dfxm-identify. Takes ~50 s wall-clock at the "
            "default Nrays=1e8."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config containing a [reciprocal] block.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination pickle path. Defaults to <pkl_fpath>/<pkl_fn> "
            "(the path dfxm-forward reads at import time)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pickle at the destination.",
    )
    args = parser.parse_args(argv)

    with args.config.open("rb") as f:
        data = tomllib.load(f)
    if "reciprocal" not in data:
        print(
            f"error: {args.config} has no [reciprocal] block; "
            "see configs/default.toml for the expected schema.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_path = args.output if args.output is not None else Path(fm.pkl_fpath) / fm.pkl_fn

    if output_path.exists() and not args.force:
        print(
            f"refusing to overwrite existing pickle at {output_path}; "
            "pass --force to regenerate.",
            file=sys.stderr,
        )
        return 1

    kwargs = dict(data["reciprocal"])
    written = generate_kernel(output_path=output_path, **kwargs)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(cli_main())
```

- [ ] **Step 4: Wire up the console script in `pyproject.toml`**

In `pyproject.toml` `[project.scripts]`, add:

```toml
[project.scripts]
dfxm-forward = "dfxm_geo.pipeline:cli_main"
dfxm-identify = "dfxm_geo.pipeline:cli_main_identify"
dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"
```

- [ ] **Step 5: Reinstall to register the entry point**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pip install -e . --no-deps -q`
Expected: success; `dfxm-bootstrap --help` should now resolve.

- [ ] **Step 6: Run the test suite**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_kernel_cli.py -v`
Expected: PASS, including the 6 `TestCliMain` tests + the earlier `TestGenerateKernelOutputPath` + `TestDefaultConfigReciprocalBlock`.

- [ ] **Step 7: Manual smoke test of the console script**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\dfxm-bootstrap.exe" --help`
Expected: argparse help text describing `--config`, `--output`, `--force`.

- [ ] **Step 8: mypy + ruff**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m mypy src/dfxm_geo/`
Expected: 0 errors.

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m ruff check src/dfxm_geo/`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py pyproject.toml tests/test_kernel_cli.py
git commit -m "feat(kernel): add dfxm-bootstrap CLI (kernel.cli_main)

Reads [reciprocal] from a TOML config and writes the resolution
kernel pickle to <fm.pkl_fpath>/<fm.pkl_fn> by default (the canonical
path stage 0 reads), or to --output <path>. Refuses to overwrite
without --force. See docs/cluster-runs.md for the two-step workflow.
"
```

---

## Task 7: Upgrade `_ensure_kernel_loaded` (pipeline stage 0)

Today `_ensure_kernel_loaded` only checks `fm.Resq_i is not None` and raises a generic `RuntimeError`. Spec §3 requires a `FileNotFoundError` with the bootstrap instruction when the pickle is missing on disk, and a same-process recovery via `_load_default_kernel(...)` when the pickle exists but auto-load somehow didn't run.

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:249-263` (the `_ensure_kernel_loaded` function)
- Modify: `tests/test_pipeline.py` (extend `TestPreflight`)

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, extend `TestPreflight` (currently around line 174) with two new cases:

```python
class TestPreflight:
    def test_raises_when_kernel_not_loaded_and_pickle_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pickle absent → FileNotFoundError with the dfxm-bootstrap hint."""
        monkeypatch.setattr(fm, "Resq_i", None)
        # Point the canonical path at an empty tmp dir; nothing to find.
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + "/")
        monkeypatch.setattr(fm, "pkl_fn", "missing_kernel.pkl")
        with pytest.raises(FileNotFoundError) as excinfo:
            _ensure_kernel_loaded()
        msg = str(excinfo.value)
        assert "dfxm-bootstrap" in msg
        assert "docs/cluster-runs.md" in msg
        assert "missing_kernel.pkl" in msg

    def test_recovers_when_pickle_on_disk_but_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pickle present but Resq_i not loaded → call _load_default_kernel."""
        monkeypatch.setattr(fm, "Resq_i", None)
        called: dict[str, str] = {}

        def fake_load(pkl_path: str | None = None, **kwargs: object) -> None:
            called["pkl_path"] = pkl_path or ""
            # Simulate the load succeeding by populating Resq_i.
            monkeypatch.setattr(fm, "Resq_i", np.ones((2, 2, 2)))

        monkeypatch.setattr(fm, "_load_default_kernel", fake_load)
        # Make the canonical path exist.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            tf.write(b"placeholder")
            pkl_path = tf.name
        try:
            monkeypatch.setattr(fm, "pkl_fpath", str(Path(pkl_path).parent) + "/")
            monkeypatch.setattr(fm, "pkl_fn", Path(pkl_path).name)
            _ensure_kernel_loaded()
            assert called["pkl_path"].endswith(Path(pkl_path).name)
            assert fm.Resq_i is not None
        finally:
            Path(pkl_path).unlink(missing_ok=True)

    def test_noops_when_already_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resq_i already set → no I/O, no error."""
        monkeypatch.setattr(fm, "Resq_i", np.ones((2, 2, 2)))
        # Set canonical path to a definitely-missing file: if the preflight
        # touched it, we'd hit the FileNotFoundError branch instead.
        monkeypatch.setattr(fm, "pkl_fpath", "/nonexistent/")
        monkeypatch.setattr(fm, "pkl_fn", "does_not_matter.pkl")
        _ensure_kernel_loaded()  # must not raise
```

Replace the entire existing `TestPreflight` class (lines ~174-192) with the three tests above. Then add a second class for the `run_simulation` integration check, since the new behaviour is `FileNotFoundError`, not `RuntimeError`:

```python
class TestRunSimulationPreflight:
    def test_run_simulation_short_circuits_without_kernel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run_simulation() bails via the preflight before the thread pool starts."""
        monkeypatch.setattr(fm, "Resq_i", None)
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + "/")
        monkeypatch.setattr(fm, "pkl_fn", "missing.pkl")
        with pytest.raises(FileNotFoundError, match="dfxm-bootstrap"):
            run_simulation(SimulationConfig(), tmp_path)
        assert not (tmp_path / "images10").exists()
        assert not (tmp_path / "images10_perf_crystal").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py::TestPreflight tests/test_pipeline.py::TestRunSimulationPreflight -v`
Expected: FAIL — current `_ensure_kernel_loaded` raises `RuntimeError`, not `FileNotFoundError`; no recovery branch exists.

- [ ] **Step 3: Rewrite `_ensure_kernel_loaded`**

Replace the function body in `src/dfxm_geo/pipeline.py` (lines 249-263):

```python
def _ensure_kernel_loaded() -> None:
    """Pre-flight: verify the reciprocal-space kernel is loaded.

    If the canonical pickle is on disk but the import-time auto-load didn't
    populate state (e.g. it ran before bootstrap in the same process), this
    function calls ``fm._load_default_kernel(...)`` to recover. If the pickle
    is missing altogether, raises ``FileNotFoundError`` with a clear
    `dfxm-bootstrap` instruction.
    """
    if fm.Resq_i is not None:
        return  # auto-load already populated state at import (common case)
    pkl_path = Path(fm.pkl_fpath) / fm.pkl_fn
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Reciprocal-space kernel pickle not found at {pkl_path}.\n"
            "Run 'dfxm-bootstrap --config <your.toml>' to generate it "
            "(takes ~50 s for default Nrays=1e8). See docs/cluster-runs.md "
            "for the full cluster workflow."
        )
    # Pickle is on disk but Resq_i is None — load it explicitly.
    fm._load_default_kernel(str(pkl_path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: PASS — all 4 preflight tests + every other pipeline test still green.

- [ ] **Step 5: mypy + ruff**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m mypy src/dfxm_geo/`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): stage-0 preflight loads pickle if present, hints bootstrap if not

_ensure_kernel_loaded now (1) no-ops when Resq_i is already populated;
(2) calls fm._load_default_kernel(...) when the canonical pickle is on
disk but state is empty (in-process recovery); (3) raises
FileNotFoundError with a dfxm-bootstrap hint when the pickle is missing.
"
```

---

## Task 8: `lsf/forward_single.bsub` (DTU LSF single-job template)

**Files:**
- Create: `lsf/forward_single.bsub`
- Test: `tests/test_cluster_templates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cluster_templates.py`:

```python
"""Structural checks on the LSF / SLURM batch templates.

These don't submit anything; they assert the templates exist, declare the
right scheduler directives, include the EDIT THESE block, and reference
the right CLI entry points and docs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


class TestLsfForwardSingle:
    rel = "lsf/forward_single.bsub"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_bsub_directives(self) -> None:
        text = _read(self.rel)
        assert text.startswith("#!/bin/bash"), "missing shebang"
        # Core LSF directives.
        for directive in ["#BSUB -J", "#BSUB -q", "#BSUB -W", "#BSUB -R", "#BSUB -o", "#BSUB -e"]:
            assert directive in text, f"missing {directive}"

    def test_default_queue_is_hpc(self) -> None:
        assert "#BSUB -q hpc" in _read(self.rel)

    def test_invokes_forward_cli(self) -> None:
        text = _read(self.rel)
        assert "dfxm-forward" in text
        assert "dfxm-bootstrap" in text, "template should remind users to bootstrap once"

    def test_edit_these_block(self) -> None:
        text = _read(self.rel)
        assert "EDIT THESE" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestLsfForwardSingle -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create `lsf/forward_single.bsub`**

```bash
#!/bin/bash
# ==============================================================================
# DTU HPC (LSF) — single dfxm-forward simulation
# ==============================================================================
# Submit with:   bsub < lsf/forward_single.bsub
# Monitor with:  bjobs -l <JOBID>
# See docs/cluster-runs.md for the full walkthrough.
# ==============================================================================
#
# >>> EDIT THESE >>>
#BSUB -J dfxm-forward                  # Job name (visible in bjobs)
#BSUB -q hpc                           # Queue (DTU HPC default)
#BSUB -W 24:00                         # Walltime HH:MM (24 h is the hpc queue cap)
#BSUB -n 8                             # CPU slots
#BSUB -R "rusage[mem=4GB]"             # Memory per slot
#BSUB -R "span[hosts=1]"               # Pin to a single host (joblib parallelism)
#BSUB -o logs/forward-%J.out           # stdout (%J = job id)
#BSUB -e logs/forward-%J.err           # stderr
# #BSUB -u your.address@dtu.dk -B -N   # Uncomment for email on start/end
# <<< EDIT THESE <<<

set -euo pipefail

# Workspace is the directory bsub was invoked from. Make sure it is the repo
# root or adjust the paths below.
WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/default.toml"
OUTPUT="${WORKDIR}/output/forward_${LSB_JOBID:-local}"

mkdir -p "$(dirname "${OUTPUT}")" logs

# Module load (DTU HPC site convention). Replace miniconda3 with your module
# tag if your site uses a different name.
module load python3/3.11
# Alternatively: `source ~/miniforge3/etc/profile.d/conda.sh` if you maintain
# your own conda install on the cluster.

# Activate the conda environment created by environment.yml.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

# Stage 0 — generate the kernel pickle if missing. dfxm-bootstrap is idempotent
# (it refuses to overwrite without --force), so this line is safe to leave in.
if [ ! -f reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl ]; then
    dfxm-bootstrap --config "${CONFIG}"
fi

# Stage 1 — run the forward simulation.
dfxm-forward --config "${CONFIG}" --output "${OUTPUT}"

echo "Done. Output in ${OUTPUT}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestLsfForwardSingle -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add lsf/forward_single.bsub tests/test_cluster_templates.py
git commit -m "feat(cluster): add DTU LSF forward-simulation template

bsub script for one dfxm-forward run on DTU HPC's hpc queue. 24 h
walltime, 4 GB/slot, 8 slots, single host. Idempotent stage-0 bootstrap
included.
"
```

---

## Task 9: `lsf/identify_array.bsub` (DTU LSF array job template)

**Files:**
- Create: `lsf/identify_array.bsub`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cluster_templates.py`:

```python
class TestLsfIdentifyArray:
    rel = "lsf/identify_array.bsub"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_array_directive(self) -> None:
        text = _read(self.rel)
        assert "#BSUB -J" in text
        # LSF array syntax: -J "name[1-N]"
        import re
        assert re.search(r"#BSUB -J [\"'][^\"']*\[\d+-\d+\]", text), (
            "LSF array template must declare -J 'name[1-N]'"
        )

    def test_uses_lsb_jobindex(self) -> None:
        """Tasks differentiate via $LSB_JOBINDEX."""
        assert "$LSB_JOBINDEX" in _read(self.rel) or "${LSB_JOBINDEX}" in _read(self.rel)

    def test_invokes_identify_cli(self) -> None:
        assert "dfxm-identify" in _read(self.rel)

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestLsfIdentifyArray -v`
Expected: FAIL.

- [ ] **Step 3: Create `lsf/identify_array.bsub`**

```bash
#!/bin/bash
# ==============================================================================
# DTU HPC (LSF) — ML-training-data array job for dfxm-identify
# ==============================================================================
# Each array task processes a slice of the configured n_samples. With 10
# samples per task and the default n_samples=1000, 100 array tasks cover
# the full sweep.
#
# Submit with:   bsub < lsf/identify_array.bsub
# Monitor with:  bjobs -A <ARRAYID>
# ==============================================================================
#
# >>> EDIT THESE >>>
#BSUB -J "dfxm-identify[1-100]"        # Array index range
#BSUB -q hpc
#BSUB -W 6:00                          # Walltime per array task
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/identify-%J-%I.out       # %J = job id, %I = array index
#BSUB -e logs/identify-%J-%I.err
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/identification_multi.toml"
OUTPUT_BASE="${WORKDIR}/output/identify_${LSB_JOBID:-local}"
TASK_OUTPUT="${OUTPUT_BASE}/task_$(printf '%04d' "${LSB_JOBINDEX}")"

mkdir -p "${OUTPUT_BASE}" logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

if [ ! -f reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl ]; then
    dfxm-bootstrap --config "${CONFIG}"
fi

# Each task computes 10 samples; rng_seed is offset by LSB_JOBINDEX so each
# task draws a deterministic, non-overlapping slice. NB: this assumes the
# identification TOML's `multi` block uses n_samples=10 (per-task). Adjust
# the LSF array range above to cover your total: tasks * 10 = total samples.
dfxm-identify --config "${CONFIG}" --output "${TASK_OUTPUT}"

echo "Task ${LSB_JOBINDEX} done. Output in ${TASK_OUTPUT}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestLsfIdentifyArray -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add lsf/identify_array.bsub tests/test_cluster_templates.py
git commit -m "feat(cluster): add DTU LSF identify-array template

100 array tasks × 10 samples = 1000 sample default sweep. Per-task
output dir + LSB_JOBINDEX-derived rng offset.
"
```

---

## Task 10: `slurm/forward_single.sbatch` (ESRF SLURM single-job template)

**Files:**
- Create: `slurm/forward_single.sbatch`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cluster_templates.py`:

```python
class TestSlurmForwardSingle:
    rel = "slurm/forward_single.sbatch"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_sbatch_directives(self) -> None:
        text = _read(self.rel)
        assert text.startswith("#!/bin/bash")
        for directive in [
            "#SBATCH --job-name",
            "#SBATCH --time",
            "#SBATCH --output",
            "#SBATCH --error",
            "#SBATCH --cpus-per-task",
            "#SBATCH --mem",
        ]:
            assert directive in text, f"missing {directive}"

    def test_mentions_sinfo_callout(self) -> None:
        """SLURM templates flag partition naming as cluster-specific via `sinfo`."""
        assert "sinfo" in _read(self.rel)

    def test_invokes_forward_cli(self) -> None:
        text = _read(self.rel)
        assert "dfxm-forward" in text
        assert "dfxm-bootstrap" in text

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestSlurmForwardSingle -v`
Expected: FAIL.

- [ ] **Step 3: Create `slurm/forward_single.sbatch`**

```bash
#!/bin/bash
# ==============================================================================
# ESRF (SLURM) — single dfxm-forward simulation
# ==============================================================================
# Submit with:   sbatch slurm/forward_single.sbatch
# Monitor with:  squeue -u $USER -j <JOBID>
# See docs/cluster-runs.md for the full walkthrough.
# ==============================================================================
#
# >>> EDIT THESE >>>
#SBATCH --job-name=dfxm-forward
# Verify available partitions on your cluster:  sinfo -o "%P %a %D"
#SBATCH --partition=nice               # ESRF default; check `sinfo` to confirm
#SBATCH --time=24:00:00                # HH:MM:SS
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/forward-%j.out   # %j = job id
#SBATCH --error=logs/forward-%j.err
# #SBATCH --mail-user=your.address@example.org
# #SBATCH --mail-type=BEGIN,END,FAIL
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/default.toml"
OUTPUT="${WORKDIR}/output/forward_${SLURM_JOB_ID:-local}"

mkdir -p "$(dirname "${OUTPUT}")" logs

# Module + env activation (ESRF site convention). Adjust to match your
# institution's module path or a local miniforge install.
module load conda 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

if [ ! -f reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl ]; then
    dfxm-bootstrap --config "${CONFIG}"
fi

dfxm-forward --config "${CONFIG}" --output "${OUTPUT}"

echo "Done. Output in ${OUTPUT}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestSlurmForwardSingle -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add slurm/forward_single.sbatch tests/test_cluster_templates.py
git commit -m "feat(cluster): add ESRF SLURM forward-simulation template

sbatch script for one dfxm-forward run on ESRF compute. 24 h walltime,
32 GB RAM, 8 cpus-per-task. Includes 'verify with sinfo' callout on
partition naming since ESRF partition names change per cluster.
"
```

---

## Task 11: `slurm/identify_array.sbatch` (ESRF SLURM array job template)

**Files:**
- Create: `slurm/identify_array.sbatch`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cluster_templates.py`:

```python
class TestSlurmIdentifyArray:
    rel = "slurm/identify_array.sbatch"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_array_directive(self) -> None:
        text = _read(self.rel)
        # SLURM array syntax: --array=1-N
        import re
        assert re.search(r"#SBATCH --array=\d+-\d+", text), (
            "SLURM array template must declare --array=N-M"
        )

    def test_uses_slurm_array_task_id(self) -> None:
        text = _read(self.rel)
        assert "$SLURM_ARRAY_TASK_ID" in text or "${SLURM_ARRAY_TASK_ID}" in text

    def test_invokes_identify_cli(self) -> None:
        assert "dfxm-identify" in _read(self.rel)

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestSlurmIdentifyArray -v`
Expected: FAIL.

- [ ] **Step 3: Create `slurm/identify_array.sbatch`**

```bash
#!/bin/bash
# ==============================================================================
# ESRF (SLURM) — ML-training-data array job for dfxm-identify
# ==============================================================================
# Submit with:   sbatch slurm/identify_array.sbatch
# Monitor with:  squeue -u $USER -j <ARRAYID>
# ==============================================================================
#
# >>> EDIT THESE >>>
#SBATCH --job-name=dfxm-identify
#SBATCH --partition=nice               # check `sinfo` to confirm
#SBATCH --array=1-100                  # 100 tasks × 10 samples = 1000-sample sweep
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/identify-%A_%a.out  # %A = array job id, %a = task id
#SBATCH --error=logs/identify-%A_%a.err
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/identification_multi.toml"
OUTPUT_BASE="${WORKDIR}/output/identify_${SLURM_ARRAY_JOB_ID:-local}"
TASK_OUTPUT="${OUTPUT_BASE}/task_$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")"

mkdir -p "${OUTPUT_BASE}" logs

module load conda 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

if [ ! -f reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl ]; then
    dfxm-bootstrap --config "${CONFIG}"
fi

dfxm-identify --config "${CONFIG}" --output "${TASK_OUTPUT}"

echo "Task ${SLURM_ARRAY_TASK_ID} done. Output in ${TASK_OUTPUT}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py -v`
Expected: PASS, all 20 template tests (5 per template × 4 templates).

- [ ] **Step 5: Commit**

```bash
git add slurm/identify_array.sbatch tests/test_cluster_templates.py
git commit -m "feat(cluster): add ESRF SLURM identify-array template

100 array tasks × 10 samples = 1000 sample default sweep. Per-task
output dir + SLURM_ARRAY_TASK_ID-derived layout.
"
```

---

## Task 12: `docs/cluster-runs.md`

**Files:**
- Create: `docs/cluster-runs.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cluster_templates.py`:

```python
class TestClusterRunsDoc:
    rel = "docs/cluster-runs.md"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_has_required_sections(self) -> None:
        text = _read(self.rel)
        for heading in [
            "Two-step workflow",
            "DTU HPC",
            "ESRF",
            "Output handling",
            "Memory + walltime sizing",
            "DTU vs ESRF",
        ]:
            assert heading in text, f"missing section: {heading}"

    def test_references_templates(self) -> None:
        text = _read(self.rel)
        for path in [
            "lsf/forward_single.bsub",
            "lsf/identify_array.bsub",
            "slurm/forward_single.sbatch",
            "slurm/identify_array.sbatch",
        ]:
            assert path in text, f"missing reference to {path}"

    def test_references_clis(self) -> None:
        text = _read(self.rel)
        assert "dfxm-bootstrap" in text
        assert "dfxm-forward" in text
        assert "dfxm-identify" in text

    def test_warns_lsf_vs_slurm(self) -> None:
        """DTU is LSF (bsub), ESRF is SLURM (sbatch). The doc must say so explicitly."""
        text = _read(self.rel)
        assert "LSF" in text
        assert "SLURM" in text
        assert "bsub" in text
        assert "sbatch" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestClusterRunsDoc -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create `docs/cluster-runs.md`**

```markdown
# Running DFXM forward simulations on a cluster

This guide walks through running `dfxm-forward` and `dfxm-identify` on the two
clusters this codebase has been validated against: **DTU HPC** (LSF
scheduler, `bsub`) and **ESRF** (SLURM scheduler, `sbatch`). Both flows boil
down to the same two-step workflow.

## Two-step workflow

The reciprocal-space resolution kernel is a ~128 MB pickle that takes ~50 s
to regenerate on a laptop at the default `Nrays=1e8`. The pickle is **not**
in the repo; you generate it once per environment with `dfxm-bootstrap`,
then run as many `dfxm-forward` / `dfxm-identify` jobs against it as you
like.

```bash
# Step 1 — one-time per environment.
dfxm-bootstrap --config configs/default.toml

# Step 2 — many times.
dfxm-forward --config configs/default.toml --output output/run01/
```

Why two steps? Kernel regeneration is expensive; silently regenerating it
inside `dfxm-forward` would risk burning an hour of cluster time on a typo
while the user thinks the "simulation" is running. `dfxm-forward` therefore
fails loud with a `FileNotFoundError` and a `dfxm-bootstrap` instruction
when the pickle is missing.

`dfxm-bootstrap` writes to the canonical path that `dfxm-forward` reads
(`reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl`). If you want a
different destination, pass `--output <path>`; if you want to regenerate
an existing pickle, pass `--force`.

## DTU HPC (LSF) walkthrough

The DTU HPC cluster uses the LSF scheduler. Templates live in
[`lsf/forward_single.bsub`](../lsf/forward_single.bsub) and
[`lsf/identify_array.bsub`](../lsf/identify_array.bsub).

```bash
# On the cluster (after `ssh login.hpc.dtu.dk`):
cd ~/Geometrical_Optics_master
git pull

# One-time conda env setup.
module load python3/3.11
conda env create -f environment.yml      # creates `dfxm-geo`
conda activate dfxm-geo
pip install -e .

# Run a single forward simulation.
bsub < lsf/forward_single.bsub
bjobs                                    # check status

# Once it's running, follow stdout:
tail -f logs/forward-<JOBID>.out
```

The default LSF template targets the `hpc` queue with a 24 h walltime cap,
4 GB/slot, 8 slots, single host. Override these in the `>>> EDIT THESE >>>`
block at the top.

For ML training data — sweeping many random crystal configurations — use
the array template:

```bash
bsub < lsf/identify_array.bsub
bjobs -A <ARRAYID>                       # array status
```

## ESRF (SLURM) walkthrough

The ESRF cluster uses the SLURM scheduler. Templates live in
[`slurm/forward_single.sbatch`](../slurm/forward_single.sbatch) and
[`slurm/identify_array.sbatch`](../slurm/identify_array.sbatch).

```bash
# On the cluster:
cd ~/Geometrical_Optics_master
git pull

# One-time conda env setup.
module load conda
conda env create -f environment.yml
conda activate dfxm-geo
pip install -e .

# Confirm the partition name (ESRF partitions vary per cluster).
sinfo -o "%P %a %D"

# Run a single forward simulation.
sbatch slurm/forward_single.sbatch
squeue -u $USER                          # check status
```

Array jobs:

```bash
sbatch slurm/identify_array.sbatch
squeue -u $USER -j <ARRAYID>
```

## Output handling

Templates write `output/<run-tag>_<jobid>/` relative to the directory the
job was submitted from (`bsub`/`sbatch` inherits CWD). On both clusters,
that's typically a shared scratch directory the templates assume is writable.

To pull results back to your laptop:

```bash
rsync -avh --partial \
    cluster-login:~/Geometrical_Optics_master/output/forward_<JOBID>/ \
    ./local-output/
```

## Memory + walltime sizing

| Workload | Wall time | Memory |
|---|---|---|
| `dfxm-bootstrap` (Nrays=1e8) | ~50 s | ~5 GB peak (chunked truncnorm) |
| `dfxm-forward` (61×61 grid, Nsub=2, ndis=151) | ~10–20 min | ~4 GB |
| `dfxm-forward` (61×61, Nsub=1 — fast iteration) | ~2–3 min | ~2 GB |
| `dfxm-identify --mode multi` (10 samples) | ~2 min | ~4 GB |

These numbers are rough — verify against your cluster before scaling up.

## DTU vs ESRF specifics

| | DTU HPC | ESRF |
|---|---|---|
| Scheduler | LSF | SLURM |
| Submit | `bsub < file` | `sbatch file` |
| Status | `bjobs` | `squeue` |
| Time format | `HH:MM` | `HH:MM:SS` |
| Partition flag | `#BSUB -q <queue>` | `#SBATCH --partition=<part>` |
| Memory flag | `#BSUB -R "rusage[mem=4GB]"` | `#SBATCH --mem=4G` |
| Array syntax | `#BSUB -J "name[1-N]"` | `#SBATCH --array=1-N` |
| Array index env var | `$LSB_JOBINDEX` | `$SLURM_ARRAY_TASK_ID` |
| Account flag (if required) | `#BSUB -P <project>` | `#SBATCH --account=<account>` |

If your cluster requires an account/project flag for billing, add it to the
`EDIT THESE` block at the top of each template.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_cluster_templates.py::TestClusterRunsDoc -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add docs/cluster-runs.md tests/test_cluster_templates.py
git commit -m "docs: add cluster-runs guide (DTU LSF + ESRF SLURM)

Two-step workflow (bootstrap once → forward many), per-cluster
walkthroughs, output rsync patterns, memory/walltime sizing table,
and a DTU-vs-ESRF cheat sheet.
"
```

---

## Task 13: README — "Running on a cluster" section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Create `tests/test_readme_sections.py`:

```python
"""Smoke checks that README.md has the cluster + examples sections required by v1.0."""

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _text() -> str:
    return README.read_text()


class TestReadmeClusterSection:
    def test_has_cluster_section(self) -> None:
        text = _text()
        assert "## Running on a cluster" in text or "# Running on a cluster" in text

    def test_links_to_cluster_runs(self) -> None:
        assert "docs/cluster-runs.md" in _text()

    def test_mentions_template_dirs(self) -> None:
        text = _text()
        assert "lsf/" in text
        assert "slurm/" in text

    def test_mentions_dfxm_bootstrap(self) -> None:
        assert "dfxm-bootstrap" in _text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_readme_sections.py::TestReadmeClusterSection -v`
Expected: FAIL — cluster section is not present yet.

- [ ] **Step 3: Add the section to `README.md`**

Insert immediately after the existing "Running a simulation" section (before "Project structure" — the exact insertion point can be confirmed by reading README.md before editing):

```markdown
## Running on a cluster

`dfxm-forward` and `dfxm-identify` are designed to run on HPC clusters. The
two-step workflow is `dfxm-bootstrap` once per environment, then
`dfxm-forward` / `dfxm-identify` many times. See
[`docs/cluster-runs.md`](docs/cluster-runs.md) for the full walkthrough.

Submit templates live in:

- [`lsf/`](lsf/) — DTU HPC (LSF scheduler, `bsub`)
- [`slurm/`](slurm/) — ESRF (SLURM scheduler, `sbatch`)

Each scheduler ships a single-job template (`forward_single.{bsub,sbatch}`)
and an array template for ML-training sweeps
(`identify_array.{bsub,sbatch}`). Open them and edit the
`>>> EDIT THESE >>>` block at the top to set your queue / partition,
walltime, and memory.

For conda-based cluster installs, use [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate dfxm-geo
pip install -e .
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_readme_sections.py::TestReadmeClusterSection -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme_sections.py
git commit -m "docs(readme): add 'Running on a cluster' section

Three-sentence summary linking to docs/cluster-runs.md plus a list of
the lsf/ and slurm/ template directories. conda-env quickstart block.
"
```

---

## Task 14: `scripts/render_readme_examples.py` (generates example images)

This produces the image files referenced by the README "Examples" section in Task 15.

**Files:**
- Create: `scripts/render_readme_examples.py`
- Modify: `docs/img/` will be created and populated when the script runs

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_readme_examples.py`:

```python
"""Smoke test for `scripts/render_readme_examples.py`.

Runs the script against a scaled-down config and asserts the expected PNGs
land in docs/img/. Not run in CI (slow + non-deterministic floats).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_readme_examples.py"


@pytest.mark.bench
def test_render_readme_examples_smoke(tmp_path: Path) -> None:
    """End-to-end run — gated behind the bench marker so CI skips it."""
    env = os.environ.copy()
    env["DFXM_RENDER_OUTPUT_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--small"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    # Outputs land in $DFXM_RENDER_OUTPUT_DIR (tmp_path in the test).
    pngs = list(tmp_path.glob("example_*.png"))
    assert len(pngs) >= 2, f"expected at least 2 example PNGs, got {pngs}"


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_has_small_flag() -> None:
    """The `--small` flag must be supported (used by the bench test + docs)."""
    text = SCRIPT.read_text()
    assert "--small" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_render_readme_examples.py::test_script_exists tests/test_render_readme_examples.py::test_script_has_small_flag -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/render_readme_examples.py`**

```python
"""Regenerate the example images linked from the top of README.md.

Outputs (default destination is ``docs/img/``; override with the
``DFXM_RENDER_OUTPUT_DIR`` environment variable):

- ``example_dislocs_frame.png`` — single (phi=0, chi=0) frame from
  the dislocs stack at a scaled-down rocking grid.
- ``example_mosaicity.png`` — phi-COM mosaicity map from the
  post-processing stage.

This script is *not* part of CI:

- it runs `dfxm-forward` end-to-end (~10–30 s wall-clock at the small
  variant, but loads the 128 MB kernel pickle), and
- the rendered floats are sensitive to hardware/BLAS, so we can't pin
  them in version control without churn.

Run manually after a substantive change to `dfxm_geo.viz` or
`dfxm_geo.pipeline.run_postprocess`:

    python scripts/render_readme_examples.py --small

The committed PNGs in ``docs/img/`` are the canonical version — only
overwrite them when you intend to publish the new look.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _build_small_config(tmp_dir: Path) -> Path:
    """Write a scaled-down TOML for fast example rendering."""
    cfg = tmp_dir / "render_small.toml"
    cfg.write_text(
        "[crystal]\n"
        "dis = 4\n"
        "ndis = 151\n"
        "sample_remount = \"S1\"\n"
        "\n"
        "[scan]\n"
        "phi_range = 0.034377467707849395\n"
        "phi_steps = 11\n"
        "chi_range = 0.11459155902616465\n"
        "chi_steps = 11\n"
        "\n"
        "[io]\n"
        "fn_prefix = \"/mosa_test_0000_\"\n"
        "ftype = \".npy\"\n"
        "dislocs_dirname = \"images10\"\n"
        "perfect_dirname = \"images10_perf_crystal\"\n"
        "include_perfect_crystal = true\n"
        "\n"
        "[postprocess]\n"
        "enabled = true\n"
        "chi_oversample = 5\n"
        "phi_oversample = 5\n"
        "chi_oversample_for_shift = 20\n"
        "figures_dirname = \"figures\"\n"
        "data_dirname = \"analysis\"\n"
    )
    return cfg


def _save_dislocs_frame_png(images_dir: Path, out_png: Path) -> None:
    """Save one frame from the dislocs stack as a PNG."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Pick the center frame of the rocking grid.
    npy_files = sorted(images_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy frames in {images_dir}")
    arr = np.load(npy_files[len(npy_files) // 2])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=144)
    im = ax.imshow(arr.T, origin="lower", cmap="viridis")
    ax.set_title("DFXM forward image (center of rocking grid)")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate README example images")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a scaled-down config (11x11 rocking, ~30 s wall clock).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    default_out_dir = repo_root / "docs" / "img"
    out_dir = Path(os.environ.get("DFXM_RENDER_OUTPUT_DIR", default_out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.small:
        print(
            "warning: running at full resolution — this will take minutes. "
            "Pass --small for the quick variant.",
            file=sys.stderr,
        )

    # Import here so that --help works on a clean checkout that lacks the kernel.
    from dfxm_geo.pipeline import SimulationConfig, run_postprocess, run_simulation

    with tempfile.TemporaryDirectory(prefix="dfxm_render_") as td:
        tmp = Path(td)
        cfg_path = _build_small_config(tmp) if args.small else repo_root / "configs" / "default.toml"
        cfg = SimulationConfig.from_toml(cfg_path)
        run_dir = tmp / "run"
        run_simulation(cfg, run_dir)
        run_postprocess(run_dir, cfg)

        # 1. A dislocs-stack frame.
        _save_dislocs_frame_png(
            run_dir / cfg.io.dislocs_dirname,
            out_dir / "example_dislocs_frame.png",
        )

        # 2. The README needs PNG, not SVG, for portable embedding. Re-render the
        # mosaicity figure as PNG from the saved data products.
        phi_list = np.load(run_dir / cfg.postprocess.data_dirname / "phi_list.npy")
        chi_list = np.load(run_dir / cfg.postprocess.data_dirname / "chi_list.npy")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, (axp, axc) = plt.subplots(1, 2, figsize=(8, 4), dpi=144)
        for ax, data, title in (
            (axp, phi_list, "φ COM (mosaicity)"),
            (axc, chi_list, "χ COM (mosaicity)"),
        ):
            im = ax.imshow(data, origin="lower", cmap="RdBu_r")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "example_mosaicity.png", bbox_inches="tight")
        plt.close(fig)

        # 3. (Optional) the coordinate-frame diagram if a static asset exists.
        coord_src = repo_root / "docs" / "img" / "_static" / "coordinate_frames.png"
        if coord_src.is_file():
            shutil.copy(coord_src, out_dir / "example_coordinate_frames.png")

    print(f"wrote example images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the existence test**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_render_readme_examples.py::test_script_exists tests/test_render_readme_examples.py::test_script_has_small_flag -v`
Expected: PASS.

- [ ] **Step 5: Manual end-to-end render (depends on the kernel pickle being present)**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe scripts/render_readme_examples.py --small`
Expected: wall-clock ≤ 60 s; `docs/img/example_dislocs_frame.png` and `docs/img/example_mosaicity.png` exist and are non-empty PNGs (≥ 5 KB).

If the kernel pickle is missing, the run will raise `FileNotFoundError` from the new stage 0 — that's the correct failure mode. Run `dfxm-bootstrap` first.

- [ ] **Step 6: Commit (including the generated PNGs)**

```bash
git add scripts/render_readme_examples.py docs/img/example_dislocs_frame.png docs/img/example_mosaicity.png tests/test_render_readme_examples.py
git commit -m "feat(scripts): add render_readme_examples.py + commit small variants

Generates the example PNGs embedded in README.md from a scaled-down
(11x11) rocking config. Manual regen; CI skips the bench-marked smoke
test.
"
```

---

## Task 15: README — "Examples" section

**Files:**
- Modify: `README.md`
- Test: `tests/test_readme_sections.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_readme_sections.py`:

```python
class TestReadmeExamplesSection:
    def test_has_examples_section(self) -> None:
        text = _text()
        assert "## Examples" in text or "# Examples" in text

    def test_references_example_images(self) -> None:
        text = _text()
        for img in [
            "docs/img/example_dislocs_frame.png",
            "docs/img/example_mosaicity.png",
        ]:
            assert img in text, f"missing image reference: {img}"

    def test_references_render_script(self) -> None:
        assert "scripts/render_readme_examples.py" in _text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_readme_sections.py::TestReadmeExamplesSection -v`
Expected: FAIL.

- [ ] **Step 3: Add the section to `README.md`**

Insert the section near the top of the README (just after "What this code does", before the existing "Status" section):

```markdown
## Examples

A representative DFXM forward image from a 151-dislocation crystal:

![DFXM forward image](docs/img/example_dislocs_frame.png)

The corresponding mosaicity map (per-pixel COM in φ and χ) from the
post-processing stage:

![Mosaicity map](docs/img/example_mosaicity.png)

To regenerate these images locally:

```bash
dfxm-bootstrap --config configs/default.toml      # one-time, ~50 s
python scripts/render_readme_examples.py --small  # ~30 s
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_readme_sections.py -v`
Expected: PASS, 7 tests total.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme_sections.py
git commit -m "docs(readme): add 'Examples' section with hero images

Embeds the two PNGs produced by scripts/render_readme_examples.py
near the top of the README, matching the FABLE-3DXRD/xrd_simulator
pattern. Tells users how to regenerate the images locally.
"
```

---

## Task 16: `pyproject.toml` version bump to 1.0.0

**Files:**
- Modify: `pyproject.toml` (line 7)

- [ ] **Step 1: Write the failing test**

Append to an existing pyproject-style test, or create `tests/test_pyproject_version.py`:

```python
"""Pin the project version to 1.0.0 for the cluster-integration release."""

from pathlib import Path
import tomllib

REPO = Path(__file__).resolve().parents[1]


def test_version_is_1_0_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "1.0.0"


def test_dfxm_bootstrap_script_registered() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert scripts["dfxm-bootstrap"] == "dfxm_geo.reciprocal_space.kernel:cli_main"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_pyproject_version.py -v`
Expected: FAIL on `test_version_is_1_0_0` (still 0.1.0); PASS on `test_dfxm_bootstrap_script_registered` (added in Task 6).

- [ ] **Step 3: Bump the version**

In `pyproject.toml`:

```toml
version = "1.0.0"
```

(Replace the existing `version = "0.1.0"` on line 7.)

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest tests/test_pyproject_version.py -v`
Expected: PASS, both tests.

- [ ] **Step 5: Reinstall and verify import metadata**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pip install -e . --no-deps -q`
Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -c "from importlib.metadata import version; print(version('dfxm-geo'))"`
Expected: `1.0.0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_pyproject_version.py
git commit -m "chore: bump pyproject version to 1.0.0

Aligns the importable metadata with the v1.0 cluster-integration
release. Git tags (not this field) drive Zenodo / citations, so
backfilling 0.9.0 is unnecessary.
"
```

---

## Task 17: Amend the cleanup-finalization spec (LSF/SLURM correction)

The finalization spec contains a one-line factual error: it claims DTU HPC
uses SLURM, when in fact it uses LSF. Fix it in the same PR.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md`

- [ ] **Step 1: Locate the offending line**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -c "import re; t=open(r'docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md',encoding='utf-8').read(); [print(i+1, ln) for i, ln in enumerate(t.splitlines()) if 'SLURM' in ln and 'DTU' in ln]"`
Expected: prints the one line conflating DTU with SLURM (the exact line number depends on the spec; the spec mentions "Linux SLURM clusters (DTU HPC + ESRF)").

- [ ] **Step 2: Edit the line**

Replace `Linux SLURM clusters (DTU HPC + ESRF)` (and any close paraphrase) with
`Linux clusters (DTU HPC uses LSF; ESRF uses SLURM)`. If there are nearby
references that compound the error (e.g. "use sbatch on both"), revise
those as well.

Add a corrections / amendments footer to the spec:

```markdown

## 2026-05-15 — Amendment

- Corrected scheduler attribution: DTU HPC uses LSF (`bsub` / `#BSUB`),
  not SLURM. ESRF uses SLURM. See
  [`docs/superpowers/specs/2026-05-15-cluster-integration-design.md`](2026-05-15-cluster-integration-design.md)
  for the per-scheduler plan.
```

- [ ] **Step 3: Verify the edit**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -c "t=open(r'docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md',encoding='utf-8').read(); assert 'DTU HPC uses LSF' in t, 'amendment not applied'; assert 'Linux SLURM clusters (DTU HPC + ESRF)' not in t, 'old wording still present'; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-15-cleanup-finalization-design.md
git commit -m "docs(spec): correct scheduler attribution in cleanup-finalization

DTU HPC uses LSF; only ESRF uses SLURM. Adds an amendment footer
pointing at the cluster-integration spec.
"
```

---

## Task 18: Full-suite verification + release notes draft

Final gate before merging to `main` and tagging.

- [ ] **Step 1: Full test suite + mypy + ruff**

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m pytest -v`
Expected: every test passes (the new template/env/README/render/preflight tests plus the entire pre-existing suite). Coverage report unchanged or higher.

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m mypy src/dfxm_geo/`
Expected: 0 errors.

Run: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe -m ruff check src/dfxm_geo/ scripts/ tests/`
Expected: clean.

- [ ] **Step 2: Manual end-to-end smoke**

```powershell
# From the repo root, with the .venv active and the kernel pickle already
# generated from Round 14 (Resq_i_20230913_1308.pkl).
$env:DFXM_DATA_DIR = "C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\output"
dfxm-forward --config configs/default.toml --output output\v1.0-smoke\
```

Expected: completes without errors; `output/v1.0-smoke/images10/` contains 3721 (= 61*61) `.npy` files; `output/v1.0-smoke/figures/mosaicity_maps.svg` exists.

- [ ] **Step 3: Draft release notes**

Create `docs/release-notes-1.0.0.md` (or paste straight into the `gh release create` body in the next step). Summary template:

```markdown
# v1.0.0 — Cluster integration

## What's new

- **One-command environment install** — `conda env create -f environment.yml`
  (and `environment-dev.yml` for development) covers every runtime dep
  including `xraylib`, `plotly`, and `psutil`. No more hand-translating
  pyproject.toml to conda specs.
- **`dfxm-bootstrap` console script** — the missing stage 0. Run it once
  per environment to generate the reciprocal-space kernel pickle from a
  TOML `[reciprocal]` block; fresh-clone users no longer need a tribal
  knowledge to find the right command.
- **Fail-loud stage 0 in the pipeline** — `dfxm-forward` and
  `dfxm-identify` raise a clear `FileNotFoundError` with the
  `dfxm-bootstrap` instruction when the kernel is missing, instead of
  the previous cryptic RuntimeError from inside `forward()`.
- **Cluster batch templates** — `lsf/forward_single.bsub`,
  `lsf/identify_array.bsub`, `slurm/forward_single.sbatch`,
  `slurm/identify_array.sbatch` for DTU HPC and ESRF.
- **`docs/cluster-runs.md`** — soup-to-nuts cluster guide with a
  DTU-vs-ESRF cheat sheet, memory/walltime sizing table, and rsync
  patterns.
- **README hero images** — two example PNGs (forward frame +
  mosaicity map) embedded near the top, regenerable via
  `scripts/render_readme_examples.py`.

## Known limitations

- The kernel pickle is still a 128 MB blob shipped as a generated artefact;
  alternatives are filed in
  [`followups_kernel_pickle_alternatives`](https://github.com/borgi-s/Geometrical_Optics_master/issues?q=label%3Akernel-pickle).
- Reflection runtime configuration (h, k, l beyond Al 111) is deferred to v1.1.

## Migration from v0.9.0

No code changes; the canonical install path on a cluster is now
`environment.yml` + `pip install -e .` + `dfxm-bootstrap --config <toml>`.
Existing pickles continue to work without regeneration.
```

- [ ] **Step 4: Commit release notes**

```bash
git add docs/release-notes-1.0.0.md
git commit -m "docs: draft v1.0.0 release notes"
```

- [ ] **Step 5: User-gated push + PR**

Pause here for user approval. Once approved, push and open a PR:

```bash
git push origin cleanup/main-modernization  # or whatever the v1.0 branch is named
gh pr create --base main \
    --title "v1.0: cluster integration" \
    --body-file docs/release-notes-1.0.0.md
```

Do **not** push without explicit user approval (CLAUDE.md rule).

- [ ] **Step 6: After merge, tag and release**

Once the PR merges to `main`:

```bash
git checkout main
git pull
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0 --notes-file docs/release-notes-1.0.0.md
```

The Zenodo deposit fires automatically if the user has linked
GitHub ↔ Zenodo per spec §"Path to v1.0 release" step 5.

---

## Spec coverage map (self-review)

| Spec section | Tasks |
|---|---|
| §1 environment.yml (conda-forge runtime) | Task 1 |
| §1 environment-dev.yml (dev split) | Task 2 |
| §2 dfxm-bootstrap CLI (kernel.cli_main, TOML-aware) | Tasks 3, 4, 5, 6 |
| §3 pipeline stage 0 upgrade | Task 7 |
| §4 forward_model auto-load (no change) | — (explicitly untouched) |
| §5 batch-job templates (LSF + SLURM, single + array) | Tasks 8, 9, 10, 11 |
| §6 docs/cluster-runs.md | Task 12 |
| §7 README cluster section | Task 13 |
| §7 README examples section | Task 15 |
| §8 scripts/render_readme_examples.py | Task 14 |
| §9 pyproject.toml version bump | Task 16 |
| Spec amendment (LSF/SLURM correction) | Task 17 |
| Final verification + release | Task 18 |

All nine spec components have at least one task; the spec amendment
(separate from the nine components) and the release-gating verification
are also covered.

---

## DRY / YAGNI notes

- The `output_path` plumbing (Tasks 3 + 4) is a single conceptual change —
  resolution.py picks up the kwarg, kernel.py threads it through. Two tasks
  because each is independently committable + testable.
- The four template tasks (8–11) repeat structure on purpose: each template
  is one self-contained artefact that a reader may inspect alone. The
  shared test helpers (`_read`, `REPO_ROOT`) live in
  `tests/test_cluster_templates.py` and are reused across all four.
- No abstraction layer over LSF vs SLURM. The spec called for four
  templates with sensible defaults + EDIT THESE blocks; abstracting over
  schedulers would add complexity without solving any actual problem.
- README updates (Tasks 13 + 15) are split because the cluster section
  goes mid-document and the examples section goes near the top; they're
  also reviewable independently.
- The version bump (Task 16) is its own commit so that `git blame` for
  the version field stays clean.
