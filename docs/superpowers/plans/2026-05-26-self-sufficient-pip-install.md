# Self-sufficient pip install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle the example config templates inside the `dfxm_geo` package and add a `dfxm-init` command so a pip-only user can scaffold a working `./configs/` tree without cloning the repo.

**Architecture:** Relocate repo-root `configs/` into `src/dfxm_geo/data/configs/` as a real subpackage (single source of truth). A tiny accessor in `dfxm_geo/data/__init__.py` exposes the bundled templates via `importlib.resources`; both the new `dfxm-init` CLI and the existing config-validating tests consume it. `pyproject.toml` ships the `.toml` files as package data and registers the new console script.

**Tech Stack:** Python 3.11+, setuptools, `importlib.resources`, `argparse`, `tomllib`, pytest.

---

## File Structure

- **Move:** `configs/` (repo root) → `src/dfxm_geo/data/configs/` — bundled templates (11 `.toml` files: `default.toml`, 3× `identification_*.toml`, 7× `variants/*.toml`).
- **Create:** `src/dfxm_geo/data/__init__.py` — subpackage marker + accessor (`configs_root()`, `iter_config_files()`).
- **Create:** `src/dfxm_geo/init_cmd.py` — `dfxm-init` entry point (`cli_main`).
- **Create:** `tests/test_init_cmd.py` — accessor + `dfxm-init` behavior tests.
- **Modify:** `pyproject.toml` — add `dfxm_geo.data` package, package-data globs, `dfxm-init` script.
- **Modify:** `tests/test_kernel_cli.py`, `tests/test_pipeline.py`, `tests/test_pipeline_identification.py` — repoint config reads to the bundled accessor.
- **Modify:** `src/dfxm_geo/direct_space/forward_model.py:319` — nudge to run `dfxm-init`.
- **Modify:** `README.md`, `docs/getting-started-windows-laptop.md` — add `dfxm-init` step.

> **Branch:** Work continues on `feature/self-sufficient-pip-install` (already checked out, spec committed). The repo has unrelated WIP in the working tree — **stage only the files named in each task's commit step.** Never `git add -A`.

---

### Task 1: Relocate config templates into the package

**Files:**
- Move: `configs/` → `src/dfxm_geo/data/configs/`
- Create: `src/dfxm_geo/data/__init__.py`

- [ ] **Step 1: Create the data subpackage dir and move the configs with git**

Run (PowerShell, from repo root `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`):

```powershell
New-Item -ItemType Directory -Force src\dfxm_geo\data | Out-Null
git mv configs src\dfxm_geo\data\configs
```

- [ ] **Step 2: Create the subpackage marker**

Create `src/dfxm_geo/data/__init__.py` with a placeholder docstring (the accessor is added in Task 2):

```python
"""Bundled data shipped inside the dfxm_geo wheel (config templates, etc.)."""
```

- [ ] **Step 3: Verify the move**

Run:

```powershell
git status --short; Get-ChildItem -Recurse src\dfxm_geo\data\configs | Where-Object { -not $_.PSIsContainer } | Measure-Object | Select-Object -ExpandProperty Count
```

Expected: status shows the renames under `src/dfxm_geo/data/configs/` (R entries) and a new `src/dfxm_geo/data/__init__.py`; the file count is `11`. Repo-root `configs/` no longer exists.

- [ ] **Step 4: Commit**

```powershell
git add src/dfxm_geo/data/__init__.py
git commit -m "refactor: move config templates into dfxm_geo.data package"
```

(The `git mv` already staged the renames; the explicit `git add` stages the new `__init__.py`.)

---

### Task 2: Bundled-config accessor

**Files:**
- Modify: `src/dfxm_geo/data/__init__.py`
- Test: `tests/test_init_cmd.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_init_cmd.py`:

```python
"""Tests for the bundled-config accessor and the dfxm-init command."""

from __future__ import annotations

from pathlib import Path


class TestBundledConfigAccessor:
    def test_configs_root_contains_default(self) -> None:
        from dfxm_geo.data import configs_root

        root = configs_root()
        assert (root / "default.toml").is_file()

    def test_iter_config_files_lists_full_tree(self) -> None:
        from dfxm_geo.data import iter_config_files

        rels = {rel for rel, _ in iter_config_files()}
        assert "default.toml" in rels
        assert "identification_single.toml" in rels
        assert "variants/dis_1.toml" in rels
        # 11 shipped templates: default + 3 identification_* + 7 variants
        assert len(rels) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_init_cmd.py::TestBundledConfigAccessor -v`
Expected: FAIL with `ImportError: cannot import name 'configs_root' from 'dfxm_geo.data'`.

- [ ] **Step 3: Implement the accessor**

Replace the contents of `src/dfxm_geo/data/__init__.py` with:

```python
"""Bundled data shipped inside the dfxm_geo wheel (config templates, etc.).

`configs_root()` and `iter_config_files()` are the single source of truth for
where the bundled TOML templates live. Both the `dfxm-init` CLI and the test
suite consume them so they cannot drift from what actually ships.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Iterator
from pathlib import Path


def configs_root() -> Path:
    """Filesystem path to the bundled `configs/` template directory."""
    return Path(str(importlib.resources.files("dfxm_geo.data").joinpath("configs")))


def iter_config_files() -> Iterator[tuple[str, Path]]:
    """Yield `(relative_posix_path, absolute_path)` for every bundled `.toml`.

    Recurses into subdirectories (e.g. `variants/`). The relative path uses
    forward slashes regardless of platform.
    """
    root = configs_root()
    for path in sorted(root.rglob("*.toml")):
        rel = path.relative_to(root).as_posix()
        yield rel, path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_init_cmd.py::TestBundledConfigAccessor -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/data/__init__.py tests/test_init_cmd.py
git commit -m "feat: add bundled-config accessor in dfxm_geo.data"
```

---

### Task 3: Ship the templates in the wheel and register `dfxm-init`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `dfxm_geo.data` to the packages list**

In `pyproject.toml`, the `[tool.setuptools]` block currently reads:

```toml
[tool.setuptools]
packages = [
    "dfxm_geo",
    "dfxm_geo.analysis",
    "dfxm_geo.crystal",
    "dfxm_geo.direct_space",
    "dfxm_geo.io",
    "dfxm_geo.reciprocal_space",
    "dfxm_geo.viz",
]
```

Add `"dfxm_geo.data",` to the list (keep alphabetical-ish order, place after `"dfxm_geo.crystal",`):

```toml
[tool.setuptools]
packages = [
    "dfxm_geo",
    "dfxm_geo.analysis",
    "dfxm_geo.crystal",
    "dfxm_geo.data",
    "dfxm_geo.direct_space",
    "dfxm_geo.io",
    "dfxm_geo.reciprocal_space",
    "dfxm_geo.viz",
]
```

- [ ] **Step 2: Declare the templates as package data**

Immediately after the `packages = [...]` list closes and before the existing `[tool.setuptools.package-dir]` block, add a new block (explicit globs — avoid `**` for setuptools-version portability):

```toml
[tool.setuptools.package-data]
"dfxm_geo.data" = ["configs/*.toml", "configs/variants/*.toml"]
```

- [ ] **Step 3: Register the `dfxm-init` console script**

In the `[project.scripts]` block, add the new entry after `dfxm-bootstrap`:

```toml
dfxm-init = "dfxm_geo.init_cmd:cli_main"
```

So the block becomes:

```toml
[project.scripts]
dfxm-forward = "dfxm_geo.pipeline:cli_main"
dfxm-identify = "dfxm_geo.pipeline:cli_main_identify"
dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"
dfxm-init = "dfxm_geo.init_cmd:cli_main"
dfxm-migrate-output = "dfxm_geo.io.migrate:cli_main_npy_to_h5"
dfxm-migrate-h5 = "dfxm_geo.io.migrate:cli_main_h5_to_h5"
```

- [ ] **Step 4: Verify the TOML still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml
git commit -m "build: ship config templates as package data and add dfxm-init script"
```

---

### Task 4: Implement the `dfxm-init` command

**Files:**
- Create: `src/dfxm_geo/init_cmd.py`
- Test: `tests/test_init_cmd.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_init_cmd.py`:

```python
class TestDfxmInit:
    def test_writes_full_tree(self, tmp_path: Path) -> None:
        from dfxm_geo.data import iter_config_files
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        rc = cli_main(["--dest", str(dest)])
        assert rc == 0
        for rel, src in iter_config_files():
            written = dest / rel
            assert written.is_file(), f"missing {rel}"
            assert written.read_bytes() == src.read_bytes(), f"content mismatch {rel}"

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        (dest).mkdir()
        sentinel = dest / "default.toml"
        sentinel.write_text("DO NOT OVERWRITE", encoding="utf-8")

        rc = cli_main(["--dest", str(dest)])
        assert rc == 0
        assert sentinel.read_text(encoding="utf-8") == "DO NOT OVERWRITE"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        from dfxm_geo.data import configs_root
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        dest.mkdir()
        sentinel = dest / "default.toml"
        sentinel.write_text("DO NOT OVERWRITE", encoding="utf-8")

        rc = cli_main(["--dest", str(dest), "--force"])
        assert rc == 0
        expected = (configs_root() / "default.toml").read_bytes()
        assert sentinel.read_bytes() == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_init_cmd.py::TestDfxmInit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfxm_geo.init_cmd'`.

- [ ] **Step 3: Implement the command**

Create `src/dfxm_geo/init_cmd.py`:

```python
"""Entry point for `dfxm-init`: scaffold the bundled config templates.

Copies the templates shipped inside `dfxm_geo.data` into a destination
directory (default `./configs/`) so pip-only users have an editable starting
point without cloning the repo.
"""

from __future__ import annotations


def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    from dfxm_geo.data import iter_config_files

    parser = argparse.ArgumentParser(
        prog="dfxm-init",
        description=(
            "Write the bundled DFXM config templates (default.toml, "
            "identification_*.toml, variants/*) into a directory you can edit. "
            "Existing files are left untouched unless --force is given."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("configs"),
        help="Destination directory for the templates (default: ./configs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite template files that already exist at the destination.",
    )
    args = parser.parse_args(argv)

    written = 0
    skipped = 0
    for rel, src in iter_config_files():
        target = args.dest / rel
        if target.exists() and not args.force:
            print(f"skip (exists): {target}")
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        print(f"wrote: {target}")
        written += 1

    summary = f"dfxm-init: wrote {written} file(s), skipped {skipped}."
    if skipped and not args.force:
        summary += " Re-run with --force to overwrite skipped files."
    print(summary, file=sys.stderr)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_init_cmd.py -v`
Expected: PASS (all 5 tests in the file: 2 accessor + 3 init).

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/init_cmd.py tests/test_init_cmd.py
git commit -m "feat: add dfxm-init command to scaffold bundled config templates"
```

---

### Task 5: Repoint existing config tests to the bundled accessor

These tests currently read repo-root `configs/` (now deleted), so they must point at the bundled copy via the Task 2 accessor.

**Files:**
- Modify: `tests/test_kernel_cli.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_pipeline_identification.py`

- [ ] **Step 1: Run the affected tests to confirm they now fail**

Run:

```powershell
pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock tests/test_pipeline.py::TestSimulationConfigFromToml::test_round_trip_default tests/test_pipeline.py::TestSimulationConfigFromToml::test_all_shipped_variants_parse tests/test_pipeline_identification.py::test_example_single_config_loads -v
```

Expected: FAIL — `FileNotFoundError` / `No such file or directory` for the old `configs/...` paths.

- [ ] **Step 2: Fix `tests/test_kernel_cli.py`**

There are four reads to repoint. Replace each as follows.

`test_default_toml_has_reciprocal_block` — replace:

```python
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
```

with:

```python
        from dfxm_geo.data import configs_root

        cfg_path = configs_root() / "default.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
```

`test_toml_values_match_generate_kernel_defaults` — replace:

```python
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            recip = tomllib.load(f)["reciprocal"]
```

with:

```python
        from dfxm_geo.data import configs_root

        cfg_path = configs_root() / "default.toml"
        with cfg_path.open("rb") as f:
            recip = tomllib.load(f)["reciprocal"]
```

`test_default_toml_has_hkl_key` — replace:

```python
        with open("configs/default.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["hkl"] == [-1, 1, -1]
```

with:

```python
        from dfxm_geo.data import configs_root

        with (configs_root() / "default.toml").open("rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["hkl"] == [-1, 1, -1]
```

`test_default_toml_has_keV_key` — replace:

```python
        with open("configs/default.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["keV"] == 17.0
```

with:

```python
        from dfxm_geo.data import configs_root

        with (configs_root() / "default.toml").open("rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["keV"] == 17.0
```

- [ ] **Step 3: Fix `tests/test_pipeline.py`**

`test_round_trip_default` — replace:

```python
        repo_root = Path(__file__).resolve().parents[1]
        cfg = SimulationConfig.from_toml(repo_root / "configs" / "default.toml")
```

with:

```python
        from dfxm_geo.data import configs_root

        cfg = SimulationConfig.from_toml(configs_root() / "default.toml")
```

`test_all_shipped_variants_parse` — replace:

```python
        repo_root = Path(__file__).resolve().parents[1]
        configs = [
            p
            for p in list((repo_root / "configs").glob("*.toml"))
            + list((repo_root / "configs" / "variants").glob("*.toml"))
            if not p.name.startswith("identification_")
        ]
```

with:

```python
        from dfxm_geo.data import iter_config_files

        configs = [
            path
            for rel, path in iter_config_files()
            if not Path(rel).name.startswith("identification_")
        ]
```

`test_identification_config_parses_reciprocal_block` — replace:

```python
        config = load_identification_config(Path("configs/identification_single.toml"))
```

with:

```python
        from dfxm_geo.data import configs_root

        config = load_identification_config(configs_root() / "identification_single.toml")
```

- [ ] **Step 4: Fix `tests/test_pipeline_identification.py`**

`test_example_single_config_loads` — replace:

```python
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_single.toml")
```

with:

```python
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_single.toml")
```

`test_example_multi_config_loads` — replace:

```python
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_multi.toml")
```

with:

```python
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_multi.toml")
```

`test_example_zscan_config_loads` — replace:

```python
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_zscan.toml")
```

with:

```python
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_zscan.toml")
```

- [ ] **Step 5: Run the full suite for the touched test files**

Run:

```powershell
pytest tests/test_kernel_cli.py tests/test_pipeline.py tests/test_pipeline_identification.py -q
```

Expected: PASS (no failures). If `from pathlib import Path` is now unused in any edited file, leave it — `Path` is still referenced elsewhere in all three files.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_kernel_cli.py tests/test_pipeline.py tests/test_pipeline_identification.py
git commit -m "test: read bundled configs via dfxm_geo.data accessor"
```

---

### Task 6: Update docs and the deprecation nudge

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py:316-321`
- Modify: `README.md`
- Modify: `docs/getting-started-windows-laptop.md`

- [ ] **Step 1: Update the legacy-pickle message**

In `src/dfxm_geo/direct_space/forward_model.py`, replace:

```python
        raise RuntimeError(
            f"Detected legacy pickle at {pkl_path!r}; pickle support was "
            "removed in v1.0.3. Run `dfxm-bootstrap --config configs/default.toml` "
            "to regenerate the kernel as .npz."
        )
```

with:

```python
        raise RuntimeError(
            f"Detected legacy pickle at {pkl_path!r}; pickle support was "
            "removed in v1.0.3. If you don't have a configs/ directory yet, run "
            "`dfxm-init` first, then `dfxm-bootstrap --config configs/default.toml` "
            "to regenerate the kernel as .npz."
        )
```

- [ ] **Step 2: Add a `dfxm-init` step to the README quickstart**

In `README.md`, find the first place that instructs the reader to run `dfxm-bootstrap --config configs/default.toml` (search for `dfxm-bootstrap`). Immediately before that command, insert:

```markdown
If you installed from PyPI/wheel (no repo clone), first create local copies of
the config templates:

```bash
dfxm-init   # writes ./configs/default.toml and the rest of the template tree
```
```

- [ ] **Step 3: Add the same step to the Windows getting-started doc**

In `docs/getting-started-windows-laptop.md`, find the first `dfxm-bootstrap --config configs/default.toml` (or first reference to `configs/default.toml`). Immediately before it, insert:

```markdown
Pip-only install (no cloned repo)? Generate the config templates first:

```powershell
dfxm-init   # creates .\configs\ with default.toml and the variant templates
```
```

- [ ] **Step 4: Verify the docstring change imports/loads cleanly**

Run: `python -c "import dfxm_geo.direct_space.forward_model; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/direct_space/forward_model.py README.md docs/getting-started-windows-laptop.md
git commit -m "docs: point users at dfxm-init for bundled config templates"
```

---

### Task 7: End-to-end acceptance — build the wheel and verify it ships configs

**Files:** none (verification only).

- [ ] **Step 1: Build a wheel**

Run (from repo root):

```powershell
pip wheel . --no-deps -w dist
```

Expected: produces `dist/dfxm_geo-2.1.0-py3-none-any.whl` (filename may vary by version).

- [ ] **Step 2: Confirm the templates are inside the wheel**

Run:

```powershell
python -c "import zipfile,glob; w=sorted(glob.glob('dist/dfxm_geo-*.whl'))[-1]; names=zipfile.ZipFile(w).namelist(); cfgs=[n for n in names if '/data/configs/' in n and n.endswith('.toml')]; print(len(cfgs)); print('\n'.join(sorted(cfgs)))"
```

Expected: prints `11` followed by the 11 template paths under `dfxm_geo/data/configs/` (including `variants/`).

- [ ] **Step 3: Install into a throwaway venv and exercise `dfxm-init` + `dfxm-bootstrap`**

Run (from repo root — installs the built wheel into a throwaway venv, then runs the installed `dfxm-init` console script in an empty temp dir):

```powershell
python -m venv .tmp-verify-venv
.\.tmp-verify-venv\Scripts\python.exe -m pip install --quiet (Get-ChildItem dist\dfxm_geo-*.whl | Select-Object -Last 1).FullName
$work = Join-Path $env:TEMP "dfxm-init-check"
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
Push-Location $work
& "$(Resolve-Path $PSScriptRoot\..\..\.tmp-verify-venv\Scripts\dfxm-init.exe -ErrorAction SilentlyContinue)"
Test-Path .\configs\default.toml
Pop-Location
```

If `$PSScriptRoot` is empty (running interactively rather than from a script), use the absolute path to the venv script directly, e.g. `& "C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\.tmp-verify-venv\Scripts\dfxm-init.exe"`.

Expected: `dfxm-init` prints the `wrote: ...\configs\default.toml` lines and a summary; `Test-Path` prints `True`.

- [ ] **Step 4: Clean up verification artifacts**

Run:

```powershell
Remove-Item -Recurse -Force .tmp-verify-venv, dist -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $env:TEMP "dfxm-init-check") -ErrorAction SilentlyContinue
```

Expected: no error. These artifacts are untracked, so `git status` is unchanged by their removal.

- [ ] **Step 5: Run the full test suite one final time**

Run: `pytest -q`
Expected: PASS (the default `addopts` skip markers exclude bench/slow tests).

> No commit in this task — it is verification only.

---

## Notes for the implementer

- **Stage narrowly.** The working tree contains unrelated WIP (`pipeline.py`, `output-format.md`, untracked stray dirs). Every commit step lists exact paths — use them, never `git add -A`.
- **pre-commit hooks** run on commit (ruff, trailing-whitespace, etc.) and may auto-fix formatting; if a commit is modified by a hook, re-stage the hook's changes and commit again.
- **`importlib.resources.files`** returns a real filesystem path for a normally-installed or editable wheel, which is why `configs_root()` can return a `Path` and the tests can pass it straight to `SimulationConfig.from_toml`. `read_bytes()` in `dfxm-init` is zip-safe regardless.
