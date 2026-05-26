# Self-sufficient pip install via bundled configs + `dfxm-init`

**Date:** 2026-05-26
**Status:** Approved (design)

## Problem

A user who installs `dfxm-geo` from PyPI/wheel (without cloning the repo) gets
the console commands (`dfxm-forward`, `dfxm-bootstrap`, `dfxm-identify`, …) but
**no config files**. The example configs live at repo-root `configs/`, which is
outside the `src/dfxm_geo` package and is not declared as package data, so the
wheel ships none of them.

Symptom observed during install testing, run from an empty directory:

```
dfxm-bootstrap --config configs/default.toml
error: config file not found: configs\default.toml
```

The `--config` argument is required and resolved relative to the current
directory; nothing in the package provides a starting template. Nothing in the
code *loads* configs as resources — they are purely user-facing input
templates, referenced only in help text and docstrings.

## Goal

Make `pip install dfxm-geo` self-sufficient: a pip-only user can obtain a
working set of config templates without cloning the repo.

## Non-goals (YAGNI)

- No auto-fallback config loading (omitting `--config` will **not** silently
  load a bundled default).
- No network fetch of templates.
- No per-preset selection syntax (`dfxm-init <name>`); the command copies the
  full tree.

## Design

### 1. Bundle the config templates inside the package

Move the example configs from repo-root `configs/` into the package, preserving
the tree:

```
src/dfxm_geo/data/configs/
  default.toml
  identification_single.toml
  identification_multi.toml
  identification_zscan.toml
  variants/
    dis_0p25.toml  dis_0p5.toml  dis_1.toml  dis_2.toml
    forward_strain_scan.toml  forward_z_scan.toml  sample_remount_S2.toml
```

`dfxm_geo.data` becomes a **real subpackage** (add `__init__.py`) so the data
ships reliably and is addressable through `importlib.resources`.

The repo-root `configs/` directory is **deleted** — the bundled copy is the
single source of truth. Repo developers obtain working configs the same way pip
users do: by running `dfxm-init`.

### 2. Accessor in `src/dfxm_geo/data/__init__.py`

The one place that knows where templates live. Used by both the CLI and the
tests so they cannot drift:

- `configs_root() -> Traversable` — the bundled `configs/` directory
  (`importlib.resources.files("dfxm_geo.data") / "configs"`).
- `iter_config_files() -> Iterator[tuple[str, Traversable]]` — yields
  `(relative_posix_path, traversable)` for every bundled `.toml`, recursing into
  `variants/`.

### 3. New `dfxm-init` command

Module: `src/dfxm_geo/init_cmd.py`, exposing `cli_main(argv=None) -> int`.

```
dfxm-init                 # write the full tree into ./configs/
dfxm-init --dest DIR      # target directory (default: configs)
dfxm-init --force         # overwrite files that already exist
```

Behavior:

- Recreates the bundled tree (including `variants/`) under `--dest`.
- For each target file that already exists: **skip with a notice** unless
  `--force` is given.
- Prints each written/skipped path and a final summary line
  (`wrote N file(s), skipped M`).
- Returns exit code 0 on success.

Because the command writes into `./configs/`, every existing help string and
doc example that says `configs/default.toml` remains correct after the user runs
`dfxm-init` — no mass docstring rewrite is required.

### 4. `pyproject.toml` changes

- Add `"dfxm_geo.data"` to `[tool.setuptools] packages`.
- Add `[tool.setuptools.package-data]` with
  `"dfxm_geo.data" = ["configs/**/*.toml"]`.
- Add `dfxm-init = "dfxm_geo.init_cmd:cli_main"` to `[project.scripts]`.

### 5. Tests

- Repoint the existing config-reading tests away from `repo_root/configs/...`
  to the bundled accessor, so they validate what actually ships:
  - `tests/test_kernel_cli.py` — `test_default_toml_has_reciprocal_block`,
    `test_default_toml_has_hkl_key`, and the `keV` assertion (currently
    `open("configs/default.toml")`).
  - `tests/test_pipeline.py` — `test_round_trip_default`,
    `test_all_shipped_variants_parse`, and the
    `Path("configs/identification_single.toml")` case.
  - `tests/test_pipeline_identification.py` — `test_example_single_config_loads`,
    `test_example_multi_config_loads`, `test_example_zscan_config_loads`.
- New `tests/test_init_cmd.py`:
  - `dfxm-init --dest tmp_path` recreates the full tree.
  - Written file contents byte-match the bundled originals.
  - Existing files are not overwritten without `--force`.
  - `--force` overwrites.

### 6. Docs (targeted)

- `README.md` and `docs/getting-started-windows-laptop.md`: add a "run
  `dfxm-init` first" step for pip-only users.
- `src/dfxm_geo/direct_space/forward_model.py:319` deprecation message: add a
  one-line nudge to run `dfxm-init` if `configs/` is absent.
- Historical specs/plans under `docs/superpowers/` are point-in-time records and
  are left untouched.

## Acceptance

- `pip install` of the built wheel, run from an empty directory:
  `dfxm-init` creates `./configs/` with the full tree; `dfxm-bootstrap --config
  configs/default.toml` then succeeds.
- The bundled `.toml` files are present in the built wheel
  (`dfxm_geo/data/configs/...`).
- Full test suite passes against the bundled configs.
