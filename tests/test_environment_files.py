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
