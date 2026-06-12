"""Stage 4.2 packaging: [cif] optional extra + gemmi in dev + mypy override."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _load() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def test_cif_extra_pins_gemmi() -> None:
    extras = _load()["project"]["optional-dependencies"]
    assert "cif" in extras
    assert any(d.startswith("gemmi") for d in extras["cif"])


def test_dev_extra_includes_gemmi() -> None:
    extras = _load()["project"]["optional-dependencies"]
    assert any(d.startswith("gemmi") for d in extras["dev"])


def test_mypy_ignores_gemmi_imports() -> None:
    overrides = _load()["tool"]["mypy"]["overrides"]
    modules = [m for o in overrides for m in o.get("module", [])]
    assert "gemmi" in modules
