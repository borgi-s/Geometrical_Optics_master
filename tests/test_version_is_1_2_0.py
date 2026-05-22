"""Pin the project version to 1.2.0 for the identification-HDF5 release."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_1_2_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "1.2.0"
