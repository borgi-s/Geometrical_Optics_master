"""Pin the dfxm-bootstrap entry point registered in pyproject.toml."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_dfxm_bootstrap_script_registered() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert scripts["dfxm-bootstrap"] == "dfxm_geo.reciprocal_space.kernel:cli_main"
