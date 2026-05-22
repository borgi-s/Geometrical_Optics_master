import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_dfxm_migrate_output_entry_point_registered() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert scripts["dfxm-migrate-output"] == "dfxm_geo.io.migrate:cli_main_npy_to_h5"


def test_dfxm_migrate_h5_entry_point_registered() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert scripts["dfxm-migrate-h5"] == "dfxm_geo.io.migrate:cli_main_h5_to_h5"
