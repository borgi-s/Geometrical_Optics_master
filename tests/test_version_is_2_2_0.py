"""Pin the project version to 2.2.0 for the forward-throughput arc release.

v2.2.0 = Find_Hg numba fusion (~14.7x) + float32 detector HDF5 +
scripts/fanout.py in-node launcher + the io.write_strain_provenance flag.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_2_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.2.0"
