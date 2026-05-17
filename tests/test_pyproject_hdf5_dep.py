"""Pin h5py as a direct runtime dependency (added in v1.1.0)."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_h5py_is_runtime_dependency() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    assert any(d.startswith("h5py") for d in deps), f"h5py not in runtime dependencies; got {deps}"
