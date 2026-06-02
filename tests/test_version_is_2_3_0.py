"""Pin the project version to 2.3.0 for the oblique-angle geometry release.

v2.3.0 = oblique-angle DFXM geometry (the [geometry] block, eta azimuth,
run-reflection theta) wired through BOTH forward and identification; the
identification rl-units fix (metres -> micrometres); ReciprocalConfig.lattice_a;
the MC-LUT read bin-step reconciliation (#5); the HDF5 provenance string-field
fix; and a batch of correctness/cleanup backlog items (#4-#15).
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_3_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.3.0"
