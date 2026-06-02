"""Pin the project version to 2.3.1 (patch on the oblique-angle release).

v2.3.1 fixes `dfxm-bootstrap --config configs/default.toml`: the bootstrap
CLI no longer mis-parses a forward-layout [crystal] (dislocation layout) as an
oblique mount, so the documented cluster kernel-regen command works again
(regression from the v2.3.0 oblique arc). Docs-only cleanup of the deleted
compute_chi_shift references rides along.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_3_1() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.3.1"
