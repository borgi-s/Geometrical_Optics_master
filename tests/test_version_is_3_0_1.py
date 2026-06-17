"""Pin the project version to 3.0.1 (stdio-server-safe git provenance).

v3.0.1 is a PATCH release (see docs/release-notes-3.0.1.md):

- io.hdf5._get_git_sha_and_dirty() now pins the git child's stdin to DEVNULL.
  Without it, collecting git-SHA provenance behind an stdio server (e.g. the
  dfxm-geo-mcp MCP transport) hangs forever: the git subprocess inherits the
  parent's stdin — the live JSON-RPC pipe — and communicate() never returns.
  Provenance values are unchanged; pure embedded/stdio-safety hardening.

No API, dependency, or CLI-script changes — conda-forge can take the autotick
bot's version+sha256 bump as-is (no feedstock hand-edit required).
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_3_0_1() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "3.0.1"
