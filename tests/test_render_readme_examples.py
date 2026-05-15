"""Smoke test for `scripts/render_readme_examples.py`.

Runs the script against a scaled-down config and asserts the expected PNGs
land in docs/img/. Not run in CI (slow + non-deterministic floats).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_readme_examples.py"


@pytest.mark.bench
def test_render_readme_examples_smoke(tmp_path: Path) -> None:
    """End-to-end run — gated behind the bench marker so CI skips it."""
    env = os.environ.copy()
    env["DFXM_RENDER_OUTPUT_DIR"] = str(tmp_path)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--small"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    # Outputs land in $DFXM_RENDER_OUTPUT_DIR (tmp_path in the test).
    pngs = list(tmp_path.glob("example_*.png"))
    assert len(pngs) >= 2, f"expected at least 2 example PNGs, got {pngs}"


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_has_small_flag() -> None:
    """The `--small` flag must be supported (used by the bench test + docs)."""
    text = SCRIPT.read_text()
    assert "--small" in text
