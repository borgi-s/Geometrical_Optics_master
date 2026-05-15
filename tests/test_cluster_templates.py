"""Structural checks on the LSF / SLURM batch templates.

These don't submit anything; they assert the templates exist, declare the
right scheduler directives, include the EDIT THESE block, and reference
the right CLI entry points and docs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


class TestLsfForwardSingle:
    rel = "lsf/forward_single.bsub"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_bsub_directives(self) -> None:
        text = _read(self.rel)
        assert text.startswith("#!/bin/bash"), "missing shebang"
        # Core LSF directives.
        for directive in ["#BSUB -J", "#BSUB -q", "#BSUB -W", "#BSUB -R", "#BSUB -o", "#BSUB -e"]:
            assert directive in text, f"missing {directive}"

    def test_default_queue_is_hpc(self) -> None:
        assert "#BSUB -q hpc" in _read(self.rel)

    def test_invokes_forward_cli(self) -> None:
        text = _read(self.rel)
        assert "dfxm-forward" in text
        assert "dfxm-bootstrap" in text, "template should remind users to bootstrap once"

    def test_edit_these_block(self) -> None:
        text = _read(self.rel)
        assert "EDIT THESE" in text
