"""Smoke checks for essential README.md content.

The exact section structure is fluid (README is rewritten periodically for
audience changes — e.g. internal-developer-focused vs PyPI-landing-page).
These tests check for durable substrings any user needs to find, not for
specific headings.
"""

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


class TestReadmeContent:
    def test_mentions_template_dirs(self) -> None:
        """HPC users need to find the batch templates."""
        text = _text()
        assert "lsf/" in text
        assert "slurm/" in text

    def test_mentions_dfxm_bootstrap(self) -> None:
        """Two-step workflow is the main install / run sequence."""
        assert "dfxm-bootstrap" in _text()

    def test_mentions_dfxm_forward(self) -> None:
        """The forward-simulation CLI is the primary user-facing tool."""
        assert "dfxm-forward" in _text()

    def test_mentions_pip_install(self) -> None:
        """Essential for a publishable package: users need a clear install path."""
        assert "pip install" in _text()

    def test_mentions_paper_doi(self) -> None:
        """The IUCrJ 2024 paper is the canonical reference for the model."""
        assert "10.1107/S1600576724001183" in _text()
