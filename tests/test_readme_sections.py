"""Smoke checks that README.md has the cluster + examples sections required by v1.0."""

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _text() -> str:
    return README.read_text()


class TestReadmeClusterSection:
    def test_has_cluster_section(self) -> None:
        text = _text()
        assert "## Running on a cluster" in text or "# Running on a cluster" in text

    def test_links_to_cluster_runs(self) -> None:
        assert "docs/cluster-runs.md" in _text()

    def test_mentions_template_dirs(self) -> None:
        text = _text()
        assert "lsf/" in text
        assert "slurm/" in text

    def test_mentions_dfxm_bootstrap(self) -> None:
        assert "dfxm-bootstrap" in _text()
