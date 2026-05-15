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


class TestLsfIdentifyArray:
    rel = "lsf/identify_array.bsub"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_array_directive(self) -> None:
        text = _read(self.rel)
        assert "#BSUB -J" in text
        # LSF array syntax: -J "name[1-N]"
        import re

        assert re.search(r"#BSUB -J [\"'][^\"']*\[\d+-\d+\]", text), (
            "LSF array template must declare -J 'name[1-N]'"
        )

    def test_uses_lsb_jobindex(self) -> None:
        """Tasks differentiate via $LSB_JOBINDEX."""
        assert "$LSB_JOBINDEX" in _read(self.rel) or "${LSB_JOBINDEX}" in _read(self.rel)

    def test_invokes_identify_cli(self) -> None:
        assert "dfxm-identify" in _read(self.rel)

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)


class TestSlurmForwardSingle:
    rel = "slurm/forward_single.sbatch"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_sbatch_directives(self) -> None:
        text = _read(self.rel)
        assert text.startswith("#!/bin/bash")
        for directive in [
            "#SBATCH --job-name",
            "#SBATCH --time",
            "#SBATCH --output",
            "#SBATCH --error",
            "#SBATCH --cpus-per-task",
            "#SBATCH --mem",
        ]:
            assert directive in text, f"missing {directive}"

    def test_mentions_sinfo_callout(self) -> None:
        """SLURM templates flag partition naming as cluster-specific via `sinfo`."""
        assert "sinfo" in _read(self.rel)

    def test_invokes_forward_cli(self) -> None:
        text = _read(self.rel)
        assert "dfxm-forward" in text
        assert "dfxm-bootstrap" in text

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)


class TestSlurmIdentifyArray:
    rel = "slurm/identify_array.sbatch"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_array_directive(self) -> None:
        text = _read(self.rel)
        # SLURM array syntax: --array=1-N
        import re

        assert re.search(r"#SBATCH --array=\d+-\d+", text), (
            "SLURM array template must declare --array=N-M"
        )

    def test_uses_slurm_array_task_id(self) -> None:
        text = _read(self.rel)
        assert "$SLURM_ARRAY_TASK_ID" in text or "${SLURM_ARRAY_TASK_ID}" in text

    def test_invokes_identify_cli(self) -> None:
        assert "dfxm-identify" in _read(self.rel)

    def test_edit_these_block(self) -> None:
        assert "EDIT THESE" in _read(self.rel)


class TestClusterRunsDoc:
    rel = "docs/cluster-runs.md"

    def test_exists(self) -> None:
        assert (REPO_ROOT / self.rel).is_file()

    def test_has_required_sections(self) -> None:
        text = _read(self.rel)
        for heading in [
            "Two-step workflow",
            "DTU HPC",
            "ESRF",
            "Output handling",
            "Memory + walltime sizing",
            "DTU vs ESRF",
        ]:
            assert heading in text, f"missing section: {heading}"

    def test_references_templates(self) -> None:
        text = _read(self.rel)
        for path in [
            "lsf/forward_single.bsub",
            "lsf/identify_array.bsub",
            "slurm/forward_single.sbatch",
            "slurm/identify_array.sbatch",
        ]:
            assert path in text, f"missing reference to {path}"

    def test_references_clis(self) -> None:
        text = _read(self.rel)
        assert "dfxm-bootstrap" in text
        assert "dfxm-forward" in text
        assert "dfxm-identify" in text

    def test_warns_lsf_vs_slurm(self) -> None:
        """DTU is LSF (bsub), ESRF is SLURM (sbatch). The doc must say so explicitly."""
        text = _read(self.rel)
        assert "LSF" in text
        assert "SLURM" in text
        assert "bsub" in text
        assert "sbatch" in text
