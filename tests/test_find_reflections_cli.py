"""dfxm-find-reflections CLI — Table A.2-style enumeration."""

from __future__ import annotations

import pytest

from dfxm_geo.find_reflections_cmd import cli_main

PAPER_CONFIG = """
[reciprocal]
keV = 19.1

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


@pytest.fixture
def paper_config(tmp_path):
    p = tmp_path / "paper.toml"
    p.write_text(PAPER_CONFIG, encoding="utf-8")
    return p


def test_table_a2_group1_rows_present(paper_config, capsys):
    rc = cli_main(["--config", str(paper_config), "--hkl-max", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    # Paper Table A.2 group 1: η = 20.233°, θ = 15.417° — all four {113} variants
    for hkl_str in ("1 1 3", "-1 -1 3", "1 -1 3", "-1 1 3"):
        assert hkl_str in out
    assert "20.23" in out  # eta, degrees
    assert "15.41" in out  # theta, degrees


def test_eta_target_filters(paper_config, capsys):
    rc = cli_main(
        [
            "--config",
            str(paper_config),
            "--eta-target-deg",
            "20.233",
            "--hkl-max",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 1 3" in out
    # Verify no {111} reflections (different eta group) appear.
    # Parse lines: hkl is the first column, check no line has hkl matching 1 1 1 family.
    # Checking " 1 1 1" (with leading space from right-aligned column) avoids
    # false-match inside e.g. "-1 1 1" — but to be safe, parse each data line.
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("#")]
    for line in lines:
        # The hkl field is the first 10 chars (right-aligned), strip it.
        hkl_part = line[:10].strip()
        # Reject any {111} family hkl
        assert hkl_part not in (
            "1 1 1",
            "-1 -1 1",
            "1 -1 1",
            "-1 1 1",
            "1 1 -1",
            "-1 -1 -1",
            "1 -1 -1",
            "-1 1 -1",
        )


def test_missing_config_errors(tmp_path, capsys):
    rc = cli_main(["--config", str(tmp_path / "nope.toml")])
    assert rc != 0


def test_default_mount_when_no_crystal_block(tmp_path, capsys):
    p = tmp_path / "min.toml"
    p.write_text("[reciprocal]\nkeV = 19.1\n", encoding="utf-8")
    rc = cli_main(["--config", str(p), "--hkl-max", "2"])
    assert rc == 0
    assert "keV" in capsys.readouterr().out
