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

# Minimal config with no [reciprocal] block — used to test the keV fallback warning.
CRYSTAL_ONLY_CONFIG = """
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
    # Skip comment lines (start with '#') AND the column-header line (first column
    # is the literal string 'hkl', not a Miller index).
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("#")]
    for line in lines:
        # The hkl field is the first 12 chars (right-aligned), strip it.
        hkl_part = line[:12].strip()
        # Skip the column-header line — its first column is 'hkl', not a Miller index.
        if hkl_part == "hkl":
            continue
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


def test_malformed_toml_returns_rc1_with_message(tmp_path, capsys):
    """Malformed TOML must return rc=1 and print a diagnostic to stderr."""
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not = [valid toml\n", encoding="utf-8")
    rc = cli_main(["--config", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid TOML" in err


def test_no_kev_in_config_warns_on_stderr(tmp_path, capsys):
    """When neither --keV nor [reciprocal] keV is present, a warning goes to stderr."""
    cfg = tmp_path / "crystal_only.toml"
    cfg.write_text(CRYSTAL_ONLY_CONFIG, encoding="utf-8")
    rc = cli_main(["--config", str(cfg), "--hkl-max", "1"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "defaulting to 17.0" in err


HEX_CONFIG = """
[reciprocal]
keV = 17.0

[crystal]
lattice = "hexagonal"
a       = 3.2094e-10
c       = 5.2108e-10
mount_x = [2, -1, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


def test_hexagonal_config_enumerates(tmp_path, capsys):
    p = tmp_path / "mg.toml"
    p.write_text(HEX_CONFIG, encoding="utf-8")
    rc = cli_main(["--config", str(p), "--hkl-max", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 -1 0" in out
