"""Stage 4.2: dfxm-bootstrap rejects extinct hkl + resolves relative cif paths."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.reciprocal_space.kernel import cli_main

DATA = Path(__file__).parent / "data" / "cif"


def _config(tmp_path: Path, hkl: str) -> Path:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    p = tmp_path / "config.toml"
    p.write_text(
        '[crystal]\ncif = "al.cif"\n'  # relative on purpose: base_dir test
        "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        '[geometry]\nmode = "oblique"\neta = 0.0\n'
        f"[reciprocal]\nhkl = {hkl}\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return p


def test_bootstrap_rejects_extinct_hkl(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli_main(["--config", str(_config(tmp_path, "[1, 0, 0]"))])
    assert rc == 1
    assert "systematically absent" in capsys.readouterr().err


def test_bootstrap_resolves_relative_cif_and_validates(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Allowed hkl gets PAST mount parsing + extinction; avoid the minutes-long
    # kernel run by pointing --output at an existing file WITHOUT --force, so
    # the overwrite guard exits 1 AFTER validation succeeds.
    out = tmp_path / "kernel.npz"
    out.write_bytes(b"placeholder")
    rc = cli_main(["--config", str(_config(tmp_path, "[-1, 1, -1]")), "--output", str(out)])
    err = capsys.readouterr().err
    assert "systematically absent" not in err
    assert "not found" not in err  # relative cif path resolved
    assert rc == 1  # refused overwrite — but only after the parse succeeded
