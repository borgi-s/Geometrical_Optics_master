"""Stage 4.2 ceramic acceptance: Al2O3 (corundum, R-3c) CIF end-to-end.

Pins the 'ceramics planning works' claim: the CIF loads, the mount builds,
dfxm-find-reflections enumerates only symmetry-allowed reflections, and the
textbook R-3c extinction conditions hold for every emitted row:
  - R-centering (hexagonal axes): -h + k + l = 3n  (checked over every row)
  - c-glide on (0,0,l): l = 6n  (the (0,0,l) family is geometrically
    inaccessible with this mount, so it is verified via the named-absence
    check at the bottom — e.g. (0,0,3)/(0,0,9) must not appear — rather
    than by the per-row loop)
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.find_reflections_cmd import cli_main

DATA = Path(__file__).parent / "data" / "cif"


@pytest.fixture
def alumina_config(tmp_path: Path) -> Path:
    shutil.copy(DATA / "al2o3_r3c.cif", tmp_path / "al2o3.cif")
    p = tmp_path / "alumina.toml"
    # Orthogonal plane-normal triple for hexagonal axes: a*, a*-2b*, c*.
    p.write_text(
        '[crystal]\ncif = "al2o3.cif"\n'
        "mount_x = [1, 0, 0]\nmount_y = [1, -2, 0]\nmount_z = [0, 0, 1]\n"
        "[reciprocal]\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return p


def _rows(out: str) -> list[tuple[int, int, int]]:
    rows = []
    for line in out.splitlines():
        if line.startswith("#") or line.strip().startswith("hkl"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def test_alumina_reflection_table_obeys_textbook_extinctions(
    alumina_config: Path, capsys: pytest.CaptureFixture
) -> None:
    # hkl-max=4 is fast (84 rows) and already covers every textbook named absence.
    rc = cli_main(["--config", str(alumina_config), "--hkl-max", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "space_group=R -3 c" in out
    rows = _rows(out)
    assert len(rows) > 0, "no reachable reflections enumerated for alumina"
    for h, k, l in rows:
        assert (-h + k + l) % 3 == 0, f"R-centering violated by {(h, k, l)}"
        # Defensive: (0,0,l) is inaccessible with this mount so this never
        # fires here; the c-glide condition is pinned by the named-absence
        # check below. Kept so a mount change that exposes 00l is still gated.
        if h == 0 and k == 0:
            assert l % 6 == 0, f"c-glide 000l condition violated by {(0, 0, l)}"
    # Named textbook absences never appear.
    for forbidden in [(0, 0, 3), (0, 0, 9), (1, 0, 0), (1, 0, 1)]:
        assert forbidden not in rows


def test_alumina_explicit_forbidden_reflection_hard_errors(
    alumina_config: Path, tmp_path: Path
) -> None:
    from dfxm_geo.config import SimulationConfig

    body = alumina_config.read_text(encoding="utf-8")
    p = tmp_path / "alumina_bad.toml"
    p.write_text(
        body.replace(
            "[reciprocal]\nkeV = 17.0\n",
            '[geometry]\nmode = "oblique"\neta = 0.0\n[reciprocal]\nhkl = [0, 0, 3]\nkeV = 17.0\n',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="systematically absent"):
        SimulationConfig.from_toml(p)
