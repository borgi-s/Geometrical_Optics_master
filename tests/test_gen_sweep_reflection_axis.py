"""Reflection axis in the sweep generators (--hkl-list / --keV)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_identify_sweep_configs  # noqa: E402
import gen_sweep_configs  # noqa: E402

# ---------------------------------------------------------------------------
# gen_identify_sweep_configs tests
# ---------------------------------------------------------------------------


def test_hkl_list_multiplies_configs(tmp_path):
    gen_identify_sweep_configs.main(
        [
            "--n-configs",
            "2",
            "--out-dir",
            str(tmp_path),
            "--hkl-list",
            "1,1,1;2,0,0",
        ]
    )
    tomls = sorted(tmp_path.glob("*.toml"))
    assert len(tomls) == 4  # 2 seeds x 2 reflections
    hkls = set()
    for p in tomls:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        hkls.add(tuple(data["reciprocal"]["hkl"]))
        assert "hkl" in p.stem  # filename carries the reflection token
    assert hkls == {(1, 1, 1), (2, 0, 0)}


def test_kev_override(tmp_path):
    gen_identify_sweep_configs.main(
        [
            "--n-configs",
            "1",
            "--out-dir",
            str(tmp_path),
            "--keV",
            "19.1",
        ]
    )
    p = next(iter(tmp_path.glob("*.toml")))
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["reciprocal"]["keV"] == pytest.approx(19.1)


def test_default_behavior_unchanged(tmp_path):
    """No --hkl-list → legacy filenames (no hkl token) and Al-111 defaults."""
    gen_identify_sweep_configs.main(["--n-configs", "1", "--out-dir", str(tmp_path)])
    p = next(iter(tmp_path.glob("*.toml")))
    assert "hkl" not in p.stem
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert tuple(data["reciprocal"]["hkl"]) == (-1, 1, -1)
    assert data["reciprocal"]["keV"] == pytest.approx(17.0)


# ---------------------------------------------------------------------------
# gen_sweep_configs tests
# ---------------------------------------------------------------------------


def test_sweep_hkl_list_multiplies_configs(tmp_path):
    """--hkl-list '1,1,1;2,0,0' → 8 base configs x 2 reflections = 16 files."""
    gen_sweep_configs.main(
        [
            "--hkl-list",
            "1,1,1;2,0,0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    tomls = sorted(tmp_path.glob("*.toml"))
    assert len(tomls) == 16  # 8 base configs x 2 reflections
    for p in tomls:
        assert "hkl" in p.stem  # filename carries the reflection token
    hkls = set()
    for p in tomls:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        hkls.add(tuple(data["reciprocal"]["hkl"]))
    assert hkls == {(1, 1, 1), (2, 0, 0)}


def test_sweep_default_behavior_unchanged(tmp_path):
    """No --hkl-list → 8 files with legacy names and Al-111 defaults."""
    gen_sweep_configs.main(["--out-dir", str(tmp_path)])
    tomls = sorted(tmp_path.glob("*.toml"))
    assert len(tomls) == 8
    for p in tomls:
        assert "hkl" not in p.stem  # no reflection token in legacy names
    for p in tomls:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        assert tuple(data["reciprocal"]["hkl"]) == (-1, 1, -1)
        assert data["reciprocal"]["keV"] == pytest.approx(17.0)
