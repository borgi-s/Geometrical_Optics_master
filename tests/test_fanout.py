"""scripts/fanout.py — config discovery, worker env, and orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/fanout.py as a module (scripts/ is not a package).
_FANOUT = Path(__file__).resolve().parents[1] / "scripts" / "fanout.py"
_spec = importlib.util.spec_from_file_location("fanout", _FANOUT)
fanout = importlib.util.module_from_spec(_spec)
sys.modules["fanout"] = fanout
_spec.loader.exec_module(fanout)


def test_discover_configs_from_directory(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("")
    (tmp_path / "b.toml").write_text("")
    (tmp_path / "note.txt").write_text("ignore me")
    got = fanout.discover_configs(tmp_path)
    assert [p.name for p in got] == ["a.toml", "b.toml"]  # sorted, .toml only


def test_discover_configs_from_list_file(tmp_path: Path) -> None:
    c1 = tmp_path / "one.toml"
    c1.write_text("")
    c2 = tmp_path / "two.toml"
    c2.write_text("")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"# configs\n{c1}\n\n{c2}\n")
    got = fanout.discover_configs(manifest)
    assert got == [c1, c2]  # blank + comment lines skipped, order preserved


def test_discover_configs_single_toml(tmp_path: Path) -> None:
    c1 = tmp_path / "solo.toml"
    c1.write_text("")
    assert fanout.discover_configs(c1) == [c1]


def test_discover_configs_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        fanout.discover_configs(tmp_path)


def test_worker_env_caps_threads() -> None:
    env = fanout.worker_env(threads_per_worker=16, base_env={"PATH": "/x"})
    assert env["DFXM_MAX_WORKERS"] == "16"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert env["PATH"] == "/x"  # base env preserved
