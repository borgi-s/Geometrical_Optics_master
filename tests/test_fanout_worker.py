"""dfxm_geo.fanout_worker.run_one — the pool-mode in-process config runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfxm_geo import fanout_worker


@pytest.fixture(autouse=True)
def _reset_import_timer(monkeypatch):
    # Each test sees a "fresh worker": first run_one reports a real import_s.
    monkeypatch.setattr(fanout_worker, "_import_timed", False)


def test_run_one_rejects_unknown_mode(tmp_path: Path):
    res = fanout_worker.run_one(
        "nonsense", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log")
    )
    assert res["returncode"] != 0
    assert "nonsense" in (tmp_path / "c.log").read_text(encoding="utf-8")


def test_run_one_calls_identify_cli_and_writes_timing(tmp_path: Path, monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_cli(argv):
        seen["argv"] = argv
        print("Wrote 4 samples to somewhere")
        return 0

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (fake_cli, []))
    log = tmp_path / "c.log"
    res = fanout_worker.run_one("identify", "cfg.toml", str(tmp_path / "out"), str(log))
    assert res["returncode"] == 0
    assert res["wall_s"] >= 0
    assert seen["argv"][:4] == ["--config", "cfg.toml", "--output", str(tmp_path / "out")]
    text = log.read_text(encoding="utf-8")
    assert "Wrote 4 samples" in text  # stdout redirected into the log
    timing_line = [ln for ln in text.splitlines() if ln.startswith("DFXM_TIMING ")][-1]
    payload = json.loads(timing_line[len("DFXM_TIMING ") :])
    assert payload["run_s"] >= 0
    assert payload["import_s"] >= 0


def test_run_one_first_call_reports_import_then_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (lambda argv: 0, []))

    def timing_from(log: Path) -> dict:
        line = [
            ln
            for ln in log.read_text(encoding="utf-8").splitlines()
            if ln.startswith("DFXM_TIMING ")
        ][-1]
        return json.loads(line[len("DFXM_TIMING ") :])

    log1, log2 = tmp_path / "a.log", tmp_path / "b.log"
    fanout_worker.run_one("identify", "a.toml", str(tmp_path / "o1"), str(log1))
    fanout_worker.run_one("identify", "b.toml", str(tmp_path / "o2"), str(log2))
    # pipeline is already imported in the test process, so the first measured
    # value may be ~0 — the contract is: second call reports exactly 0.0.
    assert timing_from(log2)["import_s"] == 0.0


def test_run_one_exception_is_rc_minus_one_with_traceback(tmp_path: Path, monkeypatch):
    def boom(argv):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (boom, []))
    log = tmp_path / "c.log"
    res = fanout_worker.run_one("identify", "c.toml", str(tmp_path / "out"), str(log))
    assert res["returncode"] == -1
    assert "kaboom" in log.read_text(encoding="utf-8")


def test_run_one_systemexit_code_is_propagated(tmp_path: Path, monkeypatch):
    def argparse_style_exit(argv):
        raise SystemExit(2)

    monkeypatch.setattr(fanout_worker, "_resolve_cli", lambda mode: (argparse_style_exit, []))
    res = fanout_worker.run_one(
        "identify", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log")
    )
    assert res["returncode"] == 2


def test_forward_mode_appends_no_postprocess(tmp_path: Path, monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_cli(argv):
        seen["argv"] = argv
        return 0

    # forward mode resolves with the --no-postprocess extra
    monkeypatch.setattr(
        fanout_worker, "_resolve_cli", lambda mode: (fake_cli, ["--no-postprocess"])
    )
    fanout_worker.run_one("forward", "c.toml", str(tmp_path / "out"), str(tmp_path / "c.log"))
    assert seen["argv"][-1] == "--no-postprocess"
