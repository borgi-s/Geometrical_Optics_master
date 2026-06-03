"""scripts/fanout.py — config discovery, worker env, and orchestration."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm

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


def test_run_manifest_respects_concurrency_and_env(tmp_path: Path) -> None:
    configs = [tmp_path / f"c{i}.toml" for i in range(6)]
    for c in configs:
        c.write_text("")
    seen_envs = []
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake_runner(config, output_dir, env, log_path):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        seen_envs.append(dict(env))
        time.sleep(0.02)  # hold the slot so concurrency is observable
        with lock:
            active["now"] -= 1
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest(
        configs, tmp_path / "out", n_workers=2, threads_per_worker=7, runner=fake_runner
    )
    assert len(results) == 6
    assert all(r.returncode == 0 for r in results)
    assert active["max"] <= 2  # never more than n_workers at once
    assert {r.output_dir.name for r in results} == {f"c{i}" for i in range(6)}
    assert all(e["DFXM_MAX_WORKERS"] == "7" for e in seen_envs)


def test_run_manifest_reports_nonzero_returncode(tmp_path: Path) -> None:
    c = tmp_path / "bad.toml"
    c.write_text("")

    def failing_runner(config, output_dir, env, log_path):
        return 3

    results = fanout.run_manifest([c], tmp_path / "out", n_workers=1, runner=failing_runner)
    assert results[0].returncode == 3


def test_run_manifest_survives_runner_exception(tmp_path: Path) -> None:
    """A runner that raises (e.g. a config that can't spawn) is recorded as a
    failed result with rc=-1 + a log, not allowed to abort the whole batch."""
    good = tmp_path / "good.toml"
    bad = tmp_path / "boom.toml"
    good.write_text("")
    bad.write_text("")

    def flaky_runner(config, output_dir, env, log_path):
        if config.stem == "boom":
            raise RuntimeError("cannot spawn")
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest([good, bad], tmp_path / "out", n_workers=2, runner=flaky_runner)
    by_stem = {r.config.stem: r for r in results}
    assert by_stem["good"].returncode == 0
    assert by_stem["boom"].returncode == -1  # exception -> recorded failure
    assert by_stem["boom"].log_path.is_file()  # traceback captured
    assert "cannot spawn" in by_stem["boom"].log_path.read_text()


def test_fanout_end_to_end_runs_two_configs(tmp_path: Path) -> None:
    """Real subprocess integration: two tiny centered configs via the actual
    dfxm-forward CLI, thread-capped, each producing a valid float32 HDF5."""
    # Only the on-disk LUT npz is required: each config runs in its own
    # subprocess, which loads the LUT and computes Hg itself (centered crystal),
    # so the parent's `fm.Hg` global is irrelevant here.
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping integration run.")

    import h5py
    import numpy as np

    # Omit [crystal] entirely so it cascades to the default centered crystal
    # (b=(1,0,-1), n, t) — a partial [crystal.centered] would require b/n/t.
    cfg_text = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n"
        "[scan.phi]\nrange = 0.001\nsteps = 2\n\n"
        "[io]\ninclude_perfect_crystal = false\n\n"
        "[postprocess]\nenabled = false\n"
    )
    manifest_dir = tmp_path / "configs"
    manifest_dir.mkdir()
    for name in ("alpha.toml", "beta.toml"):
        (manifest_dir / name).write_text(cfg_text)

    out_root = tmp_path / "out"
    # Run the two configs SERIALLY (n_workers=1): each dfxm-forward subprocess
    # builds a full px510 Hg (~100 MB), and two concurrent renders plus the
    # resident pytest process exceed a memory-constrained box's RAM. The
    # scheduler's concurrency itself is covered by the mock-runner tests above
    # (the `active["max"] <= n_workers` assertion); this end-to-end test only
    # needs to confirm both real configs produce valid float32 HDF5.
    results = fanout.run_manifest(
        fanout.discover_configs(manifest_dir),
        out_root,
        n_workers=1,
        threads_per_worker=2,
    )
    assert len(results) == 2
    assert all(r.returncode == 0 for r in results), [r.log_path.read_text() for r in results]
    for stem in ("alpha", "beta"):
        det = out_root / stem / "scan0001" / "dfxm_sim_detector_0000.h5"
        assert det.is_file(), f"missing detector file for {stem}"
        with h5py.File(det, "r") as f:
            img = f["/entry_0000/dfxm_sim_detector/image"]
            assert img.dtype == np.float32
            assert img.shape[0] == 2  # 2 phi frames
