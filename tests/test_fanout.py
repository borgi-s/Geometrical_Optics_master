"""scripts/fanout.py — config discovery, worker env, and orchestration."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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


# ---------------------------------------------------------------------------
# Part B: --mode {forward,identify}
# ---------------------------------------------------------------------------


def test_build_cmd_forward_mode_includes_no_postprocess(tmp_path: Path) -> None:
    """Forward mode: build_cmd must produce the cli_main prefix + --no-postprocess."""
    cfg = tmp_path / "fwd.toml"
    cfg.write_text("")
    out_dir = tmp_path / "out"
    cmd = fanout.build_cmd("forward", cfg, out_dir)
    cmd_str = " ".join(cmd)
    # Must invoke cli_main (forward), not cli_main_identify
    assert "cli_main" in cmd_str
    assert "cli_main_identify" not in cmd_str
    # Must include --no-postprocess (forward-only flag)
    assert "--no-postprocess" in cmd
    # Must include --config and --output
    assert "--config" in cmd
    assert "--output" in cmd


def test_build_cmd_identify_mode_omits_no_postprocess(tmp_path: Path) -> None:
    """Identify mode: build_cmd must use cli_main_identify and OMIT --no-postprocess."""
    cfg = tmp_path / "id.toml"
    cfg.write_text("")
    out_dir = tmp_path / "out"
    cmd = fanout.build_cmd("identify", cfg, out_dir)
    cmd_str = " ".join(cmd)
    # Must invoke cli_main_identify
    assert "cli_main_identify" in cmd_str
    # Must NOT include --no-postprocess (identify has no such flag)
    assert "--no-postprocess" not in cmd
    # Must include --config and --output
    assert "--config" in cmd
    assert "--output" in cmd


def test_build_cmd_forward_is_unchanged_regression(tmp_path: Path) -> None:
    """Regression: forward mode build_cmd must be backward-compatible with the
    previous hardcoded _FORWARD_PREFIX + --no-postprocess behaviour."""
    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    out_dir = tmp_path / "out"
    cmd = fanout.build_cmd("forward", cfg, out_dir)
    # The first three elements are sys.executable + -c + import-and-call fragment
    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    assert "cli_main" in cmd[2]
    assert str(cfg) in cmd
    assert str(out_dir) in cmd


def test_run_manifest_forward_mode_uses_build_cmd(tmp_path: Path) -> None:
    """Smoke test: run_manifest accepts mode='forward' kwarg without crashing.

    The actual mode dispatch to build_cmd is verified by the build_cmd tests
    and test_run_manifest_routes_mode_to_default_runner.
    """
    configs = [tmp_path / "c0.toml"]
    configs[0].write_text("")

    def fake_runner(config, output_dir, env, log_path):
        # In the real impl _default_runner calls build_cmd(mode, ...).
        # The test verifies run_manifest passes mode correctly by checking
        # that the seam receives the right arguments (via build_cmd).
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    # Just ensure run_manifest(mode="forward") doesn't raise
    results = fanout.run_manifest(
        configs, tmp_path / "out", n_workers=1, runner=fake_runner, mode="forward"
    )
    assert results[0].returncode == 0


def test_run_manifest_identify_mode_accepted(tmp_path: Path) -> None:
    """Smoke test: run_manifest accepts mode='identify' kwarg without crashing.

    The actual mode dispatch is verified by the build_cmd tests and
    test_run_manifest_routes_mode_to_default_runner.
    """
    cfg = tmp_path / "id.toml"
    cfg.write_text("")

    def fake_runner(config, output_dir, env, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest(
        [cfg], tmp_path / "out", n_workers=1, runner=fake_runner, mode="identify"
    )
    assert results[0].returncode == 0


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
        isolate=True,  # subprocess path: each child pays its own import cost
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
        # The real child emitted the machine-readable timing line.
        timing = fanout.parse_timing_log(out_root / f"{stem}.log")
        assert timing.get("import_s", 0) > 0, f"no DFXM_TIMING in {stem}.log"
        assert timing.get("run_s", 0) > 0


def test_run_manifest_routes_mode_to_default_runner(tmp_path: Path, monkeypatch) -> None:
    """With runner=None + isolate=True (the subprocess path), run_manifest must
    thread `mode` through to _default_runner.

    isolate=True is used here because runner=None now dispatches to the pool
    by default (v2.6.0); isolate=True keeps this test on the subprocess path
    without injecting a runner seam.
    """
    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    seen: dict[str, str] = {}

    def capturing_default_runner(config, output_dir, env, log_path, *, mode="forward"):
        seen["mode"] = mode
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(fanout, "_default_runner", capturing_default_runner)
    fanout.run_manifest([cfg], tmp_path / "out", n_workers=1, mode="identify", isolate=True)
    assert seen["mode"] == "identify"


def test_build_cmd_rejects_unknown_mode(tmp_path: Path) -> None:
    """build_cmd is a public function; an unknown mode must fail loudly."""
    with pytest.raises(ValueError):
        fanout.build_cmd("nonsense", tmp_path / "c.toml", tmp_path / "out")


# ---------------------------------------------------------------------------
# Part C: --timing-json (M1 Phase 2a — measure before optimizing)
# ---------------------------------------------------------------------------


def test_build_cmd_snippet_emits_timing_line(tmp_path: Path) -> None:
    """The child -c snippet must time import vs run separately and print a
    DFXM_TIMING json line, so fanout can split fixed startup cost from work."""
    for mode in ("forward", "identify"):
        cmd = fanout.build_cmd(mode, tmp_path / "c.toml", tmp_path / "out")
        snippet = cmd[2]
        assert "DFXM_TIMING" in snippet, mode
        assert "import_s" in snippet, mode
        assert "run_s" in snippet, mode


def test_parse_timing_log_extracts_last_timing_line(tmp_path: Path) -> None:
    log = tmp_path / "c.log"
    log.write_text(
        "Defining properties of rays\n"
        'DFXM_TIMING {"import_s": 9.9, "run_s": 1.0}\n'
        "Wrote 22 images to out\n"
        'DFXM_TIMING {"import_s": 1.25, "run_s": 7.5}\n',
        encoding="utf-8",
    )
    got = fanout.parse_timing_log(log)
    assert got == {"import_s": 1.25, "run_s": 7.5, "images": 22}


def test_parse_timing_log_captures_samples_line(tmp_path: Path) -> None:
    """identify-multi prints 'Wrote N samples' (scenes, not frames) — captured
    under its own key so it is never conflated with an image count."""
    log = tmp_path / "c.log"
    log.write_text(
        'DFXM_TIMING {"import_s": 1.0, "run_s": 2.0}\nWrote 2 samples to out\n',
        encoding="utf-8",
    )
    assert fanout.parse_timing_log(log) == {"import_s": 1.0, "run_s": 2.0, "samples": 2}


def test_parse_timing_log_captures_zscan_configurations_line(tmp_path: Path) -> None:
    """identify-zscan prints 'Wrote N configurations' — captured under its own
    key so z-scan sweeps still get a per-config count in the manifest."""
    log = tmp_path / "c.log"
    log.write_text(
        'DFXM_TIMING {"import_s": 1.0, "run_s": 2.0}\nWrote 3 configurations to out\n',
        encoding="utf-8",
    )
    assert fanout.parse_timing_log(log) == {"import_s": 1.0, "run_s": 2.0, "configurations": 3}


def test_parse_timing_log_tolerates_missing_data(tmp_path: Path) -> None:
    no_line = tmp_path / "plain.log"
    no_line.write_text("just output, no timing\n", encoding="utf-8")
    assert fanout.parse_timing_log(no_line) == {}
    assert fanout.parse_timing_log(tmp_path / "nonexistent.log") == {}
    garbled = tmp_path / "bad.log"
    garbled.write_text("DFXM_TIMING {not json}\n", encoding="utf-8")
    assert fanout.parse_timing_log(garbled) == {}


def test_write_timing_json(tmp_path: Path) -> None:
    """write_timing_json assembles per-config rows + sweep-level throughput."""
    import json

    logs = []
    for i, (imp, run, images) in enumerate([(1.5, 8.5, 22), (1.4, 3.6, 22)]):
        log = tmp_path / f"c{i}.log"
        log.write_text(
            f'DFXM_TIMING {{"import_s": {imp}, "run_s": {run}}}\nWrote {images} images to out\n',
            encoding="utf-8",
        )
        logs.append(log)
    results = [
        fanout.ConfigResult(tmp_path / "c0.toml", tmp_path / "out" / "c0", 0, 10.0, logs[0]),
        fanout.ConfigResult(tmp_path / "c1.toml", tmp_path / "out" / "c1", 0, 5.0, logs[1]),
    ]
    out_json = tmp_path / "timing.json"
    fanout.write_timing_json(
        out_json,
        results,
        total_wall_s=10.0,
        n_workers=2,
        threads_per_worker=4,
        mode="identify",
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    meta = data["sweep"]
    assert meta["n_configs"] == 2
    assert meta["n_ok"] == 2
    assert meta["n_workers"] == 2
    assert meta["threads_per_worker"] == 4
    assert meta["mode"] == "identify"
    assert meta["total_wall_s"] == 10.0
    assert meta["configs_per_hour"] == pytest.approx(2 / 10.0 * 3600)
    assert meta["images_total"] == 44
    assert meta["images_per_s"] == pytest.approx(44 / 10.0)
    rows = data["configs"]
    assert rows[0]["config"].endswith("c0.toml")
    assert rows[0]["wall_s"] == 10.0
    assert rows[0]["returncode"] == 0
    assert rows[0]["import_s"] == 1.5
    assert rows[0]["run_s"] == 8.5
    assert rows[0]["images"] == 22


def test_write_timing_json_partial_child_data(tmp_path: Path) -> None:
    """Configs whose log lacks DFXM_TIMING (crash, old CLI) still get a row;
    sweep aggregates that need images are omitted rather than wrong."""
    import json

    log = tmp_path / "c0.log"
    log.write_text("crashed before timing\n", encoding="utf-8")
    results = [fanout.ConfigResult(tmp_path / "c0.toml", tmp_path / "out" / "c0", -1, 2.0, log)]
    out_json = tmp_path / "timing.json"
    fanout.write_timing_json(
        out_json, results, total_wall_s=2.0, n_workers=1, threads_per_worker=1, mode="forward"
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["sweep"]["n_ok"] == 0
    assert "images_total" not in data["sweep"]
    assert "images_per_s" not in data["sweep"]
    row = data["configs"][0]
    assert row["returncode"] == -1
    assert "import_s" not in row


# ---------------------------------------------------------------------------
# Pool mode (v2.6.0 W1). Tests inject worker_fn + executor_factory so no real
# process pool is spawned: a ThreadPoolExecutor exercises the same
# submit/as_completed/recovery code paths and imposes no picklability
# constraint on the fakes. Fidelity gap: a ThreadPoolExecutor cannot
# reproduce the all-futures-poisoned cascade of a real broken
# ProcessPoolExecutor — the real-pool paths are covered by the e2e tests
# (pool-vs-isolate bit-identity gate + pool forward e2e) further below.
# ---------------------------------------------------------------------------


def _ok_worker(mode, config, output_dir, log_path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(
        'DFXM_TIMING {"import_s": 1.5, "run_s": 2.5}\nWrote 4 images to x\n',
        encoding="utf-8",
    )
    return {"returncode": 0, "wall_s": 0.01}


def test_run_pool_basic(tmp_path: Path) -> None:
    configs = [tmp_path / f"c{i}.toml" for i in range(3)]
    for c in configs:
        c.write_text("")
    results = fanout.run_pool(
        configs,
        tmp_path / "out",
        n_workers=2,
        mode="identify",
        worker_fn=_ok_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    assert len(results) == 3
    assert all(r.returncode == 0 for r in results)
    assert {r.config for r in results} == set(configs)
    # Results come back in input order regardless of completion order.
    assert [r.config for r in results] == configs
    timing = fanout.parse_timing_log(results[0].log_path)
    assert timing["import_s"] == 1.5 and timing["images"] == 4


def test_run_pool_worker_dict_failure_is_contained(tmp_path: Path) -> None:
    c = tmp_path / "bad.toml"
    c.write_text("")

    def failing_worker(mode, config, output_dir, log_path):
        Path(log_path).write_text("boom\n", encoding="utf-8")
        return {"returncode": -1, "wall_s": 0.0}

    results = fanout.run_pool(
        [c],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=failing_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    assert results[0].returncode == -1


def test_run_pool_raising_worker_is_contained(tmp_path: Path) -> None:
    """A worker_fn that raises (not via run_one's guard) is rc=-1 + log."""
    good, bad = tmp_path / "good.toml", tmp_path / "boom.toml"
    good.write_text("")
    bad.write_text("")

    def worker(mode, config, output_dir, log_path):
        if "boom" in config:
            raise RuntimeError("worker exploded")
        return _ok_worker(mode, config, output_dir, log_path)

    results = fanout.run_pool(
        [good, bad],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    by_name = {r.config.name: r for r in results}
    assert by_name["good.toml"].returncode == 0
    assert by_name["boom.toml"].returncode == -1
    assert "worker exploded" in by_name["boom.toml"].log_path.read_text(encoding="utf-8")


def test_run_pool_recovers_from_broken_pool(tmp_path: Path) -> None:
    """First executor 'hard-crashes' on one config; run_pool rebuilds the
    executor and retries; second attempt succeeds. The other config's result
    survives. A config that breaks the pool twice is marked rc=-2."""
    ok = tmp_path / "ok.toml"
    flaky = tmp_path / "flaky.toml"
    fatal = tmp_path / "fatal.toml"
    for c in (ok, flaky, fatal):
        c.write_text("")
    attempts: dict[str, int] = {}

    def worker(mode, config, output_dir, log_path):
        name = Path(config).name
        attempts[name] = attempts.get(name, 0) + 1
        if name == "flaky.toml" and attempts[name] == 1:
            raise BrokenProcessPool("simulated worker death")
        if name == "fatal.toml":
            raise BrokenProcessPool("simulated worker death")
        return _ok_worker(mode, config, output_dir, log_path)

    results = fanout.run_pool(
        [ok, flaky, fatal],
        tmp_path / "out",
        n_workers=1,
        mode="identify",
        worker_fn=worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    by_name = {r.config.name: r for r in results}
    assert by_name["ok.toml"].returncode == 0
    assert by_name["flaky.toml"].returncode == 0  # retried, then fine
    assert by_name["fatal.toml"].returncode == -2  # gave up after 2 attempts
    assert attempts["fatal.toml"] == 2


def test_run_manifest_dispatches_pool_by_default(tmp_path: Path, monkeypatch) -> None:
    c = tmp_path / "c.toml"
    c.write_text("")
    called = {}

    def fake_run_pool(configs, output_root, **kwargs):
        called["pool"] = True
        return []

    monkeypatch.setattr(fanout, "run_pool", fake_run_pool)
    fanout.run_manifest([c], tmp_path / "out", n_workers=1, mode="identify")
    assert called.get("pool") is True


def test_run_manifest_isolate_uses_subprocess_path(tmp_path: Path, monkeypatch) -> None:
    c = tmp_path / "c.toml"
    c.write_text("")

    def fake_run_pool(*a, **k):  # must NOT be called
        raise AssertionError("pool path used despite isolate=True")

    monkeypatch.setattr(fanout, "run_pool", fake_run_pool)

    def fake_runner(config, output_dir, env, log_path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest(
        [c], tmp_path / "out", n_workers=1, runner=fake_runner, isolate=True
    )
    assert results[0].returncode == 0


def test_run_manifest_runner_seam_implies_isolate(tmp_path: Path, monkeypatch) -> None:
    """Back-compat: every existing test that injects `runner` keeps testing
    the subprocess orchestration without passing isolate."""
    c = tmp_path / "c.toml"
    c.write_text("")
    monkeypatch.setattr(fanout, "run_pool", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    def fake_runner(config, output_dir, env, log_path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return 0

    results = fanout.run_manifest([c], tmp_path / "out", n_workers=1, runner=fake_runner)
    assert results[0].returncode == 0


def test_run_pool_sizes_executor_to_n_workers(tmp_path: Path) -> None:
    """Spec §7.5 concurrency cap: the executor is created with n_workers —
    the pool's concurrency limit is delegated to the executor itself."""
    configs = [tmp_path / f"c{i}.toml" for i in range(4)]
    for c in configs:
        c.write_text("")
    sizes: list[int] = []

    def recording_factory(n: int):
        sizes.append(n)
        return ThreadPoolExecutor(max_workers=n)

    fanout.run_pool(
        configs,
        tmp_path / "out",
        n_workers=3,
        mode="identify",
        worker_fn=_ok_worker,
        executor_factory=recording_factory,
    )
    assert sizes == [3]


def test_pool_env_pinned_during_run_and_restored_after(tmp_path: Path) -> None:
    import os

    c = tmp_path / "c.toml"
    c.write_text("")
    seen_env: dict[str, str | None] = {}

    def env_spy_worker(mode, config, output_dir, log_path):
        seen_env["DFXM_MAX_WORKERS"] = os.environ.get("DFXM_MAX_WORKERS")
        seen_env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
        return _ok_worker(mode, config, output_dir, log_path)

    before = os.environ.get("DFXM_MAX_WORKERS")
    fanout.run_pool(
        [c],
        tmp_path / "out",
        n_workers=1,
        threads_per_worker=3,
        mode="identify",
        worker_fn=env_spy_worker,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    # ThreadPoolExecutor shares the parent env, so the spy observes exactly
    # what spawned children would inherit at submit time.
    assert seen_env["DFXM_MAX_WORKERS"] == "3"
    assert seen_env["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("DFXM_MAX_WORKERS") == before  # restored
