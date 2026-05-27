# HDF5 float32 + config fan-out (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Halve per-config HDF5 size (detector dtype float64→float32, lossless) and add an in-node launcher (`scripts/fanout.py`) that runs ~8 config-worker processes × ~16 threads each, so a 128-core node is used as config-level fan-out rather than one over-threaded scan.

**Architecture:** Two independent sub-deliverables. (2a) Route every detector-`image` dataset creation through one `DETECTOR_DTYPE = np.float32` constant in `io/hdf5.py`; streaming writes auto-cast on assignment. (2b) A `scripts/fanout.py` with a testable seam: pure helpers (`discover_configs`, `worker_env`) + an orchestrator (`run_manifest`) that pools `dfxm-forward` subprocesses under a `ThreadPoolExecutor(max_workers=n_workers)`, each subprocess thread-capped via `DFXM_MAX_WORKERS` + single-threaded BLAS to avoid oversubscription. One `lsf/fanout.bsub` submits the batch.

**Tech Stack:** Python 3, NumPy, h5py, numba, pytest, LSF. Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Spec:** `docs/superpowers/specs/2026-05-27-forward-throughput-arc-design.md` (Phase 2).

**Prereq:** Phase 1 is merged to local `main` (`219ab34`). This plan builds on `main`; create a fresh branch `feature/hdf5-float32-and-fanout` off `main` before Task 1.

---

## Environment notes (every task)

- Repo root / cwd: `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`
- venv python for EVERYTHING: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`. Default `python` is Python 2.7 — never use it. Conda is broken.
- PowerShell on Windows; call the venv python with the call operator: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest ...`. (In the Bash tool, drop the `&` and quote the path.)
- A pre-commit hook (ruff, ruff-format, whitespace) runs on commit and may reformat — if it does, re-`git add` and re-commit. Verify multi-path commits landed with `git show --stat HEAD` (a `git add` of several paths stages nothing if one path errors).
- Many forward tests need a bootstrapped kernel; they self-skip when absent via:
  ```python
  kernel_dir = Path(fm.pkl_fpath)
  if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")) or fm.Hg is None:
      pytest.skip("No bootstrapped kernel; skipping.")
  ```
  Use this same guard in new tests that run the pipeline.

---

## Task 0: Branch off main

- [ ] **Step 1: Create the branch**

```bash
git checkout main
git checkout -b feature/hdf5-float32-and-fanout
git rev-parse --abbrev-ref HEAD   # expect: feature/hdf5-float32-and-fanout
```

No commit (branch creation only).

---

## Task 1: Detector dtype float64 → float32 (lossless, ~2× smaller)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (`_create_detector_skeleton`, ~line 125-174; add a module constant)
- Test: `tests/test_detector_dtype_float32.py`
- Modify (docs): `docs/output-format.md`

**Background:** `_create_detector_skeleton` (`hdf5.py:125`) is the single creation site for the detector `image` dataset. `_write_detector_file` and the streaming parallel writer both route through it. The streaming branch hardcodes `dtype=np.float64`; the inline branch (`data=data`) infers dtype from the array. Writing a float64 frame into a float32 dataset auto-casts on assignment, so changing the dataset dtype is sufficient for the streaming path.

- [ ] **Step 1: Write the failing test**

```python
"""Detector image stack is stored as float32 (Phase 2a: lossless ~2x slim)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    CrystalConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    WallCrystalConfig,
    run_simulation,
)


def _kernel_or_skip() -> None:
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")) or fm.Hg is None:
        pytest.skip("No bootstrapped kernel npz found; skipping.")


def test_detector_image_is_float32(tmp_path: Path) -> None:
    _kernel_or_skip()
    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        ),
        scan=ScanConfig(phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3)),
        io=IOConfig(include_perfect_crystal=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"]
        assert img.dtype == np.float32, f"expected float32, got {img.dtype}"
        data = img[()]
    # Sanity: real signal was written, not zeros/NaNs.
    assert np.isfinite(data).all()
    assert float(data.sum()) > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_dtype_float32.py -q`
Expected: FAIL on the dtype assertion (`expected float32, got float64`) — OR skip if no kernel. If it skips, bootstrap the kernel first: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo... ` — actually run `dfxm-bootstrap --if-missing --config configs/profile_rocking.toml` via the venv (`& "...python.exe" -m pip show dfxm-geo` confirms install; the `dfxm-bootstrap` console script is in `.venv\Scripts`). If you cannot bootstrap, note it and proceed — the test will skip, and the dtype change is still verified by the existing e2e suite in Step 5.

- [ ] **Step 3: Add the dtype constant and route both branches through it**

In `src/dfxm_geo/io/hdf5.py`, add near the top (after imports, with the other module constants):

```python
# Detector image storage dtype. float32 is lossless-enough for the simulated
# intensities (the forward model computes in float64; the ~7 significant
# decimal digits of float32 are well below shot-noise) and halves on-disk size
# vs float64. Phase 2a of the forward-throughput arc.
DETECTOR_DTYPE = np.float32
```

Then in `_create_detector_skeleton`, change BOTH `create_dataset("image", ...)` calls to pin the dtype. Inline branch:

```python
    if data is not None:
        img = det.create_dataset(
            "image",
            data=data,
            dtype=DETECTOR_DTYPE,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
```

Streaming branch:

```python
    else:
        img = det.create_dataset(
            "image",
            shape=(n_frames, height, width),
            dtype=DETECTOR_DTYPE,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
```

(Passing both `data=` and `dtype=` to h5py casts the array to the dataset dtype on write — float64→float32. The streaming writer's `img[idx] = im` likewise auto-casts.)

- [ ] **Step 4: Run the new test to verify it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_dtype_float32.py -q`
Expected: PASS (or skip if no kernel — then verify via Step 5).

- [ ] **Step 5: Run the FULL suite and fix any float64-golden breakage**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`

The dtype change alters on-disk bytes. Any test that reads the detector `image` and compares it to a **float64** reference/golden may now fail at the float32 rounding level (~1e-7 relative). For EACH such failure:
- Confirm the only difference is float32 rounding (the values match the float64 reference within `rtol=1e-6`, not a structural change). A quick way: load both, `np.testing.assert_allclose(stored.astype(np.float64), golden, rtol=1e-6, atol=1e-6)`.
- If so, update that test to compare at float32 tolerance — e.g. change an exact `array_equal`/`assert_allclose(..., rtol=0)` on the detector image to `rtol=1e-6, atol` appropriate to the data scale, OR cast the golden with `.astype(np.float32)` before comparing. Add a one-line comment: `# float32 detector storage (Phase 2a): compare at float32 tolerance`.
- Likely suspects (grep for them): `tests/test_hdf5_bit_equiv.py`, `tests/test_hdf5_run_simulation_end_to_end.py`, `tests/test_io.py`, `tests/test_detector_file.py`, `tests/test_migrate_*`. Do NOT touch the pre-existing xfail `test_forward_output_matches_pickle_era_snapshot` (leave it xfailed).
- Do NOT loosen a tolerance to hide a real >1e-6 discrepancy — if something differs structurally, STOP and report it.

Re-run `pytest -q` until green (same pass/skip/xfail profile as `main`, modulo the new test).

- [ ] **Step 6: Update the output-format doc**

In `docs/output-format.md`, find the detector `image` dataset description and update the stored dtype from float64 to float32, with a short note: "Detector images are stored as float32 (lossless for the simulated intensities; halves file size). The forward model computes in float64." If the file does not mention dtype, add a one-line sentence under the detector-dataset section. (If `docs/output-format.md` does not exist, skip this step and note it.)

- [ ] **Step 7: mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py tests/test_detector_dtype_float32.py docs/output-format.md
git commit -m "Store detector images as float32 (Phase 2a: lossless ~2x slim)"
git show --stat HEAD
```
(If a golden-bearing test was edited in Step 5, `git add` it too and mention it in the message body.)

---

## Task 2: fanout pure helpers — config discovery + worker env

**Files:**
- Create: `scripts/fanout.py` (helpers + dataclass; orchestrator stub filled in Task 3)
- Test: `tests/test_fanout.py`

**Background:** The launcher needs two pure, independently-testable pieces before any subprocess logic: (1) turn a manifest into a list of config paths, (2) build the per-worker environment that caps threads. `DFXM_MAX_WORKERS` is the highest-precedence override consumed by `dfxm_geo.io.images._auto_max_workers` (it sizes the per-scan frame `ThreadPoolExecutor`). BLAS/numba thread vars are pinned to 1 so 8 concurrent processes don't oversubscribe the node (the forward kernel is nogil single-threaded numba; frame parallelism comes from `DFXM_MAX_WORKERS`, not BLAS).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_fanout.py -q`
Expected: FAIL — `scripts/fanout.py` does not exist yet (collection/exec error).

- [ ] **Step 3: Create `scripts/fanout.py` with the helpers**

```python
#!/usr/bin/env python
"""In-node config fan-out launcher for the 100k-image ML dataset.

Runs many forward configs concurrently as separate `dfxm-forward` processes,
each thread-capped, so a big node is used as config-level fan-out (~8 configs x
~16 threads) rather than one over-threaded scan (which plateaus ~16 cores /
memory-bandwidth bound — see the 2026-05-27 cluster profiling note). Phase 2b of
the forward-throughput arc.

Usage:
    python scripts/fanout.py --manifest configs/sweep/ --output /scratch/run \\
        --n-workers 8 --threads-per-worker 16

`--manifest` is a directory of *.toml configs, a single .toml, or a .txt file
listing config paths (one per line; blank lines and #comments ignored).
Each config writes to <output>/<config-stem>/ with its own log at
<output>/<config-stem>.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

# Invoke the forward CLI in a child interpreter. Using `-c` (rather than the
# `dfxm-forward` console script) makes the launcher independent of PATH.
_FORWARD_PREFIX = [
    sys.executable,
    "-c",
    "from dfxm_geo.pipeline import cli_main; raise SystemExit(cli_main())",
]

# Thread-budget env keys pinned to 1 per worker (frame parallelism comes from
# DFXM_MAX_WORKERS; pinning BLAS/numba avoids 8-process oversubscription).
_PINNED_SINGLE = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


@dataclass(frozen=True)
class ConfigResult:
    config: Path
    output_dir: Path
    returncode: int
    wall_s: float
    log_path: Path


def discover_configs(manifest: Path) -> list[Path]:
    """Resolve a manifest to a list of config paths.

    - directory  -> sorted *.toml inside it
    - *.toml file -> [that file]
    - other file (e.g. .txt) -> non-blank, non-# lines as paths, order preserved
    Raises SystemExit if nothing is found.
    """
    manifest = Path(manifest)
    if manifest.is_dir():
        configs = sorted(manifest.glob("*.toml"))
    elif manifest.suffix == ".toml":
        configs = [manifest]
    else:
        configs = []
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                configs.append(Path(line))
    if not configs:
        raise SystemExit(f"No configs found for manifest: {manifest}")
    return configs


def worker_env(threads_per_worker: int, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Per-worker environment: cap frame threads, pin BLAS/numba to 1."""
    env = dict(os.environ if base_env is None else base_env)
    env["DFXM_MAX_WORKERS"] = str(threads_per_worker)
    for key in _PINNED_SINGLE:
        env[key] = "1"
    return env


def _default_runner(config: Path, output_dir: Path, env: Mapping[str, str], log_path: Path) -> int:
    """Run one config via a `dfxm-forward` subprocess; tee output to log_path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _FORWARD_PREFIX + [
        "--config", str(config),
        "--output", str(output_dir),
        "--no-postprocess",
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(env))
    return proc.returncode


Runner = Callable[[Path, Path, "Mapping[str, str]", Path], int]


def run_manifest(
    configs: list[Path],
    output_root: Path,
    *,
    n_workers: int = 8,
    threads_per_worker: int = 16,
    runner: Runner = _default_runner,
    base_env: Mapping[str, str] | None = None,
) -> list[ConfigResult]:
    """Run `configs` concurrently, at most `n_workers` at once.

    Each config gets <output_root>/<stem>/ and <output_root>/<stem>.log. The
    `runner` seam is injected in tests to avoid real subprocesses.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    env = worker_env(threads_per_worker, base_env)

    def task(config: Path) -> ConfigResult:
        out_dir = output_root / config.stem
        log_path = output_root / f"{config.stem}.log"
        t0 = time.perf_counter()
        rc = runner(config, out_dir, env, log_path)
        return ConfigResult(config, out_dir, rc, time.perf_counter() - t0, log_path)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(task, configs))
```

(The orchestrator `run_manifest` + `main()` CLI are completed in Task 3; the helpers above are what Task 2's tests cover. `run_manifest` is included here because Task 3's tests build on it, and Step-1 tests in this task do not call it.)

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_fanout.py -q`
Expected: 5 PASSED (the discovery + env tests).

- [ ] **Step 5: mypy on the script**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy scripts/fanout.py`
Expected: `Success`. If mypy is not configured to scan `scripts/`, that's fine — at minimum it must not error. Fix any type issues in the new code.

- [ ] **Step 6: Commit**

```bash
git add scripts/fanout.py tests/test_fanout.py
git commit -m "Add fanout config discovery + worker-env helpers (Phase 2b)"
git show --stat HEAD
```

---

## Task 3: fanout orchestrator + CLI + integration test

**Files:**
- Modify: `scripts/fanout.py` (add `main()` CLI; `run_manifest` already added in Task 2)
- Modify: `tests/test_fanout.py` (add orchestration unit test + one real integration test)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fanout.py`)

```python
import threading
from pathlib import Path

import dfxm_geo.direct_space.forward_model as fm


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
    # Per-config output dir layout: <root>/<stem>/
    assert {r.output_dir.name for r in results} == {f"c{i}" for i in range(6)}
    # Env cap propagated to every worker.
    assert all(e["DFXM_MAX_WORKERS"] == "7" for e in seen_envs)


def test_run_manifest_reports_nonzero_returncode(tmp_path: Path) -> None:
    c = tmp_path / "bad.toml"
    c.write_text("")

    def failing_runner(config, output_dir, env, log_path):
        return 3

    results = fanout.run_manifest([c], tmp_path / "out", n_workers=1, runner=failing_runner)
    assert results[0].returncode == 3


def test_fanout_end_to_end_runs_two_configs(tmp_path: Path) -> None:
    """Real subprocess integration: two tiny centered configs via the actual
    dfxm-forward CLI, thread-capped, each producing a valid float32 HDF5."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")) or fm.Hg is None:
        import pytest
        pytest.skip("No bootstrapped kernel npz found; skipping integration run.")

    import h5py
    import numpy as np

    # Two tiny centered configs (2-step phi scan, no postprocess, no perfect crystal).
    cfg_text = (
        '[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n'
        '[scan.phi]\nrange = 0.001\nsteps = 2\n\n'
        '[crystal]\nmode = "centered"\n[crystal.centered]\n\n'
        '[io]\ninclude_perfect_crystal = false\n\n'
        '[postprocess]\nenabled = false\n'
    )
    manifest_dir = tmp_path / "configs"
    manifest_dir.mkdir()
    for name in ("alpha.toml", "beta.toml"):
        (manifest_dir / name).write_text(cfg_text)

    out_root = tmp_path / "out"
    results = fanout.run_manifest(
        fanout.discover_configs(manifest_dir),
        out_root,
        n_workers=2,
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
```

Note on the integration test: `scan.phi.range` is in RADIANS (scan ranges are radians end-to-end). The forward CLI writes the v1.2.0 master+per-scan layout, so the detector file is at `<output>/scan0001/dfxm_sim_detector_0000.h5`. If the centered config needs additional required fields (it should not — post-v2.0.0 an empty TOML is a valid 1-image run), read `tests/test_empty_toml_runs.py` for the minimal valid config and adjust `cfg_text`.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_fanout.py -q`
Expected: the two `run_manifest` unit tests PASS already (run_manifest was added in Task 2); the integration test runs (or skips if no kernel). If `main` (CLI) is referenced anywhere it would fail — but these tests call `run_manifest`/`discover_configs` directly, so they should pass once Task 2 landed. The point of this task is the integration coverage + the `main()` CLI below.

- [ ] **Step 3: Add the `main()` CLI to `scripts/fanout.py`** (append at end)

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Directory of *.toml, a single .toml, or a .txt list of config paths.")
    ap.add_argument("--output", type=Path, required=True, help="Output root directory.")
    ap.add_argument("--n-workers", type=int, default=8, help="Concurrent config processes.")
    ap.add_argument("--threads-per-worker", type=int, default=16,
                    help="DFXM_MAX_WORKERS per config (frame thread pool size).")
    args = ap.parse_args(argv)

    configs = discover_configs(args.manifest)
    print(f"fanout: {len(configs)} configs, {args.n_workers} workers x "
          f"{args.threads_per_worker} threads -> {args.output}")
    t0 = time.perf_counter()
    results = run_manifest(
        configs, args.output,
        n_workers=args.n_workers, threads_per_worker=args.threads_per_worker,
    )
    wall = time.perf_counter() - t0

    n_fail = sum(1 for r in results if r.returncode != 0)
    for r in results:
        status = "ok" if r.returncode == 0 else f"FAIL(rc={r.returncode})"
        print(f"  {r.config.name:<30} {status:>12}  {r.wall_s:7.1f}s  log={r.log_path}")
    print(f"fanout done: {len(results) - n_fail}/{len(results)} ok in {wall:.1f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full fanout test file**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_fanout.py -q`
Expected: all PASS (integration test may SKIP if no kernel). If the integration test fails with a config error, inspect the failing `log_path` content (the test prints logs on failure) and fix `cfg_text` per `tests/test_empty_toml_runs.py`.

- [ ] **Step 5: Smoke-run the CLI end to end** (only if a kernel is bootstrapped)

```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/fanout.py --manifest <a dir with 2 tiny tomls> --output <tmp> --n-workers 2 --threads-per-worker 2
```
Expected: prints a per-config table and "2/2 ok"; exit code 0. (If no kernel, skip this manual smoke and rely on the integration test's skip.)

- [ ] **Step 6: mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy scripts/fanout.py`
Expected: `Success` / no errors.

- [ ] **Step 7: Commit**

```bash
git add scripts/fanout.py tests/test_fanout.py
git commit -m "Add fanout orchestrator + CLI + integration test (Phase 2b)"
git show --stat HEAD
```

---

## Task 4: LSF batch template + docs

**Files:**
- Create: `lsf/fanout.bsub`
- Modify: `docs/cluster-profiling.md` (or create `docs/cluster-fanout.md` if a separate doc is cleaner)

**Background:** Mirror `lsf/profile.bsub` (conda activation, kernel bootstrap, node-info echo). The job submits ONE bsub for the whole batch and calls `scripts/fanout.py`.

- [ ] **Step 1: Create `lsf/fanout.bsub`**

```bash
#!/bin/bash
# ==============================================================================
# DTU HPC (LSF) — config fan-out batch for the ML-data pipeline (Phase 2b)
# ==============================================================================
# Submit with:   bsub < lsf/fanout.bsub
# Monitor with:  bjobs -l <JOBID>
# Edit MANIFEST + the BSUB resource lines for your node + sweep.
# ==============================================================================
#
# >>> EDIT THESE >>>
#BSUB -J dfxm-fanout
#BSUB -q hpc
#BSUB -W 04:00                          # Walltime HH:MM (size to your sweep)
#BSUB -n 128                            # full node; query `bhosts hpc` first
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"                # single host — in-node fan-out
#BSUB -o logs/fanout-%J.out
#BSUB -e logs/fanout-%J.err
MANIFEST="configs/sweep"                # dir of *.toml, a .toml, or a .txt list
N_WORKERS=8                             # concurrent config processes
THREADS_PER_WORKER=16                   # DFXM_MAX_WORKERS per config
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
OUTROOT="${WORKDIR}/output/fanout_${LSB_JOBID:-local}"
mkdir -p "${OUTROOT}" logs

echo "===== NODE INFO ====="
echo "host: $(hostname)"
nproc || true
echo "manifest: ${MANIFEST}  workers: ${N_WORKERS} x ${THREADS_PER_WORKER} threads"
echo "====================="

# --- conda activation (mirrors lsf/profile.bsub) ---
CONDA_BASE="${CONDA_BASE:-$(dirname "$(dirname "$(command -v conda 2>/dev/null || true)")")}"
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    for cand in "${HOME}/miniforge3" "${HOME}/miniconda3"; do
        [ -f "${cand}/etc/profile.d/conda.sh" ] && CONDA_BASE="${cand}" && break
    done
fi
if [ ! -f "${CONDA_BASE:-}/etc/profile.d/conda.sh" ]; then
    echo "ERROR: could not locate a conda install. Set CONDA_BASE explicitly above." >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dfxm-geo

# Stage 0 — ensure the resolution kernel exists (idempotent).
dfxm-bootstrap --if-missing --config "${WORKDIR}/configs/profile_rocking.toml"

# Stage 1 — fan out the configs across the node.
python scripts/fanout.py \
    --manifest "${MANIFEST}" \
    --output "${OUTROOT}" \
    --n-workers "${N_WORKERS}" \
    --threads-per-worker "${THREADS_PER_WORKER}"

echo "Done. Per-config outputs under ${OUTROOT}"
```

- [ ] **Step 2: Verify the template's tracked-file expectations**

There is a test `tests/test_cluster_templates.py` that checks the LSF templates. Run it to confirm the new template doesn't break template assertions and (if the test scans all of `lsf/`) that `fanout.bsub` fits the expected shape:
Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_cluster_templates.py -q`
Expected: PASS. If the test enumerates expected templates by name and now fails because it doesn't know about `fanout.bsub`, read the test and add `fanout.bsub` to its expected set (only if the test is name-driven; if it globs `lsf/*.bsub` and checks shared properties, ensure `fanout.bsub` satisfies them — e.g. shebang, `#BSUB -J`, conda activation block).

- [ ] **Step 3: Add a short docs section**

In `docs/cluster-profiling.md` (it exists per `lsf/profile.bsub`'s reference), add a "## Fan-out batch (Phase 2b)" section: how to build a manifest (a directory of per-config TOMLs, or generate with `scripts/scaling_sweep.py`-style sweeps), how to submit (`bsub < lsf/fanout.bsub`), the `--n-workers` × `--threads-per-worker` guidance (~8×16 on a 128-core node; total threads ≈ workers×threads should ≈ node cores), and where outputs land. Keep it ~15 lines. If `docs/cluster-profiling.md` does not exist, create `docs/cluster-fanout.md` with the same content.

- [ ] **Step 4: Commit**

```bash
git add lsf/fanout.bsub docs/cluster-profiling.md tests/test_cluster_templates.py
git commit -m "Add LSF fanout batch template + cluster docs (Phase 2b)"
git show --stat HEAD
```
(Only `git add` `tests/test_cluster_templates.py` / `docs/cluster-fanout.md` if you actually modified/created them.)

---

## Task 5: Full suite + mypy + storage re-check

**Files:**
- Modify: `docs/superpowers/notes/2026-05-27-find-hg-baseline.md` (append a Phase-2 storage note) — or create `docs/superpowers/notes/2026-05-27-phase2-fanout.md`

- [ ] **Step 1: Full suite**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: green at the same pass/skip/xfail profile as `main` plus the new `test_detector_dtype_float32` and `test_fanout` tests (integration test may skip without a kernel). No unexpected failures.

- [ ] **Step 2: mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found`.

- [ ] **Step 3: Re-check the storage extrapolation**

Run `scripts/scaling_sweep.py` on a profiling config to measure the NEW per-config on-disk size (now float32) and re-extrapolate to 100k images:
Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/scaling_sweep.py --config configs/profile_rocking.toml --output C:\Users\borgi\tmp\phase2_sweep --repeats 1 --max-workers 4`
(`--max-workers 4` keeps the laptop run quick; we only need the per-config MB + extrapolated TB, not the full scaling curve.) If no kernel is bootstrapped, bootstrap first or note that this step must run on a machine with the kernel.
Record the new "per-config on disk MB" and "total storage TB" vs the pre-float32 baseline (~0.51 TB / 106 MB-per-config from the cluster note). Expect ~half.

- [ ] **Step 4: Write the Phase-2 note**

Append a "## Phase 2 (float32 + fan-out)" section to `docs/superpowers/notes/2026-05-27-find-hg-baseline.md` (or a new note file): the measured per-config MB before/after float32, the re-extrapolated TB for 100k, and a one-line confirmation that `fanout.py` runs N configs concurrently with bounded threads (cite the integration test). Note the deliverables: `scripts/fanout.py`, `lsf/fanout.bsub`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/
git commit -m "Phase 2 verification: float32 storage re-check + fanout note"
git show --stat HEAD
```

- [ ] **Step 6: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill. Per CLAUDE.md: confirm with the user before pushing/PR; the established pattern is a `--no-ff` merge to `main` after local verification. This completes the two-phase forward-throughput arc — at wrap-up, update `docs/superpowers/notes/`, write/refresh the session-handoff memory, and flip CLAUDE.md's "NEXT" note (the arc is done; decide whether a version bump/PyPI release is warranted now that Phase 1 + Phase 2 are both in).

---

## Self-review notes

- **Spec coverage:** Phase 2a (float32 detector dtype, lossless, ~2×) → Task 1. Phase 2b (in-node launcher: ~8 processes × ~16 threads, thread-budget knob via `DFXM_MAX_WORKERS` + pinned BLAS, config manifest, one bsub, local small-scale validation) → Tasks 2 (helpers), 3 (orchestrator + CLI + integration), 4 (LSF template + docs). Storage/throughput re-check → Task 5. The spec's "tie into scaling_sweep.py" is honored via the manifest being a directory of TOMLs (sweep-generated) and the Task-5 storage re-check reusing `scaling_sweep.py`.
- **Type consistency:** `discover_configs(manifest) -> list[Path]`, `worker_env(threads_per_worker, base_env) -> dict[str,str]`, `run_manifest(configs, output_root, *, n_workers, threads_per_worker, runner, base_env) -> list[ConfigResult]`, and the `ConfigResult` fields (`config, output_dir, returncode, wall_s, log_path`) are used identically across Tasks 2 and 3. The `Runner` callable signature matches `_default_runner` and the test fakes.
- **Decoupling:** Tasks 1 (float32) and 2-4 (fanout) are independent; only Task 5 depends on both. float32 lands first so the storage re-check in Task 5 reflects it.
- **Placeholder scan:** every code step has complete code; the only conditional branches ("if docs/output-format.md doesn't exist", "if cluster_templates test is name-driven") give explicit fallback instructions rather than leaving TODOs.
- **Known risk flagged:** Task 1 Step 5 explicitly handles float64-golden test breakage from the dtype change — the one non-obvious consequence — with a bounded fix (compare at float32 tolerance, never hide >1e-6 drift).
