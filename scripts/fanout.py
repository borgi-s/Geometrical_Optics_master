#!/usr/bin/env python
"""In-node config fan-out launcher for the 100k-image ML dataset.

Runs many forward configs concurrently via a persistent worker pool (v2.6.0
W1 default) or, with ``--isolate``, as separate ``dfxm-forward`` subprocesses.
Pool mode amortizes the ~47%-of-wall per-config import/JIT/kernel-load cost:
workers import ``dfxm_geo`` once and reuse the kernel LUT across configs,
so a big node is used as config-level fan-out (~8 configs x ~16 threads).

Usage:
    # default: persistent pool (amortizes import/JIT; recommended for sweeps)
    python scripts/fanout.py --manifest configs/sweep/ --output /scratch/run \\
        --n-workers 8 --threads-per-worker 16

    # --isolate: one subprocess per config (pre-v2.6.0 behavior; hard isolation)
    python scripts/fanout.py --manifest configs/sweep/ --output /scratch/run \\
        --n-workers 8 --threads-per-worker 16 --isolate

``--manifest`` is a directory of *.toml configs, a single .toml, or a .txt file
listing config paths (one per line; blank lines and #comments ignored).
Each config writes to <output>/<config-stem>/ with its own log at
<output>/<config-stem>.log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

# Invoke the forward/identify CLIs in a child interpreter. Using `-c` (rather
# than the `dfxm-forward` / `dfxm-identify` console scripts) makes the launcher
# independent of PATH — the child always uses the same interpreter as the parent.
# The snippet times the import (interpreter + module import = the fixed startup
# cost a persistent worker pool would amortize) separately from the CLI run and
# prints one machine-readable DFXM_TIMING line, which lands in the per-config
# log and is harvested by --timing-json.
_CHILD_SNIPPET = (
    "import json, time\n"
    "_t = time.perf_counter()\n"
    "from dfxm_geo.pipeline import {entry}\n"
    "_import_s = time.perf_counter() - _t\n"
    "_t = time.perf_counter()\n"
    "_rc = {entry}()\n"
    "_run_s = time.perf_counter() - _t\n"
    "print('DFXM_TIMING ' + json.dumps("
    "{{'import_s': round(_import_s, 3), 'run_s': round(_run_s, 3)}}))\n"
    "raise SystemExit(_rc)\n"
)
_FORWARD_PREFIX = [sys.executable, "-c", _CHILD_SNIPPET.format(entry="cli_main")]
_IDENTIFY_PREFIX = [sys.executable, "-c", _CHILD_SNIPPET.format(entry="cli_main_identify")]

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


def worker_env(
    threads_per_worker: int, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Per-worker environment: cap frame threads, pin BLAS/numba to 1."""
    env = dict(os.environ if base_env is None else base_env)
    env["DFXM_MAX_WORKERS"] = str(threads_per_worker)
    for key in _PINNED_SINGLE:
        env[key] = "1"
    return env


@contextlib.contextmanager
def _pinned_environ(threads_per_worker: int):
    """Pin worker env keys in os.environ for the pool's lifetime, restore after.

    Pool workers inherit the env at process creation (fork on Linux/LSF,
    spawn on Windows), so the keys must be set in the PARENT before the
    executor spawns its first worker — and restored afterwards so the
    calling process (tests, notebooks) is not permanently mutated.

    Note: os.environ mutation is process-global — do not call run_pool
    concurrently from multiple threads of one process (separate processes,
    e.g. pytest-xdist workers, are fine).
    """
    target = worker_env(threads_per_worker)
    keys = ["DFXM_MAX_WORKERS", *_PINNED_SINGLE]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = target[k]
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_pool(
    configs: list[Path],
    output_root: Path,
    *,
    n_workers: int = 8,
    threads_per_worker: int = 16,
    mode: str = "forward",
    worker_fn: Callable[..., dict] | None = None,
    executor_factory: Callable[[int], ProcessPoolExecutor | ThreadPoolExecutor] | None = None,
    max_attempts: int = 2,
) -> list[ConfigResult]:
    """Run `configs` on a persistent worker pool (v2.6.0 W1, the default mode).

    Workers import dfxm_geo once, JIT once, and retain the kernel LUT
    (`pipeline._KERNEL_CTX_CACHE`) across configs — amortizing the fixed
    ~47%-of-wall per-subprocess cost the old isolate mode pays every config.

    Failure containment:
    - worker_fn exceptions and rc!=0 dicts -> per-config failed result;
    - a hard worker death (`BrokenProcessPool`) aborts the current executor:
      finished results are kept, undone configs are resubmitted on a fresh
      executor; a config still broken after `max_attempts` is rc=-2.

    `worker_fn` / `executor_factory` are test seams; production uses
    `dfxm_geo.fanout_worker.run_one` on a `ProcessPoolExecutor`.
    """
    if worker_fn is None:
        from dfxm_geo import fanout_worker

        worker_fn = fanout_worker.run_one
    if executor_factory is None:
        executor_factory = lambda n: ProcessPoolExecutor(max_workers=n)  # noqa: E731

    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[Path, ConfigResult] = {}
    attempts: dict[Path, int] = dict.fromkeys(configs, 0)
    pending = list(configs)

    def _record(config: Path, rc: int, wall_s: float) -> None:
        results[config] = ConfigResult(
            config,
            output_root / config.stem,
            rc,
            wall_s,
            output_root / f"{config.stem}.log",
        )

    with _pinned_environ(threads_per_worker):
        while pending:
            batch, pending = pending, []
            broken: list[Path] = []
            with executor_factory(n_workers) as ex:
                futs: dict[Future, Path] = {}
                for c in batch:
                    attempts[c] += 1
                    out_dir = output_root / c.stem
                    log_path = output_root / f"{c.stem}.log"
                    futs[ex.submit(worker_fn, mode, str(c), str(out_dir), str(log_path))] = c
                not_done: set[Future] = set(futs)
                # When a real ProcessPoolExecutor breaks, terminate_broken()
                # poisons ALL outstanding futures at once — every fut.result()
                # raises BrokenProcessPool, not just the config that killed the
                # worker. The loop below drains them all; unstarted configs
                # land in broken[] alongside the actual killer and retry as a
                # group on the fresh executor.
                while not_done:
                    done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
                    for fut in done:
                        c = futs[fut]
                        try:
                            payload = fut.result()
                            _record(
                                c,
                                int(payload["returncode"]),
                                float(payload.get("wall_s", 0.0)),
                            )
                        except BrokenProcessPool:
                            broken.append(c)
                        except Exception:
                            # worker_fn raised (run_one's guard normally
                            # prevents this) — failed config, not a dead pool.
                            log_path = output_root / f"{c.stem}.log"
                            log_path.write_text(
                                f"fanout: pool worker raised for {c}\n\n{traceback.format_exc()}",
                                encoding="utf-8",
                            )
                            _record(c, -1, 0.0)
            for c in broken:
                if attempts[c] < max_attempts:
                    pending.append(c)
                else:
                    log_path = output_root / f"{c.stem}.log"
                    log_path.write_text(
                        f"fanout: worker pool died twice running {c}; giving up "
                        f"(run with --isolate to debug this config)\n",
                        encoding="utf-8",
                    )
                    _record(c, -2, 0.0)

    return [results[c] for c in configs]


def build_cmd(mode: str, config: Path, output_dir: Path) -> list[str]:
    """Build the child-process command list for one config.

    Args:
        mode: ``"forward"`` or ``"identify"``.
        config: Path to the TOML config file.
        output_dir: Per-config output directory.

    Returns:
        A ``list[str]`` ready to pass to ``subprocess.run``.

    ``--no-postprocess`` is a forward-only flag — it is appended only in
    forward mode. Identify mode omits it.
    """
    if mode == "forward":
        prefix = _FORWARD_PREFIX
        extra = ["--no-postprocess"]
    elif mode == "identify":
        prefix = _IDENTIFY_PREFIX
        extra = []
    else:
        raise ValueError(f"mode must be 'forward' or 'identify', got {mode!r}")
    return prefix + ["--config", str(config), "--output", str(output_dir)] + extra


def _default_runner(
    config: Path,
    output_dir: Path,
    env: Mapping[str, str],
    log_path: Path,
    *,
    mode: str = "forward",
) -> int:
    """Run one config via a child `dfxm-forward` or `dfxm-identify` subprocess.

    Tees stdout+stderr to *log_path*. The *mode* keyword selects which CLI is
    called and whether ``--no-postprocess`` is appended (forward-only flag).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(mode, config, output_dir)
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
    runner: Runner | None = None,
    base_env: Mapping[str, str] | None = None,
    mode: str = "forward",
    isolate: bool = False,
) -> list[ConfigResult]:
    """Run `configs` concurrently, at most `n_workers` at once.

    Each config gets <output_root>/<stem>/ and <output_root>/<stem>.log. The
    `runner` seam is injected in tests to avoid real subprocesses; when
    *runner* is ``None`` (the default), ``_default_runner`` is called with the
    *mode* captured in a closure so the correct child CLI is selected without
    changing the ``Runner`` type signature.

    *mode* selects the child CLI:
      ``"forward"``   (default) → ``dfxm-forward`` + ``--no-postprocess``
      ``"identify"``            → ``dfxm-identify`` (no ``--no-postprocess``)

    *isolate* (v2.6.0): when True, run each config in a fresh subprocess
    (pre-v2.6.0 behavior) instead of the persistent worker pool. An injected
    *runner* implies isolate — the runner seam IS the subprocess abstraction.
    Pool mode (the default) amortizes the fixed ~47% per-config import/JIT
    cost across the whole sweep.
    """
    # Pool is the default (v2.6.0). The subprocess-per-config path remains
    # behind --isolate; an injected `runner` implies it (the runner seam IS
    # the subprocess abstraction, and all pre-v2.6.0 tests use it).
    if runner is None and not isolate:
        return run_pool(
            configs,
            output_root,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            mode=mode,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    env = worker_env(threads_per_worker, base_env)

    # When no runner is injected, bind *mode* via a closure so _default_runner
    # receives it without widening the Runner type alias.
    effective_runner: Runner
    if runner is None:
        _mode = mode

        def effective_runner(
            config: Path,
            output_dir: Path,
            _env: Mapping[str, str],
            log_path: Path,
        ) -> int:
            return _default_runner(config, output_dir, _env, log_path, mode=_mode)

    else:
        effective_runner = runner

    def task(config: Path) -> ConfigResult:
        out_dir = output_root / config.stem
        log_path = output_root / f"{config.stem}.log"
        t0 = time.perf_counter()
        # Batch resilience: a config that fails to even spawn (or any runner
        # exception) is recorded as a failed result with rc=-1 rather than
        # aborting the whole sweep — one bad config shouldn't kill a 100k run.
        try:
            rc = effective_runner(config, out_dir, env, log_path)
        except Exception:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"fanout: runner raised for {config}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            rc = -1
        return ConfigResult(config, out_dir, rc, time.perf_counter() - t0, log_path)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(task, configs))


_TIMING_PREFIX = "DFXM_TIMING "
# The three identify CLIs report different units on their final line; each is
# kept under its own key so a scene/configuration count is never summed as an
# image count: single prints "Wrote N images", multi "Wrote N samples"
# (scenes, each a full rocking scan of frames), z-scan "Wrote N configurations"
# (z-layer x b x angle combos, each a phi x chi grid).
_COUNT_RES = {
    "images": re.compile(r"^Wrote (\d+) images", re.MULTILINE),
    "samples": re.compile(r"^Wrote (\d+) samples", re.MULTILINE),
    "configurations": re.compile(r"^Wrote (\d+) configurations", re.MULTILINE),
}


def parse_timing_log(log_path: Path) -> dict[str, float | int]:
    """Harvest child-emitted timing facts from one per-config log.

    Returns a (possibly empty) dict with whichever of these were found:
    ``import_s`` / ``run_s`` from the last ``DFXM_TIMING {json}`` line
    (``import_s`` covers the dfxm_geo module import only — interpreter launch
    happens before the child's timer starts; the launcher-side ``wall_s``
    minus ``import_s + run_s`` is that spawn overhead), and the CLI's final
    count line (``images`` / ``samples`` / ``configurations`` depending on
    mode). Missing files, absent lines, or garbled json all degrade to ``{}``
    — a config that crashed before printing timing must not break the sweep
    summary.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: dict[str, float | int] = {}
    timing_lines = [line for line in text.splitlines() if line.startswith(_TIMING_PREFIX)]
    if timing_lines:
        with contextlib.suppress(json.JSONDecodeError):
            out.update(json.loads(timing_lines[-1][len(_TIMING_PREFIX) :]))
    for key, regex in _COUNT_RES.items():
        counts = regex.findall(text)
        if counts:
            out[key] = int(counts[-1])
    return out


def write_timing_json(
    path: Path,
    results: list[ConfigResult],
    *,
    total_wall_s: float,
    n_workers: int,
    threads_per_worker: int,
    mode: str,
    isolate: bool = False,
) -> None:
    """Write the sweep timing manifest: per-config rows + throughput summary.

    Sweep-level ``images_total`` / ``images_per_s`` are only emitted when every
    successful config reported an image count — a partial sum would understate
    throughput and look like a regression. ``configs_per_hour`` counts ALL
    configs (failures included): it measures sweep progress rate, not useful
    yield — read it together with ``n_ok``.

    Returncode legend in the per-config rows: 0 = ok; >0 = the config's CLI
    exit code; -1 = worker exception (traceback in the log); -2 = pool-mode
    worker death after max retries (re-run that config with ``--isolate``).

    *isolate*: when False (the default, pool mode), ``import_s`` in each
    per-config row is the per-worker amortized cost — real on the worker's
    first config, 0.0 on subsequent warm ones (the import/JIT/kernel-load
    is paid once per worker, not once per config). ``wall_s`` in pool mode
    excludes queue-wait time between config submissions. When True (--isolate
    subprocess mode), every config pays the full import cost.
    """
    rows = []
    image_counts: list[int] = []
    for r in results:
        row: dict[str, object] = {
            "config": str(r.config),
            "output_dir": str(r.output_dir),
            "returncode": r.returncode,
            "wall_s": round(r.wall_s, 3),
            "log": str(r.log_path),
        }
        timing = parse_timing_log(r.log_path)
        row.update(timing)
        rows.append(row)
        if r.returncode == 0 and "images" in timing:
            image_counts.append(int(timing["images"]))

    n_ok = sum(1 for r in results if r.returncode == 0)
    sweep: dict[str, object] = {
        "mode": mode,
        "isolate": isolate,
        "n_configs": len(results),
        "n_ok": n_ok,
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "total_wall_s": round(total_wall_s, 3),
        "configs_per_hour": round(len(results) / total_wall_s * 3600, 2)
        if total_wall_s > 0
        else None,
    }
    if image_counts and len(image_counts) == n_ok and n_ok > 0:
        sweep["images_total"] = sum(image_counts)
        sweep["images_per_s"] = round(sum(image_counts) / total_wall_s, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sweep": sweep, "configs": rows}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Directory of *.toml, a single .toml, or a .txt list of config paths.",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output root directory.")
    ap.add_argument("--n-workers", type=int, default=8, help="Concurrent config processes.")
    ap.add_argument(
        "--threads-per-worker",
        type=int,
        default=16,
        help="DFXM_MAX_WORKERS per config (frame thread pool size).",
    )
    ap.add_argument(
        "--mode",
        choices=["forward", "identify"],
        default="forward",
        help=(
            "Child CLI to invoke for each config. "
            "'forward' (default) runs `dfxm-forward` with --no-postprocess; "
            "'identify' runs `dfxm-identify` (no --no-postprocess). "
            "Set MODE=identify in lsf/fanout.bsub to fan out identification."
        ),
    )
    ap.add_argument(
        "--timing-json",
        type=Path,
        default=None,
        help=(
            "Write a JSON timing manifest here: per-config wall time + the "
            "child-reported import/run split (DFXM_TIMING log lines) + sweep "
            "throughput (configs/hour, images/sec). M1 Phase 2a baseline data."
        ),
    )
    ap.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Run each config in a fresh subprocess (pre-v2.6.0 behavior) "
            "instead of the persistent worker pool. Slower (re-pays import/"
            "JIT/kernel-load per config) but gives hard crash isolation — "
            "use it to debug a config that kills pool workers."
        ),
    )
    args = ap.parse_args(argv)

    configs = discover_configs(args.manifest)
    run_label = f"{args.mode}, isolate" if args.isolate else f"{args.mode}, pool"
    print(
        f"fanout: {len(configs)} configs, {args.n_workers} workers x "
        f"{args.threads_per_worker} threads [{run_label}] -> {args.output}"
    )
    t0 = time.perf_counter()
    results = run_manifest(
        configs,
        args.output,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        mode=args.mode,
        isolate=args.isolate,
    )
    wall = time.perf_counter() - t0

    n_fail = sum(1 for r in results if r.returncode != 0)
    for r in results:
        status = "ok" if r.returncode == 0 else f"FAIL(rc={r.returncode})"
        print(f"  {r.config.name:<30} {status:>12}  {r.wall_s:7.1f}s  log={r.log_path}")
    print(f"fanout done: {len(results) - n_fail}/{len(results)} ok in {wall:.1f}s")
    if args.timing_json is not None:
        write_timing_json(
            args.timing_json,
            results,
            total_wall_s=wall,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            mode=args.mode,
            isolate=args.isolate,
        )
        print(f"fanout: timing manifest -> {args.timing_json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
