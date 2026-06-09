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
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Invoke the forward/identify CLIs in a child interpreter. Using `-c` (rather
# than the `dfxm-forward` / `dfxm-identify` console scripts) makes the launcher
# independent of PATH — the child always uses the same interpreter as the parent.
_FORWARD_PREFIX = [
    sys.executable,
    "-c",
    "from dfxm_geo.pipeline import cli_main; raise SystemExit(cli_main())",
]
_IDENTIFY_PREFIX = [
    sys.executable,
    "-c",
    "from dfxm_geo.pipeline import cli_main_identify; raise SystemExit(cli_main_identify())",
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


def worker_env(
    threads_per_worker: int, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Per-worker environment: cap frame threads, pin BLAS/numba to 1."""
    env = dict(os.environ if base_env is None else base_env)
    env["DFXM_MAX_WORKERS"] = str(threads_per_worker)
    for key in _PINNED_SINGLE:
        env[key] = "1"
    return env


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
    """
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
    args = ap.parse_args(argv)

    configs = discover_configs(args.manifest)
    print(
        f"fanout: {len(configs)} configs, {args.n_workers} workers x "
        f"{args.threads_per_worker} threads [{args.mode}] -> {args.output}"
    )
    t0 = time.perf_counter()
    results = run_manifest(
        configs,
        args.output,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        mode=args.mode,
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
