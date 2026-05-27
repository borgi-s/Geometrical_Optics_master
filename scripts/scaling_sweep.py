#!/usr/bin/env python
"""Worker-count scaling sweep + 100k-image extrapolation for one rocking scan.

Two measurements:

  (A) Pure-forward scaling. Mirrors write_simulation_h5's worker pattern:
      build base_qc = precompute_forward_static(Hg) ONCE (shared read-only),
      then run all frames through forward_from_static under a
      ThreadPoolExecutor at workers = 1,2,4,...,ncores. forward_from_static's
      numba kernel is nogil, so threads parallelize. No HDF5, no config-gen in
      the timed region -> a clean speedup curve that tests whether the laptop's
      ~17x threaded-scan result holds on the node.

  (B) Realistic per-config cost. One full run_simulation at the best worker
      count, measuring wall time AND the on-disk output size. Extrapolated to
      100k images: configs = ceil(100000 / n_frames), node-hours, storage TB.

No cProfile here: it serializes/distorts threaded numba and would invalidate
the scaling curve.

Usage:
    python scripts/scaling_sweep.py --config configs/profile_rocking.toml \
        --output /tmp/sweep --repeats 3
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    SimulationConfig,
    _build_scan_frames,
    _load_resolution,
    run_simulation,
)

TARGET_IMAGES = 100_000


def usable_cores() -> int:
    """Cores actually available to this process (respects cgroup/affinity)."""
    try:
        return len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        return os.cpu_count() or 1  # Windows/macOS fallback


def worker_grid(ncores: int) -> list[int]:
    """[1, 2, 4, 8, ...] up to and including ncores, deduped + sorted."""
    grid = [1]
    w = 2
    while w < ncores:
        grid.append(w)
        w *= 2
    grid.append(ncores)
    return sorted(set(grid))


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_forward_scan(base_qc, frames, workers: int) -> float:
    """Wall-clock seconds to render all frames at the given worker count."""
    phi = frames.phi_pf
    chi = frames.chi_pf
    twodt = frames.two_dtheta_pf

    def render(i: int):
        return fm.forward_from_static(base_qc, float(phi[i]), float(chi[i]), float(twodt[i]))

    t0 = time.perf_counter()
    if workers == 1:
        for i in range(frames.n_frames):
            render(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(render, range(frames.n_frames)))
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3, help="best-of-N per worker count")
    ap.add_argument("--max-workers", type=int, default=None, help="cap the sweep")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.from_toml(args.config)

    ncores = usable_cores()
    if args.max_workers is not None:
        ncores = min(ncores, args.max_workers)
    grid = worker_grid(ncores)

    # --- Setup for (A): load kernel, build population, Hg, base_qc, frames. ---
    _load_resolution(config.reciprocal)
    fov_lateral_um = fm.Npixels * fm.psize * 1e6
    population = fm.build_dislocation_population(
        config.crystal, fov_lateral_um=fov_lateral_um, rng=None
    )
    if config.crystal.mode == "wall":
        raise SystemExit("scaling_sweep expects centered/random_dislocations configs")
    Hg, q_hkl = fm.Find_Hg_from_population(
        population,
        h=config.reciprocal.hkl[0],
        k=config.reciprocal.hkl[1],
        l=config.reciprocal.hkl[2],
    )
    base_qc = fm.precompute_forward_static(Hg)
    frames = _build_scan_frames(config.scan)
    n_rays = base_qc.shape[1]

    # JIT pre-warm (compile cost excluded from the timed sweep).
    t0 = time.perf_counter()
    fm.forward_from_static(
        base_qc, float(frames.phi_pf[0]), float(frames.chi_pf[0]), float(frames.two_dtheta_pf[0])
    )
    warm_s = time.perf_counter() - t0

    lines: list[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    emit(f"config         : {args.config}")
    emit(f"usable cores   : {ncores}")
    emit(f"frames/scan    : {frames.n_frames}")
    emit(f"rays/frame     : {n_rays}")
    emit(f"Npixels        : {fm.Npixels}   Nsub: {fm.Nsub}")
    emit(f"JIT warmup     : {warm_s:.3f} s (per-fresh-process compile)")
    emit("")
    emit("=== (A) pure-forward scaling (no HDF5, no config-gen) ===")
    emit(f"{'workers':>8} {'wall_s':>10} {'speedup':>9} {'frames/s':>10}")

    base_wall = None
    for w in grid:
        best = min(run_forward_scan(base_qc, frames, w) for _ in range(args.repeats))
        if base_wall is None:
            base_wall = best
        speedup = base_wall / best
        fps = frames.n_frames / best
        emit(f"{w:>8} {best:>10.3f} {speedup:>9.2f} {fps:>10.1f}")
    best_workers = grid[-1]

    # --- (B) realistic per-config cost: full run_simulation at best_workers. ---
    emit("")
    emit("=== (B) realistic per-config cost (full run_simulation + HDF5) ===")
    config.io.max_workers = best_workers
    run_dir = args.output / "one_config"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    t0 = time.perf_counter()
    run_simulation(config, run_dir)
    per_config_s = time.perf_counter() - t0
    out_bytes = _dir_bytes(run_dir)

    configs_needed = math.ceil(TARGET_IMAGES / frames.n_frames)
    node_seconds = configs_needed * per_config_s
    storage_tb = configs_needed * out_bytes / 1e12

    emit(f"workers            : {best_workers}")
    emit(f"per-config wall    : {per_config_s:.3f} s  ({frames.n_frames} frames)")
    emit(f"per-config on disk : {out_bytes / 1e6:.2f} MB")
    emit("")
    emit(f"=== extrapolation to {TARGET_IMAGES:,} images ===")
    emit(f"configs needed     : {configs_needed:,}  (= {TARGET_IMAGES:,} / {frames.n_frames})")
    emit(f"single-node time   : {node_seconds / 3600:.1f} node-hours")
    emit(f"total storage      : {storage_tb:.2f} TB")

    (args.output / "scaling_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.output / 'scaling_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
