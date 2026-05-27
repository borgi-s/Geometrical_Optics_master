#!/usr/bin/env python
"""Single-thread cProfile of one rocking-scan forward run.

Why single-thread: cProfile cleanly attributes only the main thread, and the
forward kernel (_mc_lut_forward) is @njit(nogil) parallelized across frames by
a ThreadPoolExecutor. Profiling multi-threaded would mis-attribute worker time.
The numba kernel itself is a compiled opaque lump under cProfile; this run
measures everything *around* it (precompute matmul, Find_Hg/population config
generation, h5py writes), which is exactly the post-kernel breakdown we want.

JIT warmup: the first call in a fresh process compiles the kernel (seconds),
then @njit(cache=True) amortizes it on disk. We run one untimed warm pass
first (and report its wall time as the per-fresh-process compile cost), then
profile a second, fully-warm pass.

Usage:
    python scripts/profile_forward.py --config configs/profile_rocking.toml \
        --output /tmp/prof_centered
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from dfxm_geo.pipeline import SimulationConfig, run_simulation

# Functions whose cumulative time forms the per-stage breakdown. Reported
# explicitly on top of the generic top-N so the stage story is unambiguous.
STAGE_FUNCS = (
    "build_dislocation_population",
    "Find_Hg_from_population",
    "Find_Hg",
    "precompute_forward_static",
    "forward_from_static",
    "write_simulation_h5",
)


def _run(config: SimulationConfig, out: Path) -> None:
    # Force single-thread so cProfile attribution is clean.
    config.io.max_workers = 1
    run_simulation(config, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.from_toml(args.config)

    # 1) Warm pass: compiles the numba kernel + warms disk caches. Untimed for
    #    the profile, but we record its wall time as the cold-start cost.
    t0 = time.perf_counter()
    _run(SimulationConfig.from_toml(args.config), args.output / "warm")
    warm_s = time.perf_counter() - t0

    # 2) Profiled pass: fully warm, single-thread.
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.runcall(_run, config, args.output / "profiled")
    total_s = time.perf_counter() - t0

    prof_path = args.output / "forward_single_thread.prof"
    prof.dump_stats(str(prof_path))

    # Build the text summary.
    buf = io.StringIO()
    st = pstats.Stats(prof, stream=buf)
    buf.write(f"config            : {args.config}\n")
    buf.write(f"cold-start (warm) : {warm_s:8.3f} s  (includes numba compile)\n")
    buf.write(f"warm total        : {total_s:8.3f} s  (single-thread)\n")
    buf.write("\n=== stage breakdown (cumulative seconds) ===\n")
    # pstats keys are (filename, lineno, funcname); match on funcname.
    stats_dict = st.stats  # type: ignore[attr-defined]
    for want in STAGE_FUNCS:
        hits = [(k, v) for k, v in stats_dict.items() if k[2] == want]
        for k, v in hits:
            # v = (cc, nc, tt, ct, callers); ct = cumulative time.
            buf.write(f"  {want:32s} {v[3]:8.3f}  ({k[0].split('/')[-1]}:{k[1]})\n")
        if not hits:
            buf.write(f"  {want:32s}    (not called)\n")
    buf.write("\n=== top 30 by cumulative time ===\n")
    st.sort_stats("cumulative").print_stats(30)
    buf.write("\n=== top 30 by total (self) time ===\n")
    st.sort_stats("tottime").print_stats(30)

    summary = buf.getvalue()
    (args.output / "profile_summary.txt").write_text(summary)
    print(summary)
    print(f"\nWrote {prof_path}")
    print(f"Wrote {args.output / 'profile_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
