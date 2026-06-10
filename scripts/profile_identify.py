#!/usr/bin/env python
"""Single-thread profile of one `dfxm-identify` run with a per-stage breakdown.

M1 Phase 2a of the June-2026 roadmap: measure where an identify config's time
goes — startup (import + JIT + kernel load) vs Hg geometry vs frame compute vs
HDF5 write — before optimizing any of it. Mirrors scripts/profile_forward.py
but adds explicit perf_counter stage accumulators, because two things are
invisible to cProfile here:

* frames are computed in ThreadPoolExecutor worker threads (cProfile sees only
  the main thread), and
* the Hg/geometry work happens lazily inside ``write_identification_h5`` (the
  scan generator is consumed by the writer), so the writer's cumulative time
  conflates geometry, frame compute and actual I/O.

The accumulators wrap the stage functions at their call sites
(pipeline-module globals + ``io.hdf5._compute_frame`` + ``fm.Z_shift`` /
``fm.precompute_forward_static``) and time every call across all threads.
Buckets reported:

* ``kernel_load_s``  — ``_lookup_and_load_kernel`` (the npz LUT load)
* ``hg_s``           — ``Fd_find_mixed`` + ``Fd_find_multi_dislocs_mixed`` +
  ``fast_inverse2`` + ``Z_shift``
* ``frames_s``       — ``precompute_forward_static`` + every
  ``_compute_frame`` call (sum over worker threads)
* ``poisson_s``      — ``_maybe_apply_poisson_noise`` (post-write pass)
* ``io_other_s``     — ``write_identification_h5`` wall minus the hg/frames
  work nested inside it ≈ h5py writes + layout overhead
* ``unattributed_s`` — run total minus everything above (dispatch, config
  parsing, provenance)

JIT warmup: an untimed warm pass first (its wall time is the per-fresh-process
cold-start cost: numba compile + first kernel load), then the measured pass.
``io.max_workers`` is forced to 1 so cProfile attribution is clean and the
frame accumulator is not inflated by thread oversubscription.

Usage:
    python scripts/profile_identify.py --config configs/profile_identify_multi.toml \
        --output tmp/prof_identify_multi
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import subprocess
import sys
import threading
import time
from pathlib import Path

import dfxm_geo.direct_space.forward_model as fm
import dfxm_geo.io.hdf5 as hdf5
import dfxm_geo.pipeline as pipeline
from dfxm_geo.pipeline import load_identification_config, run_identification


class _Acc:
    """Thread-safe wall-time + call-count accumulator for one stage function."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, dt: float) -> None:
        with self._lock:
            self.seconds += dt
            self.calls += 1


def _wrap(fn, acc: _Acc):
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            acc.add(time.perf_counter() - t0)

    return timed


# stage name -> (module, attribute). Patched at the call-site module so the
# pipeline's module-global lookups hit the wrapper.
_STAGE_SITES = {
    "kernel_load": (pipeline, "_lookup_and_load_kernel"),
    "fd_mixed": (pipeline, "Fd_find_mixed"),
    "fd_multi": (pipeline, "Fd_find_multi_dislocs_mixed"),
    "fast_inverse2": (pipeline, "fast_inverse2"),
    "z_shift": (fm, "Z_shift"),
    "precompute": (fm, "precompute_forward_static"),
    "frames": (hdf5, "_compute_frame"),
    "writer": (pipeline, "write_identification_h5"),
    "poisson": (pipeline, "_maybe_apply_poisson_noise"),
}


def _install_stage_timers() -> tuple[dict[str, _Acc], callable]:
    """Wrap every stage function; return (accumulators, restore_fn)."""
    accs: dict[str, _Acc] = {}
    originals: list[tuple[object, str, object]] = []
    for name, (mod, attr) in _STAGE_SITES.items():
        original = getattr(mod, attr)
        acc = _Acc()
        accs[name] = acc
        originals.append((mod, attr, original))
        setattr(mod, attr, _wrap(original, acc))

    def restore() -> None:
        for mod, attr, original in originals:
            setattr(mod, attr, original)

    return accs, restore


def _fresh_import_seconds() -> float:
    """Time `import dfxm_geo.pipeline` in a fresh interpreter (per-subprocess
    fixed cost that a persistent worker pool would amortize)."""
    snippet = (
        "import time; t = time.perf_counter(); "
        "import dfxm_geo.pipeline; print(time.perf_counter() - t)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
    )
    return float(proc.stdout.strip().splitlines()[-1])


def profile_one(config_path: Path, output_dir: Path, *, skip_cprofile_dump: bool = False) -> dict:
    """Profile one identify config: warm pass, then a measured single-thread pass.

    Returns the stage-breakdown dict (also written to
    ``output_dir/stage_timings.json`` with a human summary alongside).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load():
        config = load_identification_config(config_path)
        config.io.max_workers = 1  # clean attribution; IOConfig is mutable
        return config

    # 1) Warm pass: numba compile + first kernel load + disk caches. Its wall
    #    time is the per-fresh-process cold-start cost.
    t0 = time.perf_counter()
    run_identification(_load(), output_dir / "warm")
    cold_start_s = time.perf_counter() - t0

    # 2) Measured pass: fully warm, single-thread, stage accumulators + cProfile.
    accs, restore = _install_stage_timers()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    try:
        prof.runcall(run_identification, _load(), output_dir / "profiled")
    finally:
        restore()
    total_s = time.perf_counter() - t0

    hg_s = (
        accs["fd_mixed"].seconds
        + accs["fd_multi"].seconds
        + accs["fast_inverse2"].seconds
        + accs["z_shift"].seconds
    )
    frames_s = accs["frames"].seconds + accs["precompute"].seconds
    writer_s = accs["writer"].seconds
    # Hg + frame work runs nested inside the writer (lazy generator), so the
    # writer's own I/O share is what remains of its wall time.
    io_other_s = max(0.0, writer_s - hg_s - frames_s)
    unattributed_s = max(
        0.0,
        total_s - accs["kernel_load"].seconds - writer_s - accs["poisson"].seconds,
    )

    result = {
        "config": str(config_path),
        "total_s": round(total_s, 3),
        "cold_start_s": round(cold_start_s, 3),
        "kernel_load_s": round(accs["kernel_load"].seconds, 3),
        "hg_s": round(hg_s, 3),
        "frames_s": round(frames_s, 3),
        "poisson_s": round(accs["poisson"].seconds, 3),
        "io_other_s": round(io_other_s, 3),
        "unattributed_s": round(unattributed_s, 3),
        "stage_calls": {name: acc.calls for name, acc in accs.items()},
        "stage_seconds_raw": {name: round(acc.seconds, 3) for name, acc in accs.items()},
    }

    prof_path = output_dir / "identify_single_thread.prof"
    if not skip_cprofile_dump:
        prof.dump_stats(str(prof_path))

    buf = io.StringIO()
    buf.write(f"config            : {config_path}\n")
    buf.write(f"cold-start (warm) : {cold_start_s:8.3f} s  (numba compile + first kernel load)\n")
    buf.write(f"warm total        : {total_s:8.3f} s  (single-thread)\n")
    buf.write("\n=== stage breakdown (wall seconds, all threads) ===\n")
    for key in ("kernel_load_s", "hg_s", "frames_s", "io_other_s", "poisson_s", "unattributed_s"):
        share = 100.0 * result[key] / total_s if total_s > 0 else 0.0
        buf.write(f"  {key:18s} {result[key]:8.3f}  ({share:5.1f} %)\n")
    buf.write("\n=== per-function accumulators ===\n")
    for name, acc in accs.items():
        buf.write(f"  {name:16s} {acc.seconds:8.3f} s  in {acc.calls:6d} calls\n")
    if not skip_cprofile_dump:
        buf.write("\n=== top 30 by cumulative time (main thread only) ===\n")
        st = pstats.Stats(prof, stream=buf)
        st.sort_stats("cumulative").print_stats(30)

    (output_dir / "profile_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
    (output_dir / "stage_timings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True, help="dfxm-identify TOML config.")
    ap.add_argument("--output", type=Path, required=True, help="Output directory.")
    ap.add_argument(
        "--skip-import-timing",
        action="store_true",
        help="Skip the fresh-subprocess import_s measurement (saves a few seconds).",
    )
    args = ap.parse_args()

    result = profile_one(args.config, args.output)
    if not args.skip_import_timing:
        result["fresh_import_s"] = round(_fresh_import_seconds(), 3)
        (args.output / "stage_timings.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    print((args.output / "profile_summary.txt").read_text(encoding="utf-8"))
    if "fresh_import_s" in result:
        print(f"fresh-process import: {result['fresh_import_s']:.3f} s")
    print(f"\nWrote {args.output / 'stage_timings.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
