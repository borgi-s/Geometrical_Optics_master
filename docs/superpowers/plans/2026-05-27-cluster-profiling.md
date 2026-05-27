# Cluster Profiling Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained profiling harness that runs on DTU HPC (LSF) to (1) confirm the forward-model perf arc holds on a compute node, (2) locate the post-numba-kernel Python bottleneck, and (3) size the 100k-image ML dataset in node-hours and storage.

**Architecture:** Two standalone scripts under `scripts/` + two throwaway configs + one LSF template + a walkthrough doc. `profile_forward.py` cProfiles one single-threaded rocking scan (clean attribution; numba kernel is an opaque lump). `scaling_sweep.py` mirrors `write_simulation_h5`'s worker pattern (shared `base_qc` + `ThreadPoolExecutor` over `forward_from_static`) to measure pure-forward speedup vs. worker count, then runs one realistic full `run_simulation` to get per-config wall time + on-disk bytes and extrapolate to 100k. Node core count is auto-detected at runtime via `os.sched_getaffinity`. **No `src/dfxm_geo/` changes** — the shipped package is untouched.

**Tech Stack:** Python 3.11/3.12 (venv on laptop, conda `dfxm-geo` on cluster), numpy, numba, h5py, cProfile/pstats, concurrent.futures, LSF `bsub`, snakeviz (local analysis only).

---

## File Structure

- Create: `configs/profile_rocking.toml` — centered (1 dislocation) rocking-scan unit.
- Create: `configs/profile_rocking_random.toml` — random_dislocations variant (exercises config-gen).
- Create: `scripts/profile_forward.py` — single-thread cProfile + stage breakdown.
- Create: `scripts/scaling_sweep.py` — worker-scaling table + 100k extrapolation.
- Create: `lsf/profile.bsub` — runner that bundles artifacts into a tarball.
- Create: `docs/cluster-profiling.md` — setup → submit → bring-back → analyze walkthrough.

All files are new. No existing files are modified. No package tests are added (nothing in `src/` changes); verification is a local smoke run on a downscaled config.

---

### Task 1: Throwaway rocking-scan configs

**Files:**
- Create: `configs/profile_rocking.toml`
- Create: `configs/profile_rocking_random.toml`

- [ ] **Step 1: Write `configs/profile_rocking.toml`**

Derived from `configs/default.toml`: same `[reciprocal]` block (so it uses the same kernel `dfxm-bootstrap --config configs/default.toml` produces), a 1D phi rocking scan (no `[scan.chi]`), centered single dislocation, perfect-crystal scan OFF (isolate one scan), postprocess OFF (we are profiling the forward kernel, not analysis).

```toml
# Throwaway profiling config: 1 dislocation config -> 1D rocking scan (~21 frames).
# Used by scripts/profile_forward.py and scripts/scaling_sweep.py. NOT a science config.

[reciprocal]
hkl        = [-1, 1, -1]
keV        = 17.0
Nrays      = 100_000_000
npoints1   = 400
npoints2   = 200
npoints3   = 200
qi1_range  = 5e-4
qi2_range  = 7.5e-3
qi3_range  = 7.5e-3
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.1106382978723403e-4
eps_rms     = 6.0e-5
D  = 0.000565685424949238
d1 = 0.274
beamstop   = true
bs_height  = 25e-3
aperture   = true
knife_edge = false
dphi_range = 0.0

[scan.phi]
range = 6e-4
steps = 21
# [scan.chi] intentionally omitted -> 1D rocking scan.

[crystal]
mode = "centered"

[crystal.centered]
b = [1, -1, 0]
n = [1, 1, 1]
t = [1, 1, -2]

[io]
include_perfect_crystal = false   # isolate the dislocation scan for profiling
# max_workers is set by the scripts, not here.

[postprocess]
enabled = false                   # profiling the forward kernel, not analysis
```

- [ ] **Step 2: Write `configs/profile_rocking_random.toml`**

Identical except the crystal block selects `random_dislocations` (exercises the `Find_Hg_from_population` config-gen path a 100k-config sweep pays every sample). Seed is fixed for reproducibility.

```toml
# Throwaway profiling config: random_dislocations variant (config-gen cost).
# Same rocking scan as profile_rocking.toml; different crystal mode.

[reciprocal]
hkl        = [-1, 1, -1]
keV        = 17.0
Nrays      = 100_000_000
npoints1   = 400
npoints2   = 200
npoints3   = 200
qi1_range  = 5e-4
qi2_range  = 7.5e-3
qi3_range  = 7.5e-3
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.1106382978723403e-4
eps_rms     = 6.0e-5
D  = 0.000565685424949238
d1 = 0.274
beamstop   = true
bs_height  = 25e-3
aperture   = true
knife_edge = false
dphi_range = 0.0

[scan.phi]
range = 6e-4
steps = 21

[crystal]
mode = "random_dislocations"

[crystal.random_dislocations]
ndis = 4
sigma = 5.0
min_distance = 4.0
seed = 42

[io]
include_perfect_crystal = false

[postprocess]
enabled = false
```

- [ ] **Step 3: Validate both configs parse**

Run (laptop venv):
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "from dfxm_geo.pipeline import SimulationConfig; from pathlib import Path; [print(p, SimulationConfig.from_toml(Path('configs')/p).scan.derived_mode_name(), SimulationConfig.from_toml(Path('configs')/p).crystal.mode) for p in ['profile_rocking.toml','profile_rocking_random.toml']]"
```
Expected: prints `profile_rocking.toml rock <something> centered` style output (mode label for phi-only is whatever `derived_mode_name()` returns for a single scanned axis) and `... random_dislocations`, with NO traceback.

- [ ] **Step 4: Commit**

```
git add configs/profile_rocking.toml configs/profile_rocking_random.toml
git commit -m "profiling: throwaway rocking-scan configs (centered + random)"
```

---

### Task 2: `scripts/profile_forward.py` — single-thread cProfile + stage breakdown

**Files:**
- Create: `scripts/profile_forward.py`

- [ ] **Step 1: Write the script**

```python
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
        hits = [
            (k, v) for k, v in stats_dict.items() if k[2] == want
        ]
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
```

- [ ] **Step 2: Smoke-test locally on a downscaled config**

Make a tiny temp config (5 frames) so the run is fast on the laptop, and confirm artifacts appear. (Requires a bootstrapped kernel locally — see note at end of Task.)

Run (laptop venv, from repo root):
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" - <<'PY'
import tomllib, tomli_w  # if tomli_w absent, just hand-edit a copy
PY
```
Simpler: copy `configs/profile_rocking.toml` to `configs/_smoke.toml` and change `steps = 21` to `steps = 5`, then:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_forward.py --config configs/_smoke.toml --output C:\Users\borgi\tmp\prof_smoke
```
Expected: prints the stage breakdown + top-30 tables; `C:\Users\borgi\tmp\prof_smoke\forward_single_thread.prof` and `profile_summary.txt` exist; `precompute_forward_static` and `forward_from_static` appear with nonzero cumulative time. No traceback.

> If this fails with `FileNotFoundError: Reciprocal-space kernel npz not found`, bootstrap a local kernel first:
> `& "...\.venv\Scripts\python.exe" -m dfxm_geo.reciprocal_space.kernel --if-missing --config configs/default.toml`
> (or `dfxm-bootstrap --if-missing --config configs/default.toml` if the entry point is on PATH).

- [ ] **Step 3: Clean up the smoke config and commit**

```
del configs\_smoke.toml
git add scripts/profile_forward.py
git commit -m "profiling: single-thread cProfile + stage breakdown script"
```

---

### Task 3: `scripts/scaling_sweep.py` — worker scaling + 100k extrapolation

**Files:**
- Create: `scripts/scaling_sweep.py`

- [ ] **Step 1: Write the script**

```python
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
    population = fm.build_dislocation_population(config.crystal, fov_lateral_um=fov_lateral_um, rng=None)
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
    fm.forward_from_static(base_qc, float(frames.phi_pf[0]), float(frames.chi_pf[0]), float(frames.two_dtheta_pf[0]))
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
```

- [ ] **Step 2: Smoke-test locally on a downscaled config**

```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/scaling_sweep.py --config configs/profile_rocking.toml --output C:\Users\borgi\tmp\sweep_smoke --repeats 1 --max-workers 4
```
Expected: prints the (A) scaling table (workers 1,2,4), then (B) per-config wall + the 100k extrapolation (configs needed ≈ 4762, a node-hours figure, a TB figure). `C:\Users\borgi\tmp\sweep_smoke\scaling_summary.txt` exists. No traceback. (On Windows `usable_cores()` uses the `os.cpu_count()` fallback — that's expected and fine for the smoke.)

- [ ] **Step 3: Commit**

```
git add scripts/scaling_sweep.py
git commit -m "profiling: worker-scaling sweep + 100k extrapolation script"
```

---

### Task 4: `lsf/profile.bsub` — runner

**Files:**
- Create: `lsf/profile.bsub`

- [ ] **Step 1: Write the LSF template**

Modeled on `lsf/forward_single.bsub` (same conda-activation block, same `dfxm-bootstrap --if-missing` guard). Prints node info, runs `profile_forward.py` for both crystal variants, then `scaling_sweep.py`, then tars artifacts.

```bash
#!/bin/bash
# ==============================================================================
# DTU HPC (LSF) — cluster profiling run for the ML-data pipeline
# ==============================================================================
# Submit with:   bsub < lsf/profile.bsub
# Monitor with:  bjobs -l <JOBID>
# See docs/cluster-profiling.md for the full walkthrough.
# ==============================================================================
#
# >>> EDIT THESE >>>
#BSUB -J dfxm-profile
#BSUB -q hpc                           # DTU HPC default queue
#BSUB -W 02:00                         # Walltime HH:MM
#BSUB -n 32                            # CPU slots — set to a full node; query
                                       # `lshosts`/`bhosts hpc` first. The sweep
                                       # auto-detects actual usable cores, so an
                                       # over-estimate here is harmless.
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"               # single host (thread parallelism)
#BSUB -o logs/profile-%J.out
#BSUB -e logs/profile-%J.err
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
OUTROOT="${WORKDIR}/output/profile_${LSB_JOBID:-local}"
mkdir -p "${OUTROOT}" logs

# --- Node info: recorded so artifacts say exactly what we ran on. ---
echo "===== NODE INFO ====="
echo "host: $(hostname)"
nproc || true
lscpu || true
echo "====================="

# --- conda activation (mirrors lsf/forward_single.bsub) ---
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
dfxm-bootstrap --if-missing --config "${WORKDIR}/configs/default.toml"

# Stage 1 — single-thread cProfile for both crystal variants.
python scripts/profile_forward.py --config configs/profile_rocking.toml        --output "${OUTROOT}/centered"
python scripts/profile_forward.py --config configs/profile_rocking_random.toml --output "${OUTROOT}/random"

# Stage 2 — worker-scaling sweep + 100k extrapolation (centered variant).
python scripts/scaling_sweep.py --config configs/profile_rocking.toml --output "${OUTROOT}/scaling" --repeats 3

# Stage 3 — bundle artifacts (drop the heavy HDF5 outputs; keep summaries+prof).
find "${OUTROOT}" -name '*.h5' -delete
find "${OUTROOT}" -type d -name 'scan*' -exec rm -rf {} + 2>/dev/null || true
TARBALL="${WORKDIR}/profile_${LSB_JOBID:-local}.tar.gz"
tar -czf "${TARBALL}" -C "$(dirname "${OUTROOT}")" "$(basename "${OUTROOT}")"
echo "Done. Artifacts in ${TARBALL}"
```

- [ ] **Step 2: Lint the script syntax locally**

Run (Bash tool):
```
bash -n lsf/profile.bsub && echo "syntax OK"
```
Expected: `syntax OK` (no parse errors).

- [ ] **Step 3: Commit**

```
git add lsf/profile.bsub
git commit -m "profiling: LSF runner bundling cProfile + scaling artifacts"
```

---

### Task 5: `docs/cluster-profiling.md` — walkthrough

**Files:**
- Create: `docs/cluster-profiling.md`

- [ ] **Step 1: Write the walkthrough**

```markdown
# Cluster profiling run (ML-data pipeline)

Profiles the forward model on a DTU HPC compute node to answer: does the perf
arc hold on the node, where is the post-numba-kernel bottleneck, and is 100k
images feasible in compute + storage. See the design spec at
`docs/superpowers/specs/2026-05-27-cluster-profiling-design.md`.

## 0. Find the node size first

On the login node, before submitting, check how many cores a node on the
target queue has:

```bash
ssh <user>@login.hpc.dtu.dk
lshosts            # MAX columns / ncpus per host type
bhosts hpc         # per-host slot counts on the hpc queue
nodestat -F hpc    # (if available) live node status
```

Set `#BSUB -n` in `lsf/profile.bsub` to a full node. Over-estimating is
harmless — `scaling_sweep.py` auto-detects the actually-usable cores at
runtime and caps the sweep to that.

## 1. One-time setup

```bash
cd ~/Geometrical_Optics_master
git fetch origin
git checkout profile/cluster-ml-data    # or main once merged
git pull

conda env create -f locks/environment-dtu-linux-64.yml   # ~2 min (fast path)
conda activate dfxm-geo
pip install -e ".[dev]"

# Sanity: tests green before profiling.
python -m pytest -q

# Generate the resolution kernel once (login node, ~50 s).
dfxm-bootstrap --if-missing --config configs/default.toml
```

## 2. Submit

```bash
bsub < lsf/profile.bsub
bjobs -l <JOBID>
tail -f logs/profile-<JOBID>.out      # node info + live progress
```

Walltime is 2 h; the run itself is minutes. Output lands in
`output/profile_<JOBID>/` and is tarred to `profile_<JOBID>.tar.gz`.

## 3. Bring the artifacts back

```powershell
# From the laptop (PowerShell). Use scp if rsync is unavailable.
scp <user>@login.hpc.dtu.dk:~/Geometrical_Optics_master/profile_<JOBID>.tar.gz .
tar -xzf profile_<JOBID>.tar.gz
```

The tarball contains, per crystal variant:
`forward_single_thread.prof` + `profile_summary.txt`, and
`scaling/scaling_summary.txt`.

## 4. Analyze locally

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pip install snakeviz
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m snakeviz profile_<JOBID>\centered\forward_single_thread.prof
```

`snakeviz` opens a browser icicle chart. Read `profile_summary.txt` for the
stage breakdown and `scaling_summary.txt` for the speedup curve + the 100k
node-hours/TB extrapolation.

## Interpreting the numbers

- **`scaling_summary.txt` (A) table** — if `speedup` plateaus well below
  `usable cores`, the threaded scan is memory-bandwidth bound on the node
  (the laptop plateaued ~4 workers). That caps single-node throughput and
  argues for array fan-out across nodes.
- **`profile_summary.txt` stage breakdown** — `forward_from_static`
  cumulative is mostly the opaque numba kernel. Large `write_simulation_h5`
  or `Find_Hg_from_population` cumulative means HDF5 I/O or config-gen is the
  next optimization target, not the kernel.
- **Extrapolation** — `total storage` TB is the feasibility gate for 100k
  rocking scans; `node-hours` sizes the fan-out.
```

- [ ] **Step 2: Commit**

```
git add docs/cluster-profiling.md
git commit -m "profiling: cluster profiling walkthrough doc"
```

---

### Task 6: Final local verification + branch wrap-up

**Files:** none (verification only)

- [ ] **Step 1: Full local smoke of both scripts on the real configs (small worker cap)**

```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_forward.py --config configs/profile_rocking.toml --output C:\Users\borgi\tmp\prof_centered
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_forward.py --config configs/profile_rocking_random.toml --output C:\Users\borgi\tmp\prof_random
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/scaling_sweep.py --config configs/profile_rocking.toml --output C:\Users\borgi\tmp\sweep --repeats 1 --max-workers 4
```
Expected: all three produce their summary files with no traceback; `profile_summary.txt` shows `precompute_forward_static` + `forward_from_static` with nonzero cumulative time; `scaling_summary.txt` shows the extrapolation block.

- [ ] **Step 2: Confirm no `src/` changes leaked in**

Run:
```
git diff --stat main -- src/
```
Expected: empty output (the package is untouched).

- [ ] **Step 3: Clean up laptop scratch (CLAUDE.md wrap-up rule)**

Delete any `Fg_*.npy` >10 MB this session created under
`direct_space/deformation_gradient_tensors/`, and the `C:\Users\borgi\tmp\prof_*`/`sweep*` scratch dirs.
```
Remove-Item -Recurse -Force C:\Users\borgi\tmp\prof_centered, C:\Users\borgi\tmp\prof_random, C:\Users\borgi\tmp\sweep -ErrorAction SilentlyContinue
```

- [ ] **Step 4: Push the branch (only after user confirms) and open a PR or merge per the project flow.**

Per CLAUDE.md: confirm before pushing. The branch `profile/cluster-ml-data` is ready once Tasks 1–5 are committed and Step 1 here is green.

---

## Self-Review

**Spec coverage:**
- Unit-of-work config (1D rocking, 21 frames, centered + random) → Task 1. ✓
- Single-thread cProfile + stage breakdown + JIT-warmup-separated → Task 2. ✓
- Multi-thread scaling sweep + auto core-detection + 100k extrapolation (node-hours + TB) → Task 3. ✓
- LSF runner with node-info print + `dfxm-bootstrap --if-missing` + artifact tarball → Task 4. ✓
- Setup walkthrough + bring-back + snakeviz analysis → Task 5. ✓
- Node-size discovery at all three layers (scheduler query / job log / runtime affinity) → Task 5 §0, Task 4 node-info, Task 3 `usable_cores()`. ✓
- No production-code changes; verification via local smoke → Tasks 2–3 smoke steps + Task 6 Step 2. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output. ✓

**Type/name consistency:** `run_simulation(config, output_dir)`, `SimulationConfig.from_toml`, `config.io.max_workers`, `config.io.include_perfect_crystal`, `fm.precompute_forward_static`, `fm.forward_from_static(base_qc, phi, chi, two_dtheta)`, `fm.Find_Hg_from_population(population, h, k, l)`, `fm.build_dislocation_population(crystal, fov_lateral_um, rng)`, `_build_scan_frames(scan)` → `ScanFrames(phi_pf, chi_pf, two_dtheta_pf, n_frames)`, `_load_resolution(config.reciprocal)` — all verified against `pipeline.py` / `forward_model.py` / `io/hdf5.py`. ✓

**Known risk:** `derived_mode_name()` label for a phi-only scan is not asserted to a specific string (Task 1 Step 3 only checks it parses) — intentional, since the label text doesn't affect profiling.
