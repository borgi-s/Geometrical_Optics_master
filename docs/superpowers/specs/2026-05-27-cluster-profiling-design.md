# Cluster profiling harness for the ML-data pipeline — design

**Date:** 2026-05-27
**Status:** approved (design), pending implementation plan
**Author:** Sina Borgi (with Claude Code)

## Motivation

The forward-model perf arc (static hoist + numba MC kernel, main `e1d432b`)
delivered ~6.6x/frame and ~17x threaded-scan speedups **on a laptop**. The
end goal is generating a large DFXM dataset for ML training: **~100,000
images** spanning **many dislocation configurations**, with each config's
parameters stored alongside its images.

Open questions this run answers:

1. Do the laptop speedups translate to a DTU HPC compute node (more cores,
   different memory bandwidth, NUMA)?
2. After the numba kernel, where does the *remaining* Python-level time go
   (precompute matmul, config generation, HDF5 write)?
3. What is the per-config wall time and **on-disk size**, and therefore the
   node-hours and **total storage (TB)** to reach 100k images — is it even
   feasible to store?

## Scope

- **Target cluster:** DTU HPC (LSF / `bsub`). Env + clone + kernel need
  first-time setup; the walkthrough includes it.
- **Unit of work:** 1 dislocation config → 1 **rocking scan** (1D phi scan,
  ~21 frames). 100k images ≈ 100,000 / 21 ≈ **~4,760 configs**.
- **No production-code changes.** All artifacts are new files under
  `scripts/`, a throwaway config, and an LSF template. The PyPI-shipped
  package (`src/dfxm_geo/`) is untouched; no new package tests.

Out of scope: implementing any optimization the profiling reveals (that is a
follow-on arc), and the array-job fan-out itself (this run characterizes one
node so we can *size* the fan-out).

## Key technical constraints

- **cProfile vs. numba.** `_mc_lut_forward` is `@njit(cache=True, nogil=True)`.
  cProfile only sees Python frames, so the compiled kernel appears as a
  single opaque call. The cProfile pass therefore measures everything
  *around* the kernel; it cannot break down the kernel's interior.
- **cProfile vs. threads.** cProfile cleanly attributes only the main thread.
  The pipeline parallelizes frames with a `nogil` numba kernel under a
  ThreadPoolExecutor, so worker-thread time is mis-attributed under cProfile.
  → **cProfile runs single-threaded; scaling is measured separately with
  wall-clock timing.**
- **JIT warmup.** A fresh process compiles the kernel on first call (seconds),
  then the on-disk cache (`cache=True`) amortizes it. 100k cold array jobs
  starting simultaneously can stampede the compile. → Every measured region
  is preceded by a throwaway pre-warm call; compile time is reported
  *separately* as the per-fresh-process cost.
- **Node size is unknown up front.** Discovered three ways (see §"Node-size
  discovery").

## Components

### 1. `configs/profile_rocking.toml` (throwaway unit-of-work config)

Derived from `configs/default.toml` with:

- `[scan.phi]` `steps = 21`; **`[scan.chi]` removed** → 1D rocking scan.
- `Nsub = 1`, default detector/kernel sizing (the `default.toml` kernel).

Two crystal variants are profiled (the script selects via `--crystal`):

- `centered` (1 dislocation) — cheapest; isolates pure forward cost.
- `random_dislocations` (`ndis≈4`, `seed` fixed) — exercises the
  **config-generation** path (`Find_Hg` / population sampling) that a
  100k-config sweep pays on every sample.

### 2. `scripts/profile_forward.py` — single-thread cProfile + stage timing

Runs **one** rocking scan at `workers=1`:

1. Pre-warm the numba JIT (one throwaway `forward_from_static`); report
   compile time separately.
2. `perf_counter` stage timings: kernel+config load → config-gen
   (`Find_Hg` / population) → `precompute_forward_static` (matmul) →
   per-frame forward loop → HDF5 write.
3. Wrap the compute region in `cProfile`; dump `forward_single_thread.prof`;
   print top-30 functions by cumulative and by tottime.

Output: `forward_single_thread.prof` + a `profile_summary.txt`.

### 3. `scripts/scaling_sweep.py` — multi-thread throughput (no cProfile)

After JIT pre-warm, runs the full rocking scan through the real `save` /
HDF5 path at `workers = 1, 2, 4, 8, 16, … up to usable cores`
(best-of-N wall-clock per point). Prints:

- table: `workers | wall_s | speedup | frames/s | per-config_s`
- one-config **output bytes** (HDF5 file size on disk)
- **extrapolations to 100k images**: configs needed (100k / 21), total
  node-hours at the best worker count, and **total storage (TB)**.

Worker range is **capped at the runtime-detected usable core count** (see
below) — never hardcoded.

### 4. `lsf/profile.bsub` — runner

- One full compute node: `#BSUB -n <cores>` (an `EDIT THESE` knob),
  `#BSUB -R "span[hosts=1]"`, generous walltime, `hpc` queue.
- Prints `lscpu` and `nproc` into the log for the record.
- Sources conda, `conda activate dfxm-geo`, `dfxm-bootstrap --if-missing`.
- Runs `profile_forward.py` (both crystal variants) then
  `scaling_sweep.py`.
- Tars artifacts into `profile_<JOBID>.tar.gz`.

### 5. Setup walkthrough (DTU needs first-time setup)

`ssh login.hpc.dtu.dk` → clone/`git pull` →
`conda env create -f locks/environment-dtu-linux-64.yml` →
`conda activate dfxm-geo` → `pip install -e ".[dev]"` →
`dfxm-bootstrap --if-missing --config configs/default.toml` →
`bsub < lsf/profile.bsub`.

### 6. Bring-back + local analysis

`rsync`/`scp` `profile_<JOBID>.tar.gz` back to the laptop; open the `.prof`
with `snakeviz` (browser flame chart) using the venv python.

## Node-size discovery (the open unknown)

Node core count is not known in advance and is handled at three layers:

1. **Before submitting** — query the scheduler on the login node:
   `lshosts` / `bhosts hpc` / `nodestat -F hpc` to see per-host core counts
   on the target queue; set `#BSUB -n` accordingly.
2. **In the job log** — `lscpu` + `nproc` are printed at job start, so the
   artifacts record exactly what we ran on.
3. **At runtime** — `scaling_sweep.py` calls
   `len(os.sched_getaffinity(0))` (falling back to `os.cpu_count()`) and
   caps the worker sweep to that, so it adapts to whatever node LSF assigns
   even if `#BSUB -n` is mis-set.

## Deliverables

- `configs/profile_rocking.toml`
- `scripts/profile_forward.py`
- `scripts/scaling_sweep.py`
- `lsf/profile.bsub`
- A short `docs/cluster-profiling.md` walkthrough (setup → submit →
  bring-back → analyze).

## Success criteria

- One `bsub` submission on DTU HPC produces a tarball containing:
  a single-thread `.prof` + stage-timing summary for both crystal variants,
  a worker-scaling table, and the 100k extrapolation (node-hours + TB).
- The artifacts answer: (1) does the speedup hold on the node, (2) what is
  the post-kernel bottleneck, (3) is 100k images feasible in compute and
  storage.
