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

# Generate the resolution kernel once (login node, ~50 s). We use
# profile_rocking.toml — its [reciprocal] block matches the packaged default,
# and it exists at the repo root (configs/default.toml does not; the canonical
# default lives under src/dfxm_geo/data/configs/).
dfxm-bootstrap --if-missing --config configs/profile_rocking.toml
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

## Fan-out batch (Phase 2b)

Profiling showed the threaded scan plateaus ~16 cores (memory-bandwidth bound),
so a big node is best used as config-level fan-out: many configs in parallel,
each thread-capped, rather than one over-threaded scan.

1. **Build a manifest.** Drop your per-config TOMLs in a directory (e.g.
   `configs/sweep/`) — one config per DFXM image/scan you want generated. The
   manifest can be that directory, a single `.toml`, or a `.txt` listing config
   paths (one per line; `#` comments and blanks ignored).
2. **Submit.** Edit `MANIFEST`, the `#BSUB` resource lines, `N_WORKERS`, and
   `THREADS_PER_WORKER` in `lsf/fanout.bsub`, then:

   ```bash
   bsub < lsf/fanout.bsub
   bjobs -l <JOBID>
   ```
3. **Size workers × threads.** `N_WORKERS × THREADS_PER_WORKER` should ≈ node
   cores. On a 128-core node, `8 × 16` is the sweet spot — since one scan
   plateaus ~16 cores, prefer more concurrent configs over more threads/scan.
4. **Outputs.** Each config writes `output/fanout_<JOBID>/<config-stem>/` with a
   float32 HDF5 plus its own log at `output/fanout_<JOBID>/<config-stem>.log`.
