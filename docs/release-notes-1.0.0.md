# v1.0.0 — Cluster integration

## What's new

- **One-command environment install** — `conda env create -f environment.yml`
  (and `environment-dev.yml` for development) covers every runtime dep
  including `xraylib`, `plotly`, and `psutil`. No more hand-translating
  pyproject.toml to conda specs.
- **`dfxm-bootstrap` console script** — the missing stage 0. Run it once
  per environment to generate the reciprocal-space kernel pickle from a
  TOML `[reciprocal]` block; fresh-clone users no longer need tribal
  knowledge to find the right command. Validates unknown keys + missing
  files with clean error messages instead of Python tracebacks.
- **Fail-loud stage 0 in the pipeline** — `dfxm-forward` and
  `dfxm-identify` raise a clear `FileNotFoundError` with the
  `dfxm-bootstrap` instruction when the kernel is missing, instead of
  the previous cryptic `RuntimeError` from inside `forward()`. The
  pipeline also recovers automatically if the pickle exists on disk but
  hasn't been loaded yet (e.g. bootstrap-then-run in the same process).
- **Cluster batch templates** — `lsf/forward_single.bsub`,
  `lsf/identify_array.bsub`, `slurm/forward_single.sbatch`,
  `slurm/identify_array.sbatch` for DTU HPC (LSF) and ESRF (SLURM).
  Each template has an `EDIT THESE` block for queue/partition,
  walltime, and memory, plus an idempotent stage-0 bootstrap guard.
- **`docs/cluster-runs.md`** — soup-to-nuts cluster guide with a
  DTU-vs-ESRF cheat sheet, memory/walltime sizing table, and rsync
  patterns.
- **README hero images** — two example PNGs (forward frame +
  mosaicity map) embedded near the top, regenerable via
  `scripts/render_readme_examples.py --small`.
- **`[reciprocal]` block in `configs/default.toml`** — drives
  `dfxm-bootstrap` with the CDD_inc canonical recipe (Al 111 at 17 keV,
  25 mm BFP beamstop, Nrays=1e8). A drift test pins these values to
  `generate_kernel`'s defaults so the TOML and the function signature
  can never silently diverge.

## Configuration internals

- `reciprocal_res_func` and `generate_kernel` both accept
  `output_path: Path | None`, letting callers pin the pickle to a
  specific path (the canonical `dfxm-forward` lookup location, by
  default) instead of always writing under `pkl_files/` in CWD.

## Known limitations

- The kernel pickle is still a 128 MB blob shipped as a generated
  artefact; alternatives are filed in
  `followups_kernel_pickle_alternatives` (six candidate directions, not
  gating).
- Reflection runtime configuration (h, k, l beyond Al 111) is deferred
  to v1.1.
- The single pre-existing failing test
  (`TestDfxmForwardSampleRemountCLI::test_dfxm_forward_with_sample_remount_S2_runs`)
  is a Windows-specific subprocess-PATH issue inherited from before
  this PR. The `dfxm-forward` console script itself works correctly
  when invoked from a normal shell. Tracked separately.

## Migration from v0.9.0

No breaking API changes for typed callers. Canonical install path on a
cluster:

```bash
conda env create -f environment.yml
conda activate dfxm-geo
pip install -e .
dfxm-bootstrap --config configs/default.toml
dfxm-forward --config configs/default.toml --output output/run01/
```

`generate_kernel` now returns `Path` instead of the date string — the
only in-repo caller (`__main__`) discards the return value; external
collaborators scripting against the old `str` return should update.

## Test + lint stats at release

- 269 passed, 1 pre-existing Windows-only failure, 6 bench tests
  deselected (run with `pytest -m bench`).
- `mypy src/dfxm_geo/`: 0 errors across 25 source files.
- `ruff check src/dfxm_geo/ scripts/ tests/`: clean.
- 20 commits on `feature/cluster-integration-v1.0` since the v0.9.0
  baseline merge to `main`.
