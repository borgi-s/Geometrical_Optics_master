# Reproducing simulations

This guide describes how to reproduce a DFXM forward-simulation run using
the config-driven CLI (`dfxm-forward`, introduced in Phase 6 of the cleanup).

For the physics background, see [`physics.md`](physics.md). For the package
layout, see [`architecture.md`](architecture.md).

## Prerequisites

- Python 3.11+ in a clean virtual environment.
- A precomputed reciprocal-space resolution kernel npz (`Resq_i_*.npz`)
  with all generation parameters bundled into the same archive. This is
  **not** in the repo (it is large and depends on instrument parameters).
  Generate it once with:

  > **v1.0.3 migration note:** Legacy `.pkl` files from before v1.0.3 are no
  > longer supported by `_load_default_kernel`. Regenerate via
  > `dfxm-bootstrap --config configs/default.toml` (~50 s); the resulting
  > `.npz` is a drop-in replacement at the same canonical path.

  ```bash
  python reciprocal_space/generate_Resq_i.py
  ```

  Place the resulting `Resq_i_<timestamp>.npz`
  under `reciprocal_space/pkl_files/`. The default kernel filename the
  loader looks for is set in
  `dfxm_geo.direct_space.forward_model.pkl_fn`. To use a different
  filename, edit that module-level constant or call
  `_load_default_kernel(pkl_path)` explicitly before invoking
  the pipeline.

- Install the package in editable mode so the `dfxm-forward` console
  script is on `PATH`:

  ```bash
  pip install -e ".[dev]"
  ```

## Running a default simulation

```bash
dfxm-forward --config configs/default.toml --output ./out
```

## Output: `<output_dir>/dfxm_geo.h5`

`dfxm-forward` produces a single HDF5 file with the full simulation in it. Schema documented in [output-format.md](output-format.md). All metadata needed to reproduce the run — config TOML, kernel hash, git SHA, machine, timestamps — is embedded under `/dfxm_geo/`.

SVG figures (mosaicity maps, qi cross-section) land alongside the .h5 at `<output_dir>/figures/`.

The file contains 1-2 BLISS scans:

- `/1.1/` — full simulation with `ndis=151` dislocations at `dis=4 µm` spacing.
- `/2.1/` — the same rocking grid with `Hg=0` (perfect crystal), for differential analysis. Only present when `include_perfect_crystal = true` in the `[io]` config section.

To skip the perfect-crystal pass, set `include_perfect_crystal = false`
in the `[io]` section of the config.

> **Legacy output:** Before v1.1.0, `dfxm-forward` wrote two per-frame `.npy` directories (`images10/` and `images10_perf_crystal/`, default: 61 × 61 = 3,721 files each). These directories can be converted to the new format with `dfxm-migrate-output`.

## Config schema

```toml
[crystal]
dis = 4        # inter-dislocation distance (µm)
ndis = 151     # number of dislocations

[scan]
phi_range = 0.03438    # half-range in degrees
phi_steps = 61
chi_range = 0.11459    # half-range in degrees
chi_steps = 61

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "images10"
perfect_dirname = "images10_perf_crystal"
include_perfect_crystal = true
```

### Edge-effect guidance for `(dis, ndis)`

A note in the original `init_forward.py` recommends a minimum `ndis` per
inter-dislocation distance `dis` to avoid edge artifacts in the simulated
field. Pre-built variant configs are in `configs/`:

| `dis` (µm) | min recommended `ndis` | config file |
|------------|------------------------|-------------|
| 0.25       | 7501                   | `configs/variants/dis_0p25.toml` |
| 0.5        | 1151                   | `configs/variants/dis_0p5.toml`  |
| 1.0        | 501                    | `configs/variants/dis_1.toml`    |
| 2.0        | 251                    | `configs/variants/dis_2.toml`    |
| 4.0        | 151                    | `configs/default.toml`           |

### What is *not* yet configurable (v1)

| Parameter | Currently | Why fixed |
|-----------|-----------|-----------|
| `psize` (pixel size) | `40 nm` | Used to derive the detector ray grid `rl` at module import. |
| `zl_rms` (beam profile) | `0.15 µm / 2.35` | Same — derives `rl`. |
| `(h, k, l)` reflection | `(-1, 1, -1)` | `forward_model.q_hkl` is a module-level global; only `Find_Hg` accepts h/k/l args, and the pipeline doesn't currently override it. |
| Reciprocal-space kernel parameters | hard-coded in `generate_Resq_i.py` | Independent generation step. |

Making `psize`, `zl_rms`, and `(h, k, l)` configurable requires deriving
the detector ray grid (`rl`) and the `q_hkl` module global at runtime
rather than at import. This is tracked for a future refactor (Phase 8
revisit).

## Post-processing outputs

`dfxm-forward` runs post-processing by default after the simulation. Outputs land under:

- `<output>/analysis/phi_list.npy` — mosaicity-in-φ map (radians)
- `<output>/analysis/chi_list.npy` — mosaicity-in-χ map (radians)
- `<output>/analysis/qi_field.npy` — qi field at the (x, y) plane, z = 0 included
- `<output>/analysis/chi_shift_deg.txt` — scalar χ-axis correction (bare float, degrees)
- `<output>/figures/mosaicity_maps.svg` — "Extreme Phi" / "Extreme Chi" 2-panel figure
- `<output>/figures/qi_cross_section.svg` — qi_1 / qi_2 (x, y) cross-section

Skip post-processing for a sim-only run:

    dfxm-forward --config configs/default.toml --output output/ --no-postprocess

Re-run post-processing against an existing simulation directory without
re-simulating (useful after tweaking `[postprocess]` parameters):

    dfxm-forward --config configs/default.toml --output output/ --postprocess-only

> **Warning:** `--postprocess-only` against a directory produced with non-default
> `[crystal]` parameters re-uses the module-level Hg from the default kernel
> load, not the Hg used to produce the saved stacks. The qi-field output can
> then be inconsistent with the rocking sweep. For correctness, set
> `fm.Hg` explicitly in Python rather than via the CLI flag, or re-run the
> full pipeline.

The post-processing stage requires the reciprocal-space resolution kernel
npz (same as the simulation stage) to compute the qi field.

## Reproducing paper figures

The figures in
[Borgi et al., *J. Appl. Cryst.* (2024)](https://doi.org/10.1107/S1600576724001183)
were produced by `init_forward.py` against specific dislocation
configurations. The CLI reproduces the full workflow end-to-end:

1. Run the appropriate `dfxm-forward --config configs/<variant>.toml`
   to produce the image stacks and post-processing outputs.
2. The per-pixel center-of-mass, phi/chi mosaicity maps, and SVG figures
   are written automatically under `<output>/analysis/` and
   `<output>/figures/` (see [Post-processing outputs](#post-processing-outputs)
   above).

## Archived reference datasets

Pre-computed reference outputs for paper figures are scheduled for
deposit on Zenodo as part of Phase 9. Once published, this section will
link to the DOI and list which Zenodo records correspond to which
figures in the paper.

> **Status:** not yet deposited. Until then, contact
> [borgi@dtu.dk](mailto:borgi@dtu.dk) for access to the reference dataset.

## Numerical changes vs. older `main`

Two corrections landed during the Phase-6/7 cleanup that change the
numerics of `Fd_find` (and therefore every downstream simulation). If
you need to reproduce paper results bit-for-bit, you'll need an older
snapshot of `main` — the figures in the *J. Appl. Cryst.* 2024 paper
were generated against pre-correction code.

1. **`Fdd[1,1]` sign correction** (commit `3b71b33`, follows Appendix A
   of the paper). Pre-correction:
   `F_d[1,1] ∝ y_d (x_d² - y_d² - 2ν(x_d² + y_d²))`. Post-correction
   (current): `F_d[1,1] ∝ y_d (x_d² - y_d² + 2ν(x_d² + y_d²))`. This is
   *physics correctness* — the corrected version had been in the
   `CDD_Khaled` / `Beam_Stop` branch all along; `main` carried the
   early-days sign error and the cleanup pulled it through Phase 4.

2. **Bipolar wall consistency across the `ndis > 100` boundary**
   (commit `c07dea4`). Pre-fix, `multi_dislocs_parallel` summed walls
   at monotone offsets `{-1·dis, -2·dis, …}` while the sequential
   branch summed bipolar offsets `{+1, -1, +2, -2, …}·dis`. Both
   converge to the same infinite wall as `ndis → ∞` so this is an
   *edge-effect* difference rather than a physics-correctness one
   (the wall is meant to be infinite; finite `ndis` is a numerical
   approximation). The fix makes the two branches consistent so the
   regression-test net covers both.

## Reproducibility checklist

When publishing or sharing a simulation run, record:

- Git commit SHA of the package (`git rev-parse HEAD`).
- Package version (`python -c "import dfxm_geo; print(dfxm_geo.__version__)"`).
- The exact TOML config used.
- The `Resq_i_*.npz` filename (params bundled, no separate sidecar).
- Output of `pip freeze` (or a lockfile) for the venv.

Since v1.1.0, all of the above (except the pip freeze) are embedded automatically in `dfxm_geo.h5` under `/dfxm_geo/` — the git SHA, kernel hash, config TOML, machine hostname, and run timestamps are written at the end of every `run_simulation` call. The pip freeze / lockfile must still be recorded separately.
