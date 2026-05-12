# Reproducing simulations

This guide describes how to reproduce a DFXM forward-simulation run using
the config-driven CLI (`dfxm-forward`, introduced in Phase 6 of the cleanup).

For the physics background, see [`physics.md`](physics.md). For the package
layout, see [`architecture.md`](architecture.md).

## Prerequisites

- Python 3.11+ in a clean virtual environment.
- A precomputed reciprocal-space resolution kernel pickle (`Resq_i_*.pkl`)
  with its sidecar `_vars.txt`. This is **not** in the repo (it is large
  and depends on instrument parameters). Generate it once with:

  ```bash
  python reciprocal_space/generate_Resq_i.py
  ```

  Place the resulting `Resq_i_<timestamp>.pkl` + `<timestamp>_vars.txt`
  under `reciprocal_space/pkl_files/`. The default kernel filename the
  loader looks for is set in
  `dfxm_geo.direct_space.forward_model.pkl_fn`. To use a different
  filename, edit that module-level constant or call
  `_load_default_kernel(pkl_path, vars_path)` explicitly before invoking
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

This produces two image-stack directories under `./out/`:

- `images10/` — full simulation with `ndis=151` dislocations at
  `dis=4 µm` spacing.
- `images10_perf_crystal/` — the same rocking grid with `Hg=0` (perfect
  crystal), for differential analysis.

Each directory contains `chi_steps × phi_steps` `.npy` files
(default: 61 × 61 = 3,721 images per stack). Filename format:
`mosa_test_0000_{chi_idx:04d}_{phi_idx:04d}.npy`.

To skip the perfect-crystal pass, set `include_perfect_crystal = false`
in the `[io]` section of the config.

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

## Reproducing paper figures

The figures in
[Borgi et al., *J. Appl. Cryst.* (2024)](https://doi.org/10.1107/S1600576724001183)
were produced by `init_forward.py` against specific dislocation
configurations. The CLI reproduces the forward simulation portion of
that workflow; the post-simulation analysis (per-pixel center-of-mass,
phi/chi mosaicity maps, and SVG figure generation) still lives in
`init_forward.py` for now. **To reproduce a paper figure end-to-end as
of this release:**

1. Run the appropriate `dfxm-forward --config configs/<variant>.toml`
   to produce the image stacks.
2. Adapt `init_forward.py` to load those stacks (set `DFXM_DATA_DIR` and
   adjust the hardcoded `images10*` directory names if needed) and run
   from the analysis section onward.

A future phase (currently tracked as Phase 9 follow-up) will port the
analysis + plotting into `dfxm_geo.analysis` so that a single CLI
invocation produces both the data and the figures.

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
- The `Resq_i_*.pkl` filename and its `_vars.txt` sidecar.
- Output of `pip freeze` (or a lockfile) for the venv.

The CLI does not yet emit a run-metadata sidecar automatically; this is
tracked for a future enhancement.
