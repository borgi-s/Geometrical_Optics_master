# Architecture

How the code is organized and how data flows through it. For the physics
that the code implements, see `docs/physics.md`.

## Package layout

The implementation lives in `src/dfxm_geo/`:

```
src/dfxm_geo/
├── constants.py             Physical & geometric constants
├── crystal/
│   ├── dislocations.py      Edge-dislocation displacement field
│   └── rotations.py         Rodrigues rotation + bulk 3x3 inverse
├── direct_space/
│   └── forward_model.py     The DFXM forward model (Find_Hg, forward)
├── reciprocal_space/
│   ├── resolution.py        Monte Carlo resolution function (Resq_i)
│   ├── kernel.py            Driver script: generates Resq_i npz
│   └── exposure.py          Auxiliary exposure-time model
├── io/
│   ├── __init__.py          check_folder
│   ├── images.py            save/load DFXM image stacks (.npy / .edf)
│   └── strain_cache.py      load_or_generate_Hg with .npy caching
├── analysis/
│   ├── moments.py           Image-stack moments / FWHM
│   ├── colormaps.py         Inverse pole-figure RGB mapping
│   └── mosaicity.py         compute_com_maps (Phase 9.2)
├── viz/
│   └── mosaicity.py         plot_mosaicity_maps, plot_qi_cross_section (Phase 9.2)
└── pipeline.py              Config dataclasses, run_simulation, run_postprocess, CLI
```

The top-level files `functions.py`, `image_processor.py`, `init_forward.py`,
`direct_space/forward_model.py`, `reciprocal_space/{recspace_res,
generate_Resq_i, exposure_time}.py` are kept as **deprecation shims** that
re-export from the new package. They will be removed once all known
consumers have migrated; see the cleanup plan for the schedule.

`pipeline.py` is the authoritative config-driven orchestrator. It exports the
dataclasses `CrystalConfig`, `ScanConfig`, `IOConfig`, `PostprocessConfig`, and
`SimulationConfig`, and the functions `run_simulation` and `run_postprocess`.
The CLI entry point `dfxm-forward` (configured in `pyproject.toml`) calls
`pipeline.cli_main`.

## Data flow for a simulation run

The end-to-end pipeline has five stages. Stages 1–3 are the forward simulation
(`run_simulation`); stages 4–5 are post-processing (`run_postprocess`).

1. **Generate the reciprocal-space resolution kernel `Resq_i`** (one-off,
   slow — ~10⁸ Monte Carlo rays):

   ```bash
   python -m dfxm_geo.reciprocal_space.kernel
   ```

   This writes `pkl_files/Resq_i_<timestamp>.npz` with the array and all
   generation parameters bundled into the single archive (no separate
   sidecar). The npz is picked up at import time by
   `_load_default_kernel()` whenever its filename matches `pkl_fn` in
   `forward_model.py`.

2. **Compute or load the displacement-gradient field `Hg`** for a given
   dislocation configuration. The first call generates Hg from the
   per-pixel `Fd_find` solution and caches it as
   `direct_space/deformation_gradient_tensors/Fg_<dis>_<psize_nm>nm_<zl_rms_nm>nm_px<Npixels>_sub<Nsub>_remount<name>.npy`
   — e.g. `Fg_4_40nm_64nm_px340_sub2_remountS1.npy`. The `_px<Npixels>_sub<Nsub>`
   key (added with the Round 15 shape-mismatch fix) and the `_remount<name>`
   suffix (added with the Round 18 sample-remount port) make the cache safe
   against grid-resolution and reference-frame collisions:

   ```python
   from dfxm_geo.direct_space.forward_model import Find_Hg
   Hg, q_hkl = Find_Hg(dis=4, ndis=151, psize=40e-9, zl_rms=0.15e-6 / 2.35)
   ```

3. **Run `forward()` for each (phi, chi) point** in the rocking scan via
   `dfxm_geo.io.images.save_images_parallel`. This parallelizes over a
   `ThreadPoolExecutor` and writes one `.npy` per (phi_step, chi_step) into
   `<output>/<io.dislocs_dirname>/`. If `io.include_perfect_crystal` is true,
   a second sweep with `Hg=0` is written to `<output>/<io.perfect_dirname>/`.

   The output image shape is `(NN2 // Nsub, NN1 // Nsub)`. With
   `qi_return=True`, `forward()` also returns the scattering-vector
   field `qi` for diagnostics.

4. **Compute mosaicity maps** (`analysis.mosaicity`, Phase 9.2). Reads both
   stacks back from disk via `load_images`, then extracts per-pixel
   center-of-mass positions over the (φ, χ) grid with `compute_com_maps`
   (on the nominal χ grid; the old runtime χ-offset calibration was
   dropped). Outputs `phi_list` and `chi_list` arrays (shape `(H, W)` each,
   in radians) plus a `chi_shift_rad` scalar (radians, now always 0.0),
   stored in the HDF5 analysis group.

5. **Render SVG figures** (`viz.mosaicity`, Phase 9.2). `plot_mosaicity_maps`
   produces the two-panel "extreme φ / χ" figure; `plot_qi_cross_section`
   produces the qi-field cross-section figure. Both are saved as SVG under
   `<output>/<io.figures_dirname>/`.

The full five-stage flow can be driven from the CLI:

```bash
dfxm-forward --config configs/default.toml --output output/
```

Use `--no-postprocess` to stop after stage 3, or `--postprocess-only` to
re-run stages 4–5 against an existing output directory (see "CLI entry point"
below).

## Output file format

Since v1.1.0, `dfxm-forward` writes a single BLISS-style HDF5 file (`dfxm_geo.h5`) per simulation, replacing the legacy `images10/` directory of per-frame `.npy` files. See [output-format.md](output-format.md) for the full schema.

Key properties:
- One file per `run_simulation` call; 1-2 BLISS scans inside (`/1.1` dislocations, `/2.1` optional perfect crystal).
- `/dfxm_geo/` root group embeds full provenance (git SHA, kernel hash, config TOML, machine, timestamps).
- Compatible with darfix and darling out of the box.
- Per-frame chunks + gzip-4 + shuffle compression: ~3-5× space savings vs raw `.npy`.

Legacy `.npy` output directories can be converted via `dfxm-migrate-output`.

## Module-level state

`dfxm_geo.direct_space.forward_model` is a stateful module: it builds the
detector grid, the rotation matrices `Ud`/`Us`/`Theta`, the coordinate
grid `rl`, and the beam profile `prob_z` at import time, all from the
constants near the top of the file. It then either auto-loads the default
Resq_i npz (if present on disk) via `_load_default_kernel()`, or
defers loading until the user calls `_load_default_kernel(pkl_path)`
explicitly.

**Why**: the original code base accumulated this state at module top
level because the same geometry is used across many calls. Treating it
as global lets callers say `forward(Hg, phi, chi)` without re-passing
everything. The cost is that the module is harder to test — see the
`tests/test_forward_model_smoke.py` regression guards for the safe-import
contract.

**How to override for a different beamline/geometry**: assign to the
module-level names directly *before* calling `forward`. For example:

```python
import dfxm_geo.direct_space.forward_model as fm
fm.Npixels = 256        # smaller detector
fm.psize = 20e-9        # finer pixels
# ... then rebuild the derived globals manually, or refactor in Phase 6.
```

A cleaner alternative (planned in Phase 6 of the cleanup) is a
`SimulationConfig` dataclass that the CLI loads from a TOML file and a
pipeline function that takes it as an argument.

### Sample-remount (goniometer) frame

`Fd_find` / `Fd_find_mixed` accept a sample-remount rotation matrix `S` as a
keyword-only argument (default identity). The rotation chain is:

    rs   = Theta · rl       (sample frame from lab)
    rgon = S.T · rs         (goniometer frame after sample remount)
    rc   = Us.T · rgon      (crystal frame)
    rd   = Ud.T · rc        (dislocation frame)

With `S = identity`, `rgon == rs` and the chain reduces to the original lab
→ sample → crystal → dislocation pipeline. The four named values `S1, S2, S3,
S4` (`dfxm_geo.crystal.remount`) are ported from the Purdue 2024 paper and
model "the same defect remounted in a symmetry-equivalent orientation."
Configure via `[crystal].sample_remount = "S1" | "S2" | "S3" | "S4"` in the
`dfxm-forward` TOML.

## Dependencies between modules

```
constants
   ^
   |
crystal.rotations -- crystal.dislocations
                              ^
                              |
io.strain_cache  -------------+
   ^                          |
   |                          |
direct_space.forward_model ---+
   ^
   |
io.images        ---  used by --> pipeline.run_simulation
analysis.moments                  pipeline.run_postprocess
analysis.colormaps                     ^             ^
analysis.mosaicity ────────────────────┤             │
viz.mosaicity      ────────────────────┘             │
                                                     │
                                             pipeline.cli_main
```

`reciprocal_space.{resolution, kernel, exposure}` interact with
`direct_space.forward_model` only through the npz file produced by
`kernel.py`. They have no Python-level imports of `direct_space`.

`analysis.mosaicity` depends only on `numpy` and `scipy.ndimage`; it has no
imports from `direct_space` or `io`. `viz.mosaicity` depends only on
`matplotlib`; it takes plain `numpy` arrays and a `pathlib.Path`.

## CLI entry point

`pipeline.cli_main` is registered as the `dfxm-forward` console script in
`pyproject.toml`. It accepts:

| Flag | Effect |
|---|---|
| _(none)_ | Default: run simulation then post-processing |
| `--no-postprocess` | Simulation only; skip stages 4–5 |
| `--postprocess-only` | Skip simulation; re-run stages 4–5 against an existing output dir |

Both `--no-postprocess` and `--postprocess-only` are mutually exclusive. When
`[postprocess].enabled = false` is set in the TOML config, post-processing is
also skipped even without `--no-postprocess`.

## Adding a new entry point

Most "what if I run a new variant?" questions become:

1. Write a new TOML config under `configs/variants/` and run
   `dfxm-forward --config configs/variants/your_variant.toml --output output/`.
2. For a fully scripted workflow, write a `scripts/your_run.py` that imports
   `dfxm_geo.pipeline.run_simulation` (or `run_postprocess`) directly.
3. If your run needs different physics (e.g., screw dislocations,
   different Bragg reflection), parameterize `Find_Hg` with the relevant
   arguments rather than editing module state.

## Adding a new analysis function

Add a new function under `dfxm_geo/analysis/`. The pattern is: take a
NumPy array (image or stack), return another NumPy array (or a dict of
named summary statistics). Avoid file I/O — that belongs in
`dfxm_geo/io/`.

See `analysis/mosaicity.py` for the Phase 9.2 example: `compute_com_maps`
takes a stack and scalar grid parameters and returns two `(H, W)` maps.

## Adding a new viz function

Add a new function under `dfxm_geo/viz/`. The pattern is:

```python
def plot_<name>(data: np.ndarray, ..., out_path: Path, **kwargs) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display required
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(...)
    # ... render
    fig.savefig(out_path)
    plt.close(fig)
```

Viz functions should accept plain NumPy arrays and write to disk via a
`Path`; they must not call `plt.show()` or hold figure state. Color limits
and other tuning knobs should be surfaced as keyword arguments with
documented defaults.

## Adding a new I/O format

Add a new function under `dfxm_geo/io/`. The pattern is:

```python
def save_<format>(stack: np.ndarray, fpath: str, ...) -> bool:
    ...

def load_<format>(fpath: str, ...) -> np.ndarray:
    ...
```

If the load/save needs an external library (fabio, h5py, …) add it as an
optional dependency in `pyproject.toml`.

## Tests

The test suite lives in `tests/` and runs with pytest. Two kinds of tests:

- **Smoke tests** (`tests/test_*_smoke.py`): pin numerical output of the
  most-fragile functions (`Fd_find`, `forward`) so refactors can't
  silently break the physics. Goldens live in `tests/data/golden/`.
- **Unit tests** (`tests/test_*.py`): focused tests of the new module
  APIs — orthogonality, shape invariants, round-trips, expected return
  keys, far-field decay, etc.

Run the full suite:

```bash
pytest                                 # 35 tests, ~30 s
pytest --cov=dfxm_geo                  # with coverage
pytest tests/test_rotations.py -v      # one file
```

`pytest-benchmark` is available for performance regression tests; none
are wired up yet — see Phase 8 of the cleanup plan.
