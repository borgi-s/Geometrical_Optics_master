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
│   ├── kernel.py            Driver script: generates Resq_i pickle
│   └── exposure.py          Auxiliary exposure-time model
├── io/
│   ├── __init__.py          check_folder
│   ├── images.py            save/load DFXM image stacks (.npy / .edf)
│   └── strain_cache.py      load_or_generate_Hg with .npy caching
├── analysis/
│   ├── moments.py           Image-stack moments / FWHM
│   └── colormaps.py         Inverse pole-figure RGB mapping
└── viz/                     (reserved — currently empty)
```

The top-level files `functions.py`, `image_processor.py`, `init_forward.py`,
`direct_space/forward_model.py`, `reciprocal_space/{recspace_res,
generate_Resq_i, exposure_time}.py` are kept as **deprecation shims** that
re-export from the new package. They will be removed once all known
consumers have migrated; see the cleanup plan for the schedule.

## Data flow for a simulation run

The end-to-end forward simulation has three stages:

1. **Generate the reciprocal-space resolution kernel `Resq_i`** (one-off,
   slow — ~10⁸ Monte Carlo rays):

   ```bash
   python -m dfxm_geo.reciprocal_space.kernel
   ```

   This writes `pkl_files/Resq_i_<timestamp>.pkl` plus a sidecar
   `_vars.txt` describing the parameters used. The pickle is picked up at
   import time by `dfxm_geo.direct_space.forward_model` if its filename
   matches `pkl_fn` there.

2. **Compute or load the displacement-gradient field `Hg`** for a given
   dislocation configuration. The first call generates Hg from the
   per-pixel `Fd_find` solution and caches it as
   `direct_space/deformation_gradient_tensors/Fg_<dis>_<psize>nm_<zl_rms>nm.npy`:

   ```python
   from dfxm_geo.direct_space.forward_model import Find_Hg
   Hg, q_hkl = Find_Hg(dis=4, ndis=151, psize=40e-9, zl_rms=0.15e-6 / 2.35)
   ```

3. **Run `forward()` for each (phi, chi) point** in the rocking scan:

   ```python
   from dfxm_geo.direct_space.forward_model import forward
   image = forward(Hg, phi=0.0, chi=0.0)
   ```

   The output is a 2D image of shape `(NN2 // Nsub, NN1 // Nsub)`. With
   `qi_return=True`, the function also returns the scattering-vector
   field `qi` for diagnostics.

For a sweep over (phi, chi), the convenience function
`dfxm_geo.io.images.save_images_parallel` parallelizes step 3 over a
`ThreadPoolExecutor` and writes each image to disk.

## Module-level state

`dfxm_geo.direct_space.forward_model` is a stateful module: it builds the
detector grid, the rotation matrices `Ud`/`Us`/`Theta`, the coordinate
grid `rl`, and the beam profile `prob_z` at import time, all from the
constants near the top of the file. It then either auto-loads the default
Resq_i pickle (if present on disk) via `_load_default_kernel()`, or
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
io.images        ---  used by --> init_forward.py / scripts
analysis.moments
analysis.colormaps
```

`reciprocal_space.{resolution, kernel, exposure}` interact with
`direct_space.forward_model` only through the pickle file produced by
`kernel.py`. They have no Python-level imports of `direct_space`.

## Adding a new entry point

Most "what if I run a new variant?" questions become:

1. Write a new top-level script (recommended: `scripts/your_run.py`)
   that imports `dfxm_geo.direct_space.forward_model.Find_Hg` and
   `forward`, builds an `Hg` for the configuration you care about, and
   iterates `forward()` over your scan.
2. Drop output paths into `<DFXM_DATA_DIR>/...` so other users get
   sensible defaults — see `init_forward.py` for the pattern.
3. If your run needs different physics (e.g., screw dislocations,
   different Bragg reflection), parameterize `Find_Hg` with the relevant
   arguments rather than editing module state.

## Adding a new analysis function

Add a new function under `dfxm_geo/analysis/`. The pattern is: take a
NumPy array (image or stack), return another NumPy array (or a dict of
named summary statistics). Avoid file I/O — that belongs in
`dfxm_geo/io/`.

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
