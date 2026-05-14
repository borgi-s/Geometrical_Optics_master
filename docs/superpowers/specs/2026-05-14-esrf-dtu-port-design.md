# ESRF_DTU Port (Design)

**Date:** 2026-05-14
**Branch under port:** `origin/ESRF_DTU`
**Target:** `cleanup/main-modernization` (HEAD `be9f79d` at port start)
**Status:** Brainstorming complete; awaiting user review of this spec.

**Related work:**
- Round 16 (HEAD `be9f79d`): ported `origin/dislocation_identification` as `dfxm-identify` with `mode={single, multi}`. This port extends `dfxm-identify` with a third mode (`z-scan`) plus a new sample-coordinate visualisation.
- Spec for Round 16: `docs/superpowers/specs/2026-05-14-dislocation-identification-port-design.md`.

## Goal

Port the canonical content of `origin/ESRF_DTU` into the cleanup. The biggest addition is a **4D scan mode** (depth × Burgers × line-direction × rocking-curve) used at ESRF in 2024 for 3D dislocation mapping, plus several lower-level helpers (rotation utilities, z-shift grid generator, crystal-in-lab-frame 3D viz).

## Non-goals

- **Not** porting `Fd_find_edge` or `Fd_find_screw` as standalone physics functions. Both collapse to `Fd_find_mixed` at `rotation_deg=0` (pure edge) and `rotation_deg=90` (pure screw) respectively. Per user call (2026-05-14), document the equivalence in `Fd_find_mixed`'s docstring instead of adding redundant API.
- **Not** porting the branch's `disloc_identify.py` as-is. Same reason as Round 16: hardcoded Windows paths, `from X import *` patterns, no test coverage, no config. The `save_scan` method's intent is what we're capturing in the new pipeline mode.
- **Not** porting the branch's `Fd_find_mixed` and `Fd_find_multi_dislocs_mixed` (already landed in Round 16). The branch's variants are minor rewrites of the same math.
- **Not** porting the branch's `Fdd[1,1]` sign correction commit (`42be417`). Same fix already on cleanup as `3b71b33` (Round 7 Appendix-A fix).
- **Not** porting `load_edfs` (the branch added an EDF loader to `functions.py` that we don't have evidence anyone uses; `dfxm_geo.io.images` already has `load_image`/`load_images` for `.npy` and `.edf` mixed loaders).
- **Not** porting `check_path` (a duplicate of `dfxm_geo.io.check_folder`, same as `dislocation_identification`).

## What gets ported

| Branch artefact | Lands as | Notes |
|---|---|---|
| `Z_shift(xl_start, yl_start, zl_start, NN1, NN2, NN3, offset)` (functions.py) | `dfxm_geo.direct_space.forward_model.Z_shift(offset_um)` | Returns the module's `rl` grid shifted along the z axis by `offset_um` µm. The signature simplifies because the cleanup's `forward_model` already owns `xl_start`, `yl_start`, `zl_start`, `NN1`, `NN2`, `NN3` at module level. |
| `rotate_matrix_z_axis(matrix, angle_degrees)` (functions.py) | `dfxm_geo.crystal.rotations.rotate_matrix_z_axis` | Pure helper: left-multiply by Z-axis rotation matrix. |
| `is_valid_rotation_matrix(R)` (functions.py; commit `1024cd9`) | `dfxm_geo.crystal.rotations.is_valid_rotation_matrix` | Sanity check: `det(R) ≈ 1` and `R · Rᵀ ≈ I` within tolerance. Optional kwarg `atol=1e-6`. |
| `save_scan` (disloc_identify.py — the 4D loop) | `dfxm_geo.pipeline._run_identification_zscan` + `mode="z-scan"` + `IdentificationZScanConfig` | New pipeline mode. See output-layout section below. Random secondary dislocation per (z, b, α) configuration, seeded for reproducibility. |
| `plot_sample.py` (top-level, 201 lines) | `dfxm_geo.viz.sample.plot_crystal_in_lab` | Matplotlib 3D viz of the crystal cube in lab coordinates with axis arrows. Pure viz; matplotlib is already a required dep so no lazy import needed. |

## Architecture

User-approved approach (mirrors Round 16's pattern): third Literal value in `IdentificationConfig.mode`, third sub-dataclass `IdentificationZScanConfig`, third internal runner `_run_identification_zscan`.

### Additions to existing modules

**`src/dfxm_geo/crystal/rotations.py`** — append:

```python
def rotate_matrix_z_axis(matrix: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate `matrix` by `angle_degrees` around the lab z axis.

    Returns ``R_z(angle) @ matrix``.
    """

def is_valid_rotation_matrix(R: np.ndarray, *, atol: float = 1e-6) -> bool:
    """Check that R is a proper rotation matrix.

    Validates ``det(R) ≈ 1`` and ``R @ R.T ≈ I`` within ``atol``. Used by the
    z-scan pipeline to fail early on a malformed ``Us`` rather than producing
    garbage Fg arrays.
    """
```

**`src/dfxm_geo/direct_space/forward_model.py`** — append a single function (does not modify the existing module-level state):

```python
def Z_shift(offset_um: float) -> np.ndarray:
    """Return an `rl` grid shifted along the z axis by ``offset_um`` µm.

    Uses the module's existing detector ray-grid parameters (xl_start, NN1,
    etc.) to build the same mgrid as `rl` does at import time, but with the
    z range translated by ``-offset_um × 1e-6`` m. The original module-level
    ``rl`` is not modified.

    Returns:
        (3, X) coordinates, units of metres (matching the lab-frame
        convention used by `Fd_find` and `Fd_find_mixed`).
    """
```

**`src/dfxm_geo/crystal/dislocations.py`** — update the `Fd_find_mixed` docstring with a note that pure-edge and pure-screw are special cases (no code change):

> **Convenience equivalences:** `Fd_find_mixed(..., rotation_deg=0)` is the pure-edge field (equivalent to `Fd_find(..., ndis=1)` with matching Ud); `Fd_find_mixed(..., rotation_deg=90)` is the pure-screw field. We don't ship separate `Fd_find_edge` / `Fd_find_screw` wrappers — call `Fd_find_mixed` with the right `rotation_deg` instead.

### New module

**`src/dfxm_geo/viz/sample.py`** — port of `plot_sample.py`:

```python
def euler_matrix(angles_deg: tuple[float, float, float], order: str = "xyz") -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (degrees)."""

def plot_crystal_in_lab(
    sample_to_lab_R: np.ndarray = ...,
    *,
    side: float = 1.2,
    show_axes: bool = True,
) -> "matplotlib.figure.Figure":
    """Return a matplotlib Figure showing the sample cube in lab coords.

    Caller decides whether to `.show()` or `.savefig(path)`.
    """
```

The internals stay close to `plot_sample.py`: a cube via `Poly3DCollection`, axis arrows, optional rotation. We unify the multiple helper functions in the branch source into one `plot_crystal_in_lab` public entry point + module-private helpers (`_draw_cube`, `_unit`, `_axes`).

### Pipeline extensions

**`src/dfxm_geo/pipeline.py`** — append:

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    """z-scan mode parameters (mode='z-scan' only).

    Each (z_layer, b, α) configuration produces a (phi_steps × chi_steps)
    rocking-curve stack, with a randomly-drawn secondary dislocation if
    `include_secondary` is True.
    """

    z_offsets_um: list[float]                  # e.g. [-2.0, -1.0, 0.0, 1.0, 2.0]
    phi_range_deg: float                       # half-range, degrees
    phi_steps: int
    chi_range_deg: float
    chi_steps: int
    include_secondary: bool = True             # toggle random secondary dislocation
    secondary_rng_offset: int = 1              # spawn index for secondary stream
```

**`IdentificationConfig` changes:**

- `mode` Literal extends to `Literal["single", "multi", "z-scan"]`.
- Add `zscan: IdentificationZScanConfig | None = None`.
- `__post_init__` adds:
  - `mode == "z-scan"` requires `zscan is not None`.
  - `mode == "single"` / `"multi"` require `zscan is None`.

**`_run_identification_zscan(config, output_dir) -> dict[str, Any]`:**

Pseudocode:

```python
master = np.random.default_rng(scan_cfg.rng_seed)
secondary_rng = master.spawn(zscan_cfg.secondary_rng_offset + 1)[zscan_cfg.secondary_rng_offset]
noise_rng = ... # unused for z-scan since rocking-curve images are saved via save_images_parallel which is deterministic

for k, z_off in enumerate(zscan_cfg.z_offsets_um):
    rl = fm.Z_shift(z_off)
    layer_dir = output_dir / f"layer_{k:04d}"

    for plane in planes:                                 # canonical 4 {111} or just config.crystal.slip_plane_normal
        for j, b_idx in enumerate(b_indices):
            if exclude_invisibility and |G·b| ≈ 0: continue
            for i, alpha in enumerate(angles_deg):
                # Random secondary: same secondary across the rocking grid for this (k, b, α)
                if zscan_cfg.include_secondary:
                    sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)  # core at origin
                    specs = [primary, secondary]
                    Fg = Fd_find_multi_dislocs_mixed(rl, fm.Us, specs, fm.Theta)
                else:
                    Fg = Fd_find_mixed(rl, fm.Us, Ud_primary, alpha, fm.Theta)

                Hg = transpose(fast_inverse2(Fg), [0, 2, 1]) - I
                fm.Hg = Hg
                config_dir = layer_dir / f"n_{plane_slug}" / f"b{b_idx}_alpha{int(round(alpha)):03d}"
                save_images_parallel(Hg, phi_range_deg, phi_steps, chi_range_deg, chi_steps,
                                     str(config_dir), io.fn_prefix, io.ftype)

                manifest_rows.append({ ... })
```

`save_images_parallel` is the existing dfxm-forward routine — it writes a phi×chi grid of `mosa_test_0000_NNNN.npy` files to the destination directory. We reuse it directly for the rocking-curve writes, so output is byte-for-byte interoperable with everything that consumes dfxm-forward output (postprocess pipeline, comparison scripts, etc.).

The dispatcher `run_identification` adds the third branch. `cli_main_identify` already supports `--mode {single,multi}`; we extend the `choices` list to include `"z-scan"`.

### CLI behaviour

```
dfxm-identify --config configs/identification_zscan.toml --output output/identify_zscan
```

Single argparse choice extension; no new flags.

## Output layout

```
<output>/
├── manifest.csv
└── layer_{0001}/
    └── n_{slip_plane_slug}/
        └── b{b_idx}_alpha{deg:03d}/
            ├── mosa_test_0000_0000.npy    # phi/chi grid via save_images_parallel
            ├── mosa_test_0000_0001.npy
            └── ...
```

- `layer_{NNNN}` is zero-padded; index matches `z_offsets_um[k]`.
- `n_{slip_plane_slug}` uses the same `1_1_1` / `1_m1_1` convention as single mode.
- `b{idx}_alpha{deg}` mirrors single mode's stem.
- Inside each config directory, the phi/chi grid is written by `save_images_parallel` — the file naming and shape are identical to the legacy `init_forward.py` and to `dfxm-forward` output, so existing tooling that consumes those works unchanged.

**`manifest.csv` schema (one row per (layer, b, α) configuration that passed the invisibility filter):**

| Column | Description |
|---|---|
| `layer` | Zero-padded integer index into `z_offsets_um` |
| `z_offset_um` | The actual offset in µm |
| `n_h`, `n_k`, `n_l` | Primary slip plane normal |
| `b_idx` | Burgers vector index ∈ {0..5} for the primary |
| `b_h`, `b_k`, `b_l` | Primary Burgers vector (resolved Miller indices) |
| `alpha_deg` | Primary rotation angle (degrees) |
| `secondary_present` | `True` if include_secondary; `False` otherwise |
| `secondary_n_h/k/l` | Secondary slip plane normal (NaN if not present) |
| `secondary_b_idx` | Secondary Burgers vector index |
| `secondary_b_h/k/l` | Secondary Burgers vector (resolved Miller indices) |
| `secondary_alpha_deg` | Secondary rotation angle |
| `path` | Relative path to the configuration's directory |

## Tests

| File | Purpose | Approx count |
|---|---|---|
| `tests/test_rotations.py` (extend) | `rotate_matrix_z_axis` produces correct rotation (90° rotation of identity → permutation), `is_valid_rotation_matrix` accepts identity / scipy `Rotation.random()`, rejects scaled-identity, non-orthogonal. | 4 |
| `tests/test_forward_model_smoke.py` (extend) | `Z_shift(0.0)` matches module-level `rl`; `Z_shift(5.0)` shifts the z column by 5 µm and leaves x, y unchanged. | 2 |
| `tests/test_viz_sample.py` (NEW) | `plot_crystal_in_lab` returns a `matplotlib.figure.Figure`; `euler_matrix` is orthonormal and `det=1` for representative angle sets. | 3 |
| `tests/test_pipeline_identification.py` (extend) | `IdentificationZScanConfig` defaults; `mode='z-scan'` requires zscan block; tiny z-scan run (1 layer × 1 b × 1 α × 2 phi × 2 chi = 4 images via stubbed `save_images_parallel`); determinism on `rng_seed`; manifest schema. | 6 |

Expected total: ~15 new tests. Suite grows from 163 → ~178.

## Public API additions

- `dfxm_geo.crystal.rotations.rotate_matrix_z_axis`
- `dfxm_geo.crystal.rotations.is_valid_rotation_matrix`
- `dfxm_geo.direct_space.forward_model.Z_shift`
- `dfxm_geo.viz.sample.plot_crystal_in_lab`
- `dfxm_geo.viz.sample.euler_matrix`
- `dfxm_geo.pipeline.IdentificationZScanConfig`
- `dfxm_geo.pipeline._run_identification_zscan` (module-private but documented for advanced callers)
- Extension of `IdentificationConfig.mode` to include `"z-scan"`
- Extension of `cli_main_identify` `--mode` choices to include `"z-scan"`

## Dependencies

No new required or optional deps. Matplotlib (used by `viz/sample.py`) is already a required dep. The `viz/burgers.py` plotly path from Round 16 remains the only optional viz dep.

## Configs

`configs/identification_zscan.toml` (sane defaults for a single z-stack at ID06).

**Before a real run:** flip `Nsub = 2 → 1` in `src/dfxm_geo/direct_space/forward_model.py:42` for ~8× faster per-image forward calls. `Nsub = 1` is the typical real-run choice; the cleanup default of 2 is preserved as the publication-quality setting from Borgi 2024/2025 (see Risk #8).

```toml
mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan]
phi_rad = 1.5e-4                   # unused in z-scan but IdentificationScanConfig requires
poisson_noise = false              # noise lives in the rocking-curve forward calls
rng_seed = 0
intensity_scale = 7.0

[zscan]
z_offsets_um = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 depth slices
phi_range_deg = 0.034377467707849395         # matches dfxm-forward default (0.0006 rad)
phi_steps = 21                               # downscaled for the example; bump to 61 for a real run
chi_range_deg = 0.11459155902616465
chi_steps = 21
include_secondary = true
secondary_rng_offset = 1

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

## Risks / open questions

1. **Reflection still fixed at the forward_model module level.** Same Risk #6 from Round 16. `Z_shift` reuses the module-level geometry, so `z-scan` mode inherits the same `(-1, 1, -1)` reflection lock. Documented; not blocking.
2. **`save_images_parallel` thread-pool contention with z-scan loop.** The existing routine spawns its own ThreadPoolExecutor for the phi/chi grid. We're calling it inside a single-threaded outer loop, so no contention — but a future "parallelise the outer loop" optimisation would need care.
3. **Per-configuration random secondary** (user-confirmed choice): one secondary is drawn for the entire phi/chi rocking curve at that (z, b, α). The phi/chi sweep is deterministic from the resulting Hg; the secondary changes between configurations (not between rocking-curve steps). Same seed → identical secondaries across reruns.
4. **g·b filter** is shared with single mode's invisibility filter from Round 16. We reuse `_passes_invisibility`. No new physics.
5. **`Z_shift` shadows the existing module-level `rl`.** Each call returns a fresh array; the module's `rl` is unchanged. Callers that do `fm.rl = Z_shift(...)` are mutating module state — acceptable here because z-scan is the only place we do it, but worth noting in the docstring.
6. **`euler_matrix` is not a deep abstraction.** It's used once in `viz/sample.py` for drawing the rotated crystal. Don't promote it to `dfxm_geo.crystal.rotations` unless a second caller appears.
7. **Storage size of a real z-scan run.** 5 layers × 4 planes × 6 b × 36 α × 61 × 61 ≈ 16 M `.npy` files; multi-GB. We don't enforce a max; the user is on the hook for `--output` cleanup. Worth a one-line warning in `_run_identification_zscan` printing the projected image count after the invisibility filter so the user can sanity-check before kicking off.
8. **Nsub = 1 is the typical real-run choice** (per user, 2026-05-14). The cleanup's default `Nsub = 2` is preserved as the publication-quality setting (matches Borgi 2024 / Borgi 2025), but day-to-day workflows and ESRF z-scans almost always run at `Nsub = 1` for ~8× faster per-image forward calls. The example `configs/identification_zscan.toml` doesn't change `Nsub` (it's a module-level constant, not config-driven — see Risk #1), so the user is expected to manually flip `Nsub = 1` in `src/dfxm_geo/direct_space/forward_model.py` before a real run. Document this in the example config's preamble. The eventual runtime-configurable-reflection refactor (Risk #1) should also make `Nsub` config-driven; same blocker.
