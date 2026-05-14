# Dislocation Identification Port (Design)

**Date:** 2026-05-14
**Branch under port:** `origin/dislocation_identification`
**Target:** `cleanup/main-modernization`
**Status:** Brainstorming complete; awaiting user review of this spec.

**Canonical reference:** Borgi, S., Winther, G., Poulsen, H. F. (2025). "Individual dislocation identification in dark-field X-ray microscopy". *J. Appl. Cryst.* **58**, 813–821. DOI: `10.1107/S1600576725002614`. All physics equations cited below refer to this paper.

## Goal

Port the canonical content of `origin/dislocation_identification` into the cleanup branch as a first-class workflow `dfxm-identify`, mirroring the structure of `dfxm-forward`. The pipeline supports two modes:

- **Single-dislocation** — deterministic Cartesian sweep over Burgers vectors × rotation angles (per slip plane). **This is the methodology validated in Borgi 2025**: the published test set is 4 slip planes × 6 Burgers vectors × 36 line-direction angles = **864 images** (840 after excluding 24 near-invisibility cases, G·b within 10°). This mode reproduces that exact sweep.
- **Multi-dislocation** — Monte Carlo. Each sample image contains two mixed-character dislocations with independently-drawn random (slip plane, Burgers vector, rotation angle, in-plane position). Produces an ML-training-shaped dataset with per-image ground-truth labels. **This mode is a forward-looking extension beyond the published paper**; Borgi 2025 discusses ML training sets of "tens of thousands of images" as motivation but does not present multi-dislocation simulations. The mathematics (Eq. 1 sum over multiple dislocations + identity once) is a direct generalization of the published single-dislocation formula.

## Non-goals

- **Not** porting the legacy `disloc_identify.py` script as-is. Its hardcoded Windows path, `from X import *` patterns, and one-off forward-model edits (zl_rms 0.6 µm, Nsub 1, ndis 1, commented-out Hg auto-load) are paper-specific configuration leak, not canonical functionality.
- **Not** porting `check_path` (functions.py:191 on the branch) — it duplicates `dfxm_geo.io.check_folder`.
- **Not** porting the Appendix-A sign change from the branch — it is already in cleanup as commit `3b71b33`.
- **Not** porting the `requirements.txt` version pins on the branch — superseded by `pyproject.toml`.
- **Not** porting `create_video_from_png` (image_processor.py on the branch). Adds an `imageio` dependency for ~30 lines of glue around `ffmpeg`-equivalents. Punt; the user can `ffmpeg` over a PNG dir if needed.
- **Not** porting `Fd_find_mixed`'s misorientation/t_vec dead-args (the cleanup already removed the misorientation branch from `Fd_find` in `ddeaa06`; the new mixed functions get the same treatment).
- **Phi-sweep variant of multi-disloc** — each multi-disloc sample is one image at one phi. The future "rocking-curve per sample" extension (sample = phi-stack of images) slots in by replacing the float `phi` with a list/range in the TOML and looping inside the per-sample function. Out of scope for this port.

## What gets ported

| Branch artifact | Lands as | Notes |
|---|---|---|
| `Fd_find_mixed` (functions.py) | `dfxm_geo.crystal.dislocations.Fd_find_mixed` | Single mixed-character dislocation (Eq. 1 of Borgi 2025). Adds position-offset kwarg (lab frame); renames `a_deg`→`rotation_deg` with docstring documenting the 90° relation to the paper's α. |
| `Fd_find_multi_dislocs_mixed` (functions.py) | `dfxm_geo.crystal.dislocations.Fd_find_multi_dislocs_mixed` | Sums two mixed contributions. Refactor: both crystals via the same helper, no `_Fdd_no_I` closure. |
| `BurgersVectorsPlotter.find_b_vectors` | `dfxm_geo.crystal.burgers.burgers_vectors(slip_plane_normal)` | The `{111}` lookup table, pure function. |
| `BurgersVectorsPlotter.calculate_rotated_vectors` | `dfxm_geo.crystal.burgers.rotated_t_vectors(...)` | scipy `Rotation.from_rotvec` wrapped. |
| `BurgersVectorsPlotter.calculate_ud_matrices` | `dfxm_geo.crystal.burgers.ud_matrices(...)` | Build (rotated_t, n, b) → Ud frames. |
| `BurgersVectorsPlotter.plot_vectors` | `dfxm_geo.viz.burgers.plot_slip_plane_3d(...)` | Plotly figure (returns `go.Figure`, does NOT call `.show()`). |
| `BurgersVectorsPlotter.plot_images` (sweep loop) | `dfxm_geo.pipeline.run_identification(config, output_dir)` | Becomes the single-disloc Cartesian-sweep branch of the new entrypoint. |

## Architecture

Approach approved by user (Option 3 from brainstorming): separate entrypoint `dfxm-identify`, shared `pipeline.py` module.

### New modules

**`src/dfxm_geo/crystal/burgers.py`** (~80 lines):

```python
def burgers_vectors(slip_plane_normal: tuple[int, int, int]) -> np.ndarray:
    """Return the 6 Burgers vectors associated with a {111}-family slip plane.

    Returns: array of shape (6, 3) — three vectors and their negatives,
    normalised to magnitude 1/sqrt(2) in the cubic crystal frame.

    Raises ValueError if slip_plane_normal is not one of the four {111} variants.
    """

def rotated_t_vectors(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """For each (angle, burgers) pair, rotate the in-plane t-vector b × n.

    Returns: array of shape (n_angles, n_burgers, 3).
    """

def ud_matrices(
    slip_plane_normal: np.ndarray,
    rotated_vectors: np.ndarray,
) -> np.ndarray:
    """Construct Ud rotation matrices from (rotated_t, n, b) basis frames.

    Returns: array of shape (n_angles, n_burgers, 3, 3).
    """
```

All three are pure-numpy + scipy.spatial.transform; no I/O, no plotly, fully testable in isolation.

**`src/dfxm_geo/viz/burgers.py`** (~50 lines):

```python
def plot_slip_plane_3d(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    rotated_vectors: np.ndarray,
) -> "plotly.graph_objects.Figure":
    """Return an interactive 3D figure showing the slip plane + b-vectors + rotated t-vectors.

    Caller decides whether to .show() or .write_html(path).
    Imports plotly lazily so the package imports cleanly without the optional dep.
    """
```

Lazy `import plotly.graph_objects as go` inside the function; if missing, raise `RuntimeError` with the install hint (`pip install dfxm-geo[identification]`). Mirrors the `xraylib` pattern from the CDD_inc port.

**`tests/test_burgers.py`** — geometry unit tests (~6 tests). Coverage target 100%.

**`tests/test_dislocations_mixed.py`** — physics unit tests for `Fd_find_mixed` and `Fd_find_multi_dislocs_mixed`. Pure-edge limit `a=0°` must match `Fd_find(ndis=1)` (with matching Ud); pure-screw limit `a=90°` produces the expected screw-only field pattern; superposition linearity for the multi version; numerical golden against a fixture generated from the branch source.

**`tests/test_pipeline_identification.py`** — config round-trip, smoke run for single-disloc mode (tiny `n_angles=2, n_burgers=2`), smoke run for multi-disloc mode (`n_samples=4, seed=0`), manifest-CSV schema.

### Extensions to `dfxm_geo/crystal/dislocations.py`

Add two new top-level functions, written from scratch to match the cleanup's style (type hints, NumPy docstrings, constants from `dfxm_geo.constants`). The math implements Eq. 1 of Borgi 2025; the angle parameter naming and the convention difference between the branch source and the paper are documented in the docstrings.

```python
def Fd_find_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    Ud_mix: np.ndarray,
    rotation_deg: float,
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    position_lab_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Displacement gradient Fg for a single mixed-character dislocation.

    Implements Eq. 1 of Borgi, Winther & Poulsen (2025), J. Appl. Cryst. 58,
    813-821, doi:10.1107/S1600576725002614:

        F_d = I + screw_matrix * cos(α) + edge_matrix * sin(α)

    where α is the angle between the Burgers vector b and the dislocation line
    direction t (α=0/180° pure screw; α=90/270° pure edge).

    Parameterization note (important — differs from paper):
        ``rotation_deg`` is NOT the paper's α. It is the angle (in degrees) by
        which the dislocation line direction t has been rotated around the
        slip-plane normal n, starting from the initial in-plane reference
        ``t_0 = b × n`` (which itself has α=90°, i.e. pure edge). The two
        parameterizations satisfy ``α_paper = 90° - rotation_deg`` (modulo
        sign conventions of the rotation axis), so:

            rotation_deg = 0   ⇔ α_paper = 90°  (pure edge)
            rotation_deg = 90° ⇔ α_paper = 0°   (pure screw)

        The numerical result of this function is identical to ``F_d`` in Eq. 1
        evaluated at α = 90° - rotation_deg, with cos/sin contributions
        rearranged accordingly. The naming follows the branch source code
        rather than the paper to keep callers (which iterate ``rotation_deg``
        over [0, 360) in the deterministic single-disloc sweep) unchanged.

    Args:
        rl: Lab-frame coordinates, shape (3, X).
        Us, Ud_mix, Theta: rotation matrices per Eqs. 3, 5, 7 of Borgi 2025.
        rotation_deg: see parameterization note above.
        b: Burgers vector magnitude (default `BURGERS_VECTOR` from constants).
        ny: Poisson ratio (default `POISSON_RATIO` from constants).
        position_lab_um: shift applied to rl (in the lab frame, µm) so the
            dislocation core sits at the offset rather than the origin.
            Default (0, 0, 0) matches the legacy behavior.

    Returns:
        Fg of shape (X, 3, 3) in the grain frame, with the identity already
        added. Compose with `fast_inverse2` + transpose to get Hg.
    """

def Fd_find_multi_dislocs_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    crystals: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
    """Sum of mixed-dislocation contributions from N crystals.

    Generalizes Eq. 1 of Borgi 2025 to multiple dislocations in the same
    sample volume: each crystal's screw+edge contributions are summed (NO
    identity added per crystal), and the identity is added once at the end.
    For N=1 this reduces to ``Fd_find_mixed``; for N=2 it is the case used
    by the multi-disloc Monte Carlo pipeline mode (extension beyond the
    published paper).

    Each MixedDislocSpec carries (Ud_mix, rotation_deg, position_lab_um).
    """
```

Where `MixedDislocSpec` is a small `@dataclass(frozen=True)` with `Ud_mix: np.ndarray`, `rotation_deg: float`, `position_lab_um: tuple[float, float, float] = (0, 0, 0)`.

The math itself is a port of the branch's `Fd_find_mixed` and `Fd_find_multi_dislocs_mixed`, except:
- The branch's `denom1 = sqz + sqy` term — preserved as-is. This is the screw-out-of-plane denominator implied by ∂u_d,x/∂z; not a "fix" to attempt.
- Drop the unused `dis`, `ndis`, `misorientation`, `t_vec` kwargs (these are dead in the mixed functions on the branch).
- Use `BURGERS_VECTOR` and `POISSON_RATIO` from `dfxm_geo.constants` as defaults instead of hardcoded literals.
- Add the position offset (new functionality required by multi-disloc Monte Carlo mode).
- Rename `a_deg` → `rotation_deg` with explicit docstring relating the parameter to the paper's α (per the convention note above). The branch code's `a` collides confusingly with the paper's `α`; the rename removes the trap for future readers.

### Extensions to `dfxm_geo/pipeline.py`

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationCrystalConfig:
    slip_plane_normal: tuple[int, int, int]  # ignored in mode="multi"; required in mode="single"
    angle_start_deg: float = 0.0
    angle_stop_deg: float = 350.0
    angle_step_deg: float = 10.0
    b_vector_indices: list[int] | None = None  # None = all 6
    # Single-disloc mode only: sweep all 4 slip planes (matches Borgi 2025
    # 864-image test set) or just `slip_plane_normal`.
    sweep_all_slip_planes: bool = True
    # Exclude configurations near the invisibility criterion |G·b| < cos(80°),
    # matching Borgi 2025's 24-image exclusion that brings 864 → 840.
    exclude_invisibility: bool = True
    invisibility_threshold_deg: float = 10.0  # G·b within 10° of orthogonal

@dataclass(frozen=True, kw_only=True)
class IdentificationScanConfig:
    phi_rad: float = 150e-6
    poisson_noise: bool = True
    rng_seed: int = 0
    intensity_scale: float = 7.0  # the `* 7` from legacy

@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    n_samples: int = 1000
    pos_std_um: float = 5.0
    n_png_previews: int = 50  # only write PNGs for first N samples

@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    mode: Literal["single", "multi"]
    crystal: IdentificationCrystalConfig  # used by both modes
    scan: IdentificationScanConfig
    multi: IdentificationMonteCarloConfig | None = None  # required if mode="multi"
    io: IOConfig  # reuse from existing pipeline

def run_identification(config: IdentificationConfig, output_dir: Path) -> dict:
    """Dispatch to _run_identification_single or _run_identification_multi."""

def cli_main_identify(argv: list[str] | None = None) -> int:
    """argparse: --config, --output, --mode {single,multi}."""
```

Validation in `IdentificationConfig.__post_init__`:
- If `mode == "multi"` then `multi is not None`.
- If `mode == "single"` then `multi is None` (warn but allow — easier to share configs).
- `slip_plane_normal` must be one of the four `{111}` variants.

### CLI behavior

```
dfxm-identify --config configs/identification_single.toml --output output/identify_single
dfxm-identify --config configs/identification_multi.toml --output output/identify_multi
```

`--mode` overrides the TOML `mode` field if given on CLI. Like `dfxm-forward`, `--config` and `--output` are required.

## Output layout

### Single-disloc mode

```
<output>/
├── manifest.csv                          # rows: (sample_id, slip_plane, b_idx, b_vec, alpha_deg, image_path, data_path)
├── n_{slip_plane}/
│   ├── im_data/
│   │   ├── b0_alpha000.npy
│   │   ├── b0_alpha010.npy
│   │   ├── ...
│   │   └── b5_alpha350.npy           # 6 × 36 = 216 files
│   └── images/
│       ├── b0_alpha000.png           # same 216, as PNG previews
│       └── ...
└── geometry.html                         # plotly export, optional
```

`{slip_plane}` slug: e.g. `1_1_1` (joined Miller indices, no minus sign normalised to `m1`). Filenames use zero-padded angle so sorting matches angular order.

### Multi-disloc mode

```
<output>/
├── manifest.csv
├── im_data/
│   ├── 00000.npy
│   ├── 00001.npy
│   └── ...                            # n_samples files, zero-padded
└── images/
    ├── 00000.png
    └── ...                            # only first `n_png_previews` files
```

`manifest.csv` schema:

| Column | Description |
|---|---|
| `sample_id` | 0-padded integer |
| `n1_h`, `n1_k`, `n1_l` | slip plane normal for dislocation 1 |
| `b1_idx` | Burgers vector index ∈ {0..5} |
| `b1_h`, `b1_k`, `b1_l` | resolved b-vector for dislocation 1 |
| `alpha1_deg` | rotation angle for dislocation 1 ∈ [0, 360) |
| `x1_um`, `y1_um` | in-plane position (lab frame, µm) for dislocation 1 |
| `n2_h`, ..., `y2_um` | same columns for dislocation 2 |
| `image_path` | relative path to `.npy` |

Image content is `(NN1, NN2)` float64, same dtype/shape as `dfxm-forward` outputs.

## Tests

| File | Purpose | Approx count |
|---|---|---|
| `tests/test_burgers.py` | b-vector lookup table for all 4 `{111}` variants; geometric properties (b ⊥ n, magnitude); rotated_t_vectors shape and rotation identity at angle=0; ud_matrices is rotation (orthogonal, det=1). | 6 |
| `tests/test_dislocations_mixed.py` | Fd_find_mixed: `rotation_deg=0` (paper's α=90°, pure edge) reduces to Fd_find(ndis=1) with matching Ud (rtol=1e-12); `rotation_deg=90°` (paper's α=0°, pure screw) produces field whose only nonzero in-plane components are the (0,1) and (0,2) screw terms; cos/sin scaling check at rotation_deg=45° produces a sum of (1/√2)·edge + (1/√2)·screw equivalent forms; position offset shifts the singularity; Fd_find_multi_dislocs_mixed: N=1 case matches Fd_find_mixed; N=2 is additive (sum of two N=1 Fdd contributions + single identity). | 8 |
| `tests/test_pipeline_identification.py` | TOML round-trip for both single and multi configs; smoke run single (1 slip plane × 2 b × 2 α = 4 images); smoke run multi (n_samples=4, seed=0, asserts manifest schema + deterministic regen at same seed); validation errors (mode=multi without multi config; bad slip plane). | 7 |
| `tests/test_viz_burgers.py` | plot_slip_plane_3d returns a Figure with the expected number of traces; raises clear error if plotly missing (mocked). | 2 |

Expected total: ~23 new tests. Current cleanup is 111 tests → target ~134 after port.

## Public API additions (for the package's surface)

- `dfxm_geo.crystal.dislocations.Fd_find_mixed`
- `dfxm_geo.crystal.dislocations.Fd_find_multi_dislocs_mixed`
- `dfxm_geo.crystal.dislocations.MixedDislocSpec`
- `dfxm_geo.crystal.burgers.burgers_vectors`
- `dfxm_geo.crystal.burgers.rotated_t_vectors`
- `dfxm_geo.crystal.burgers.ud_matrices`
- `dfxm_geo.viz.burgers.plot_slip_plane_3d` (requires `[identification]` extras)
- `dfxm_geo.pipeline.IdentificationConfig` (+ sub-dataclasses)
- `dfxm_geo.pipeline.run_identification`
- `dfxm_geo.pipeline.cli_main_identify`
- Console script `dfxm-identify`

## Dependencies

New optional dep group in `pyproject.toml`:

```toml
[project.optional-dependencies]
identification = ["plotly>=5"]
```

No new required deps. `scipy.spatial.transform.Rotation` is already pulled in by existing scipy.

## Configs

Two example TOMLs ship with the port:

`configs/identification_single.toml` (reproduces Borgi 2025's 840-image test set):
```toml
mode = "single"
[crystal]
slip_plane_normal = [1, 1, 1]      # starting slip plane; sweep_all_slip_planes overrides
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
# b_vector_indices = [0, 1, 2, 3, 4, 5]  # default = all
sweep_all_slip_planes = true       # iterate all 4 {111} variants (paper's 864 set)
exclude_invisibility = true        # drop 24 configs near G·b = 0 → 840 images
invisibility_threshold_deg = 10.0
[scan]
phi_rad = 150e-6
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0
[io]
dislocs_dirname = "identify"  # unused for identify mode but IOConfig requires
```

`configs/identification_multi.toml`:
```toml
mode = "multi"
[crystal]
slip_plane_normal = [1, 1, 1]  # ignored — multi draws from all {111}
[scan]
phi_rad = 150e-6
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0
[multi]
n_samples = 1000
pos_std_um = 5.0
n_png_previews = 50
[io]
dislocs_dirname = "identify_multi"
```

## Risks / open questions

1. ~~**`Fd_find_mixed` math validation against an independent reference.**~~ **RESOLVED:** Borgi 2025 (J. Appl. Cryst. 58, 813-821) is the canonical reference, with the mixed-character formula as Eq. 1 and the Us/Ud/Theta transforms as Eqs. 3, 5, 7, 8. The cleanup port cites these equations in the `dislocations.py` module docstring and in the `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` docstrings.
2. **`denom1 = sqz + sqy` (not `sqz + sqx`)** in the screw term. Asymmetric in x vs y — preserved from branch source. The paper's Eq. 1 specifies `∂u_d,x/∂z` as the screw out-of-plane term but does not give the analytical denominator explicitly; the branch code's `(z² + y²)` form is what was used to produce the published images and is preserved verbatim. Tests pin behaviour against the branch source; this is not "fixed".
3. **Future phi-sweep extension** in multi-disloc mode is documented as the natural extension point (config schema change + inner loop) but not built. Spec is explicit.
4. **Hg cache** in `dfxm_geo.io.strain_cache` — `load_or_generate_Hg` is for `Fd_find` (edge, ndis-many). The new mixed functions skip it: each multi-disloc Monte Carlo sample needs fresh Fg per (slip_plane, b, α, position), and the inputs aren't cache-keyable in any natural way. Confirmed by user choice in brainstorming.
5. **Output dir collisions** — if the user re-runs with the same `--output` and a different `rng_seed`, the manifest will silently overwrite. Pipeline should `--force` flag or fail if the output dir exists. Default: fail-if-exists, with `--force` to overwrite (matches `dfxm-forward` behaviour).
6. **Reflection is fixed at the forward_model module level.** `direct_space.forward_model.Us`, `Theta`, `q_hkl` are computed at import time from the module-level `(h, k, l) = HKL_DEFAULT = (-1, 1, -1)`. Borgi 2025 also studies the `[020]` reflection (Eq. 13's Us); `dfxm-identify` cannot trivially switch reflections without recomputing the detector ray grid and reciprocal-space quantities. **For this port, identification runs are restricted to the cleanup's default `[111]` reflection**; the `[020]` extension is documented as a future work item, blocked on the same "runtime-configurable reflection" refactor that already blocks `dfxm-forward` from accepting `(h, k, l)` via config.
7. **Invisibility filter G·b ≈ 0** — Borgi 2025 excludes 24 of 864 single-disloc configurations where `|G·b| < cos(80°)` (Burgers vector within 10° of orthogonal to the scattering vector). The pipeline implements this filter; the filter check needs the scattering vector `q_hkl` (already exposed from `forward_model`) and the Burgers vectors (from `crystal.burgers`). Confirmed to match the paper's count of 864 → 840 in the unit tests.

## Implementation notes (clarifications for the plan)

- **Where `intensity_scale` multiplies in:** `image = forward(Hg, phi=config.scan.phi_rad) * config.scan.intensity_scale`, then `image = rng.poisson(image)` if `poisson_noise` is True. Matches the legacy `* 7` before `np.random.poisson`.
- **RNG splitting:** `rng_seed` seeds a master `np.random.Generator`. From it, derive two child streams via `master.spawn(2)`: one for Monte Carlo parameter draws (slip plane, b, α, position), one for Poisson noise. Same seed → bit-identical run. Documented in `IdentificationScanConfig` / `IdentificationMonteCarloConfig` docstrings.
- **Multi-disloc slip-plane pool:** Drawn uniformly from `{(1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1)}` — the four `{111}` family variants used in the branch. `config.crystal.slip_plane_normal` is ignored in multi mode (the example TOML notes this).
- **Burgers-vector index encoding in manifest:** integer in `[0, 6)` per dislocation; the actual `(h, k, l)` is also recorded for parseability. The 6 vectors per slip plane are ordered: first 3 from the basis lookup, then their negatives (matches branch indexing).
- **Single-disloc `slip_plane_normal` validation:** must be one of the 4 `{111}` variants; raised in `IdentificationConfig.__post_init__`. Identical pool to multi mode.
