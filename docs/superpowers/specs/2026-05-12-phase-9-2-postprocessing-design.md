# Phase 9.2 — Post-Processing Pipeline (Design)

**Status:** approved 2026-05-12
**Author:** Sina Borgi (DTU), with Claude Code
**Plan reference:** `docs/superpowers/plans/2026-05-12-codebase-cleanup.md` (Phase 9)
**PR:** #1 (draft) — `cleanup/main-modernization`

## Goal

Port the post-processing in `init_forward.py` (lines 102-269) into the `dfxm_geo` package so that `dfxm-forward` reproduces the paper figures end-to-end from a TOML config. After this change, `init_forward.py` moves to `legacy/` and the package is feature-complete with respect to its original entry point.

## Non-goals

- Pixel-level reproduction of the original SVGs. The numerical outputs must match; the figures need to convey the same information but may diverge in formatting.
- The single-pixel exploratory `plt.show()` calls in `init_forward.py` lines 55-99. These are debug-only, never saved, and out of scope.
- Performance work on `compute_com_maps`. The straight port keeps the O(pixels × oversample) double-loop. Vectorization is a Phase 8 item.
- Runtime configurability of `psize`, `zl_rms`, `(h, k, l)` — these remain bound to `forward_model` module-level defaults (deferred from Phase 6).

## What gets ported

Mapping from `init_forward.py` line ranges to new home:

| Source (`init_forward.py`) | Behavior | New location |
|---|---|---|
| L102-119 | χ-shift correction from corner pixel of perfect-crystal stack | `analysis/mosaicity.py::compute_chi_shift` |
| L150-164 | Per-pixel COM extraction over the (φ, χ) grid → `phi_list`, `chi_list` | `analysis/mosaicity.py::compute_com_maps` |
| L167-214 | "Extreme Phi / Chi" 2-panel SVG | `viz/mosaicity.py::plot_mosaicity_maps` |
| L217-269 | qi-field cross-section 2-panel SVG | `viz/mosaicity.py::plot_qi_cross_section` |
| L55-60 | Load stacks from disk | reused via existing `dfxm_geo.io.images.load_images` |

## Architecture

### New modules

**`src/dfxm_geo/analysis/mosaicity.py`** — numerical analysis only, no plotting

```python
def compute_chi_shift(
    stack_perfect: np.ndarray,
    chi_steps: int,
    chi_range: float,
    oversample: int = 100,
) -> float:
    """Measure the systematic χ offset using the corner pixel of a perfect-crystal stack.

    The corner pixel of a strain-free crystal should peak at χ=0 in detector
    coordinates. Any offset is a systematic shift introduced by the finite
    rocking grid; this function returns the shift in degrees so downstream
    χ-axis calibration can correct it.

    Returns:
        shift_deg — additive correction to apply to the χ axis when extracting COMs.
    """
```

```python
def compute_com_maps(
    stack: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    chi_shift: float = 0.0,
    oversample: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel center-of-mass extraction over the (φ, χ) rocking grid.

    Reshapes ``stack`` to ``(chi_steps, phi_steps, H, W)``, runs
    ``scipy.ndimage.center_of_mass`` per pixel against a (chi_oversample×,
    phi_oversample×) refined index grid, and returns mosaicity-in-φ and
    mosaicity-in-χ maps in radians.

    Returns:
        (phi_list, chi_list) — shape (H, W) each.
    """
```

**`src/dfxm_geo/viz/mosaicity.py`** — figure-making only, no analysis

```python
def plot_mosaicity_maps(
    phi_list: np.ndarray,
    chi_list: np.ndarray,
    xl_start: float,
    yl_start: float,
    out_path: Path,
    *,
    vmin: float = -1e-4,
    vmax: float = 1e-4,
) -> None: ...

def plot_qi_cross_section(
    qi_field: np.ndarray,
    xl_start: float, yl_start: float, zl_start: float,
    xl_steps: int, yl_steps: int, zl_steps: int,
    out_path: Path,
    *,
    vmin: float = -1e-4,
    vmax: float = 1e-4,
) -> None: ...
```

Both plotting functions use `matplotlib.use("Agg")`-friendly invocations (no `plt.show()`) and save SVG via `fig.savefig`. Color limits are surfaced as kwargs so users can tune them per-config.

### Extensions to `pipeline.py`

**New dataclass `PostprocessConfig`:**

```python
@dataclass
class PostprocessConfig:
    enabled: bool = True
    chi_oversample: int = 20
    phi_oversample: int = 20
    chi_oversample_for_shift: int = 100
    figures_dirname: str = "figures"
    data_dirname: str = "analysis"
```

Added as a field on `SimulationConfig`:

```python
@dataclass
class SimulationConfig:
    crystal: CrystalConfig = ...
    scan: ScanConfig = ...
    io: IOConfig = ...
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
```

`SimulationConfig.from_toml` is extended to parse a `[postprocess]` section, defaulting to `PostprocessConfig()` if absent.

**New function `run_postprocess`:**

```python
def run_postprocess(output_dir: Path, config: SimulationConfig) -> dict[str, Any]:
    """Run stages 2-4 of init_forward.py against an existing simulation output_dir.

    Reads the perfect-crystal and dislocated stacks from disk, computes the
    χ-shift correction, computes per-pixel COM maps, calls forward() once
    to get the qi field at z=0, and writes both .npy data products and SVG
    figures into output_dir/<data_dirname>/ and output_dir/<figures_dirname>/.

    Returns:
        dict with paths to all produced artifacts plus the in-memory phi_list,
        chi_list, qi_field, and chi_shift for callers that want them.
    """
```

### CLI behavior

`cli_main` gains two mutually-exclusive flags:

| Flag | Effect |
|---|---|
| _none_ | Default: `run_simulation` then `run_postprocess` |
| `--no-postprocess` | Run simulation only (current Phase 6 behavior) |
| `--postprocess-only` | Skip simulation; expects `output_dir` to already contain `<io.dislocs_dirname>/` and `<io.perfect_dirname>/` |

Implementation note: `run_postprocess` itself needs the kernel loaded to compute the qi field (stage 4). `_ensure_kernel_loaded()` is therefore also called at the top of `run_postprocess`, so `--postprocess-only` still surfaces a clear error if the kernel pickle is missing. When both stages run, the check is invoked twice; it's cheap and idempotent.

## Output layout

For an invocation `dfxm-forward --config X --output /path/out`:

```
/path/out/
  images10/                         # dislocated stack (Phase 6)
    mosa_test_0000_*.npy
  images10_perf_crystal/            # perfect-crystal stack (Phase 6)
    mosa_test_0000_*.npy
  analysis/                         # NEW (Phase 9.2)
    phi_list.npy                    # mosaicity map in φ (radians)
    chi_list.npy                    # mosaicity map in χ (radians)
    qi_field.npy                    # full qi field, shape (2 or 3, X, Y, Z)
    chi_shift.txt                   # scalar shift in radians, ASCII
  figures/                          # NEW (Phase 9.2)
    mosaicity_maps.svg              # was: extrem_phi+chi2.svg
    qi_cross_section.svg            # was: qi1+qi2_fields1.svg
```

The `analysis/` directory is the primary persistence; `figures/` is the human-readable summary. Raw `.npy` outputs are cheap and unblock downstream notebooks without re-running the COM loop.

## Tests

**New: `tests/test_mosaicity.py`** — numerical analysis only

1. `compute_chi_shift` on a synthetic perfect-crystal stack whose corner pixel peaks at the center: shift ≈ 0.
2. `compute_chi_shift` on a stack with the corner pixel shifted N grid points: shift ≈ N × (2 × chi_range) / chi_steps.
3. `compute_com_maps` on a synthetic stack where each pixel has a planted Gaussian centroid at a known (φ_i, χ_j): output matches plant within `1 / oversample` resolution.
4. `compute_com_maps` shape correctness across non-square scan grids (catches the `fastgrainplot` class of bug).
5. Determinism: same input → same output (no RNG entanglement).

**Extend: `tests/test_pipeline.py`**

6. `PostprocessConfig` default round-trip through TOML.
7. `SimulationConfig.from_toml` parses a config with no `[postprocess]` section into defaults.
8. `run_postprocess` golden path: write a tiny synthetic stack to a tmp dir, call `run_postprocess` with monkeypatched `forward()` returning a fixed qi field, assert all expected files appear and non-empty.
9. `run_postprocess` errors clearly when expected `images10/` / `images10_perf_crystal/` dirs are missing.
10. CLI flag parity: `--no-postprocess` produces no `analysis/`, `--postprocess-only` skips re-writing the stacks.

**Visual outputs:** assert file exists, size > 0, valid SVG header. No pixel diffing.

Coverage target: `analysis/mosaicity.py` 100%, `pipeline.py` ≥ 85%.

## Migration

1. `mkdir Geometrical_Optics_master/legacy/`
2. `git mv init_forward.py legacy/init_forward.py`
3. Update README: under "Reproducing paper figures", `dfxm-forward --config configs/default.toml --output output/` becomes the documented path. Add a small note at the bottom: "The original entry point `init_forward.py` is preserved under `legacy/` as a single-file reference of the pre-cleanup workflow."
4. Update `docs/reproducibility.md`: replace the "what's intentionally not yet configurable" caveat about post-processing with the new post-processing flow.
5. No changes to existing legacy-path import shims — Phase 5.3 (shim removal) is still blocked on collaborator rebases.

## Public API additions (for the package's surface)

- `dfxm_geo.analysis.mosaicity.compute_chi_shift`
- `dfxm_geo.analysis.mosaicity.compute_com_maps`
- `dfxm_geo.viz.mosaicity.plot_mosaicity_maps`
- `dfxm_geo.viz.mosaicity.plot_qi_cross_section`
- `dfxm_geo.pipeline.PostprocessConfig`
- `dfxm_geo.pipeline.run_postprocess`
- `SimulationConfig.postprocess` field
- CLI flags: `--no-postprocess`, `--postprocess-only`

## Risks / open questions

- **Numerical drift from the original.** The COM extraction depends on oversample factor and the χ-shift convention. The straight port keeps oversample=20 (per `init_forward.py`); any downstream comparison to paper figures should expect bitwise identity in the data products under matching params.
- **`qi_field` artifact size.** Writing the full qi field per run may be wasteful for users who only want the SVG figure. For 9.2 we always write it; if it turns out to dominate disk usage at the default ID06 grid, Phase 8 can add `[postprocess].save_qi_field = false` to opt out.
- **End-to-end validation still blocked on the kernel pickle.** All Phase 9.2 work proceeds with mocked `forward()` in tests; full validation is the same Phase 10 gate as Phase 6.
