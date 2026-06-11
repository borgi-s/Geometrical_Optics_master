# M2 Oblique Full Compatibility (G1.1 + G1.2 + G1.6 + provenance parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining roadmap M2 gaps: identify-mode HDF5 output gains the same per-scan geometry provenance attrs forward output has, all three identify modes get an oblique integration test (G1.1), z-scan + oblique is covered (G1.2), the remount × oblique interaction is tested (G1.6), and the G1.4/G1.5 pixel-level-η decision is filed as a short doc for Sina.

**Architecture:** Extract the geometry-attrs block from `write_simulation_h5` into a shared `geometry_provenance_attrs()` helper in `io/hdf5.py` (forward output stays bit-identical), thread it into `write_identification_h5` via a new optional `geometry_attrs` kwarg built from `config.geometry` + `ctx` in each identify runner. Fast stubbed wiring tests run in the default suite; real-compute oblique e2e tests use the **analytic backend** (`beamstop = false`, no kernel npz needed — same trick as Gate C `test_oblique_forward_contrast.py`) and are marked `slow` (deselected by default per `pyproject.toml` addopts `-m 'not bench and not slow'`).

**Tech Stack:** Python 3.12, pytest, h5py, frozen dataclass configs in `src/dfxm_geo/pipeline.py`, HDF5 writers in `src/dfxm_geo/io/hdf5.py`.

**Execution context:**
- Worktree: `C:\Users\borgi\Documents\GM-reworked\wt-identify-fanout`, branch `feature/m2-oblique-full-compat` (off `30fabe2` = v2.5.1).
- Python: ALWAYS the worktree venv `C:\Users\borgi\Documents\GM-reworked\wt-identify-fanout\.venv\Scripts\python.exe` (editable install points at this tree — verified). NEVER the main-tree venv.
- A 122 MB MC kernel npz exists in this worktree's `reciprocal_space/pkl_files/` so kernel-gated tests run too.
- Reference paper geometry (used throughout): reflection (−1,−1,3) @ 19.1 keV, a=4.0493e-10, identity mount, η=0.353140 rad (20.233°), θ≈15.417° — the `al_oblique_figure3.toml` / Table A.2 values.

**Known landmines (from recon, 2026-06-11):**
1. `ReciprocalConfig` default `lattice_a=4.0495e-10` ≠ paper `a=4.0493e-10`. Always set `lattice_a = 4.0493e-10` explicitly in oblique test configs so `run_theta` returns the Table-A.2 angle.
2. Oblique + `beamstop=True` (the default) tries an MC-kernel lookup `Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_*.npz` which does not exist → KeyError. ALL oblique identify tests must set `beamstop = false` (and `aperture = false`) to route to the analytic backend.
3. `write_identification_h5` raises "no resolution backend loaded" when both `loaded_kernel_path` and `analytic_eval` are None — stub ResolutionContexts in fast tests must set `analytic_eval=object()`.
4. Identify TOML: mount keys (`lattice`, `a`, `mount_x/y/z`) go in `[crystal]` (top level), where `load_identification_config` strips them before building `IdentificationCrystalConfig`. See `tests/test_identification_oblique_wiring.py:46-73` for the exact layout.
5. Programmatic (non-TOML) oblique `IdentificationConfig` construction does NOT auto-propagate η onto `ReciprocalConfig` — set `ReciprocalConfig(..., eta=0.353140)` explicitly or the analytic backend builds a simplified kernel.

---

### Task 1: Extract `geometry_provenance_attrs()` helper (forward refactor, bit-identical)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (new helper above `write_identification_h5`; `write_simulation_h5` lines ~774-789 replaced)
- Test: `tests/test_hdf5_geometry_provenance_helper.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for io.hdf5.geometry_provenance_attrs (M2 provenance parity).

The helper is the single source of the per-scan geometry attrs block written
by BOTH write_simulation_h5 (extracted from it, bit-identical) and
write_identification_h5 (new in M2).
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.io.hdf5 import geometry_provenance_attrs


def test_default_mount_is_al_identity() -> None:
    attrs = geometry_provenance_attrs(
        geometry_mode="simplified", eta=0.0, theta_0=0.1567, mount=None
    )
    assert attrs["geometry_mode"] == "simplified"
    assert attrs["eta"] == 0.0
    assert attrs["theta"] == 0.1567
    assert attrs["lattice"] == "cubic"
    assert np.isclose(attrs["a"], 4.0493e-10, rtol=1e-3)  # Al default
    np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
    np.testing.assert_array_equal(attrs["mount_y"], [0, 1, 0])
    np.testing.assert_array_equal(attrs["mount_z"], [0, 0, 1])
    assert attrs["mount_x"].dtype == np.int64


def test_oblique_mount_passthrough() -> None:
    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    attrs = geometry_provenance_attrs(
        geometry_mode="oblique", eta=0.353140, theta_0=np.deg2rad(15.417), mount=mount
    )
    assert attrs["geometry_mode"] == "oblique"
    assert np.isclose(attrs["eta"], 0.353140)
    assert np.isclose(attrs["theta"], np.deg2rad(15.417))
    assert np.isclose(attrs["a"], 4.0493e-10)
```

NOTE: before running, verify the `CrystalMount` constructor signature against `src/dfxm_geo/crystal/oblique.py` (field names/kwargs above are from the recon; `tests/test_oblique_crystal_mount.py` has known-good constructions — mirror one if the kwargs differ).

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_hdf5_geometry_provenance_helper.py -v`
Expected: FAIL with `ImportError: cannot import name 'geometry_provenance_attrs'`

- [ ] **Step 3: Implement the helper and refactor `write_simulation_h5`**

Add to `src/dfxm_geo/io/hdf5.py` (above `write_identification_h5`):

```python
def geometry_provenance_attrs(
    *,
    geometry_mode: str,
    eta: float,
    theta_0: float,
    mount: CrystalMount | None,
) -> dict[str, Any]:
    """Per-scan geometry provenance attrs (oblique-angle arc, v2.3.0+).

    Single source for the attrs block written on every /N.1 scan group by
    BOTH write_simulation_h5 and write_identification_h5. eta=0 /
    "simplified" reproduces the v2.2.0 forward behaviour; mount=None
    defaults to the Al identity mount.
    """
    resolved: CrystalMount
    if mount is not None:
        resolved = mount
    else:
        from dfxm_geo.reciprocal_space.kernel import _DEFAULT_AL_CRYSTAL

        resolved = _DEFAULT_AL_CRYSTAL
    return {
        "geometry_mode": geometry_mode,
        "eta": float(eta),
        "theta": float(theta_0),
        "lattice": resolved.lattice,
        "a": float(resolved.a),
        "mount_x": np.array(resolved.mount_x, dtype=np.int64),
        "mount_y": np.array(resolved.mount_y, dtype=np.int64),
        "mount_z": np.array(resolved.mount_z, dtype=np.int64),
    }
```

Then in `write_simulation_h5`, replace the inline block (currently lines ~774-789, from the `# Oblique-angle provenance (v2.3.0+)` comment through the three `mount_*` assignments) with:

```python
    # Oblique-angle provenance (v2.3.0+). eta=0 / simplified for v2.2.0 configs.
    attrs_1_1.update(
        geometry_provenance_attrs(
            geometry_mode=geometry_mode,
            eta=eta,
            theta_0=float(_ctx.geometry.theta_0),
            mount=mount,
        )
    )
```

(`CrystalMount` is already imported in hdf5.py — `write_simulation_h5` has a `mount: CrystalMount | None` parameter; if the import is under `TYPE_CHECKING`, move it to a runtime import.)

- [ ] **Step 4: Run the new test + the existing forward provenance tests**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_hdf5_geometry_provenance_helper.py tests/test_pipeline_writes_oblique_provenance.py -v`
Expected: ALL PASS (the second file proves forward attrs are unchanged by the extraction)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_geometry_provenance_helper.py
git commit -m "refactor(io): extract geometry_provenance_attrs from write_simulation_h5"
```

---

### Task 2: Identify HDF5 writes geometry provenance attrs (the parity gap)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (`write_identification_h5` signature + `add_scan` call, lines ~577-668)
- Modify: `src/dfxm_geo/pipeline.py` (new `_identify_geometry_attrs` helper; the three `write_identification_h5` call sites at ~1728, ~1944, ~2179)
- Test: `tests/test_identification_oblique_wiring.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identification_oblique_wiring.py` (reuse the module's existing `_write_oblique_identify_toml`, `_OBLIQUE_ETA`, `_OBLIQUE_THETA`):

```python
def _stub_analytic_resolution():
    """ResolutionContext stub that satisfies the writer's backend guard."""
    from dfxm_geo.direct_space.forward_model import ResolutionContext

    return ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=1.0,
        qi2_start=0.0,
        qi2_step=1.0,
        qi3_start=0.0,
        qi3_step=1.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=object(),  # landmine 3: writer guard needs a backend
        loaded_kernel_path=None,
    )


def _run_identify_with_stubbed_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg
) -> Path:
    """run_identification end-to-end with physics stubbed out (fast)."""
    import dfxm_geo.direct_space.forward_model as fm
    import dfxm_geo.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod,
        "_load_resolution",
        lambda *a, **k: _stub_analytic_resolution(),
    )
    monkeypatch.setattr(fm, "precompute_forward_static", lambda *a, **k: object())
    monkeypatch.setattr(
        fm, "forward_from_static", lambda *a, **k: np.ones((170, 510))
    )
    out = tmp_path / "out"
    run_identification(cfg, out)
    return out / "dfxm_identify.h5"


def test_identify_master_scan_attrs_carry_oblique_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M2 provenance parity: every /N.1 in identify output carries the same
    geometry attrs block forward output has (geometry_mode/eta/theta/mount)."""
    import h5py

    p = tmp_path / "oblique_identify.toml"
    _write_oblique_identify_toml(p)
    cfg = load_identification_config(p)
    master = _run_identify_with_stubbed_forward(monkeypatch, tmp_path, cfg)

    with h5py.File(master, "r") as f:
        scan_ids = [k for k in f if k != "dfxm_geo"]
        assert scan_ids, "expected at least one scan group"
        for sid in scan_ids:
            attrs = f[sid].attrs
            assert attrs["geometry_mode"] == "oblique"
            assert np.isclose(float(attrs["eta"]), _OBLIQUE_ETA, atol=1e-6)
            assert np.isclose(float(attrs["theta"]), _OBLIQUE_THETA, atol=1e-3)
            assert attrs["lattice"] == "cubic"
            assert np.isclose(float(attrs["a"]), 4.0493e-10)
            np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
            # the pre-existing attrs are still there (merge, not replace)
            assert attrs["identify_mode"] == "single"


def test_identify_master_scan_attrs_simplified_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Back-compat: a simplified identify run writes geometry_mode='simplified',
    eta=0 (additive attrs; nothing else changes)."""
    import h5py

    p = tmp_path / "simple_identify.toml"
    p.write_text(
        'mode = "single"\n'
        "[crystal]\n"
        "slip_plane_normal = [1, 1, -1]\n"
        "angle_start_deg = 0.0\n"
        "angle_stop_deg = 0.0\n"
        "angle_step_deg = 10.0\n"
        "b_vector_indices = [0]\n"
        "sweep_all_slip_planes = false\n"
        "exclude_invisibility = false\n"
        "[reciprocal]\n"
        "hkl = [-1, 1, -1]\n"
        "keV = 17.0\n"
    )
    cfg = load_identification_config(p)
    master = _run_identify_with_stubbed_forward(monkeypatch, tmp_path, cfg)

    with h5py.File(master, "r") as f:
        sid = next(k for k in f if k != "dfxm_geo")
        attrs = f[sid].attrs
        assert attrs["geometry_mode"] == "simplified"
        assert float(attrs["eta"]) == 0.0
```

NOTE for the implementer: if `_run_identify_with_stubbed_forward` fails on something other than the missing attrs (e.g. a deeper physics call not covered by the two `fm` stubs), open `tests/test_pipeline_identification.py`, find its `_stub_resolution()` helper and the monkeypatch set its `run_identification`-level tests use, and mirror that exact stub set — it is the proven pattern for kernel-less identify runs.

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_identification_oblique_wiring.py -v`
Expected: the two new tests FAIL with `KeyError: 'geometry_mode'` (attrs absent); the five pre-existing tests still PASS.

- [ ] **Step 3: Implement**

(a) `src/dfxm_geo/io/hdf5.py` — add the kwarg to `write_identification_h5`:

```python
def write_identification_h5(
    output_dir: Path,
    *,
    scan_iter: Iterable[ScanSpec],
    cli: str,
    config_toml: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    write_strain_provenance: bool = True,
    geometry_attrs: dict[str, Any] | None = None,
    ctx: _fm.ForwardContext,
) -> int:
```

document it in the docstring:

```
    `geometry_attrs`: optional geometry provenance attrs
    (from ``geometry_provenance_attrs``) merged into every scan's attrs —
    parity with the ``geometry_mode``/``eta``/``theta``/``mount_*`` attrs
    ``write_simulation_h5`` writes (M2, v2.5.x).
```

and merge at the `add_scan` call (currently `attrs=spec.attrs`):

```python
            master.add_scan(
                scan_id=scan_id,
                title=spec.title,
                start_time=start_time,
                end_time=end_time,
                sample=spec.sample,
                positioners=spec.positioners,
                detector_links=detector_links,
                dfxm_geo=dfxm_geo_meta,
                attrs=(
                    {**spec.attrs, **geometry_attrs} if geometry_attrs else spec.attrs
                ),
            )
```

(b) `src/dfxm_geo/pipeline.py` — add a module-level helper near `run_identification` (~line 2196). `geometry_provenance_attrs` must be added to the existing `from dfxm_geo.io.hdf5 import ...` import:

```python
def _identify_geometry_attrs(
    config: IdentificationConfig, ctx: fm.ForwardContext
) -> dict[str, Any]:
    """Geometry provenance attrs for every /N.1 of an identify run (M2 parity)."""
    return geometry_provenance_attrs(
        geometry_mode=config.geometry.mode,
        eta=config.geometry.eta,
        theta_0=float(ctx.geometry.theta_0),
        mount=config.geometry.mount,
    )
```

(c) Add `geometry_attrs=_identify_geometry_attrs(config, ctx),` to all THREE `write_identification_h5(` calls — in `_run_identification_single` (~line 1728), `_run_identification_multi` (~line 1944), `_run_identification_zscan` (~line 2179). Example (single; the other two are identical in shape):

```python
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_single(config, ctx),
        cli=" ".join(sys.argv),
        config_toml=config_toml,
        max_workers=config.io.max_workers,
        write_strain_provenance=config.io.write_strain_provenance,
        geometry_attrs=_identify_geometry_attrs(config, ctx),
        ctx=ctx,
    )
```

- [ ] **Step 4: Run the wiring file, then the full default suite**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_identification_oblique_wiring.py -v`
Expected: ALL PASS.

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest -q`
Expected: same failure set as the branch baseline (run this on the clean branch first if you haven't recorded it; expected ~777 passed / 4 skipped / 1 xfail with the kernel present, plus whatever Task 1 added). The additive attrs must not break any existing identify-HDF5 assertion.

- [ ] **Step 5: Run mypy**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py src/dfxm_geo/pipeline.py tests/test_identification_oblique_wiring.py
git commit -m "feat(identify): write geometry provenance attrs on every /N.1 (M2 parity)"
```

---

### Task 3: G1.1 — oblique identify e2e, single mode (real analytic compute, slow)

**Files:**
- Create: `tests/test_identification_oblique_e2e.py`

- [ ] **Step 1: Write the test (new file)**

```python
"""G1.1/G1.2 (roadmap M2): oblique identify integration tests, all three modes.

Real-compute e2e on the ANALYTIC resolution backend (beamstop=false, no MC
kernel npz needed — the same trick as Gate C test_oblique_forward_contrast).
Geometry: the paper Fig 3B reflection (-1,-1,3) @ 19.1 keV, eta=0.353140 rad,
theta ~ 15.417 deg (Table A.2). Marked `slow` (module-default grid, ~tens of
seconds per frame); run with `pytest -m slow`.

Each test asserts the M2 DoD: the run completes, the master + per-scan layout
exists, and the oblique geometry round-trips through the per-scan /N.1 attrs
(geometry_mode/eta/theta/mount — written by _identify_geometry_attrs since M2).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import (
    load_identification_config,
    run_identification,
)

_OBLIQUE_ETA = 0.353140
_OBLIQUE_THETA = np.deg2rad(15.417)

# Identify-config TOML for the paper oblique geometry. Landmines (recon
# 2026-06-11): mount keys live in [crystal] top level (stripped before
# IdentificationCrystalConfig); lattice_a must match the mount's a;
# beamstop/aperture false -> analytic backend (no kernel npz on disk needed).
_OBLIQUE_E2E_TOML_SINGLE = (
    'mode = "single"\n'
    "\n"
    "[crystal]\n"
    "slip_plane_normal = [1, 1, -1]\n"
    "angle_start_deg = 0.0\n"
    "angle_stop_deg = 0.0\n"
    "angle_step_deg = 10.0\n"
    "b_vector_indices = [0]\n"
    "sweep_all_slip_planes = false\n"
    "exclude_invisibility = false\n"
    'lattice = "cubic"\n'
    "a = 4.0493e-10\n"
    "mount_x = [1, 0, 0]\n"
    "mount_y = [0, 1, 0]\n"
    "mount_z = [0, 0, 1]\n"
    "\n"
    "[geometry]\n"
    'mode = "oblique"\n'
    f"eta = {_OBLIQUE_ETA}\n"
    "\n"
    "[reciprocal]\n"
    "hkl = [-1, -1, 3]\n"
    "keV = 19.1\n"
    "lattice_a = 4.0493e-10\n"
    "beamstop = false\n"
    "aperture = false\n"
    "\n"
    "[scan.phi]\n"
    "value = 0.46e-3\n"
    "\n"
    "[scan.chi]\n"
    "value = 0.067e-3\n"
    "\n"
    "[scan.two_dtheta]\n"
    "value = -0.42e-3\n"
)


def _assert_oblique_scan_attrs(master_path: Path) -> list[str]:
    """Shared DoD assertion: every /N.1 carries the oblique geometry attrs.

    Returns the scan ids for further per-mode assertions."""
    with h5py.File(master_path, "r") as f:
        scan_ids = [k for k in f if k != "dfxm_geo"]
        assert scan_ids, "identify run produced no scans"
        for sid in scan_ids:
            attrs = f[sid].attrs
            assert attrs["geometry_mode"] == "oblique"
            assert np.isclose(float(attrs["eta"]), _OBLIQUE_ETA, atol=1e-4)
            assert np.isclose(float(attrs["theta"]), _OBLIQUE_THETA, atol=1e-3)
            np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
    return scan_ids


@pytest.mark.slow
def test_identify_single_oblique_e2e(tmp_path: Path) -> None:
    """G1.1 (single): oblique identify runs end-to-end on the analytic
    backend and round-trips oblique provenance (attrs + embedded TOML)."""
    import tomllib

    cfg_path = tmp_path / "oblique_identify.toml"
    cfg_path.write_text(_OBLIQUE_E2E_TOML_SINGLE)
    cfg = load_identification_config(cfg_path)
    assert cfg.geometry.mode == "oblique"  # precondition
    assert cfg.reciprocal.eta == pytest.approx(_OBLIQUE_ETA, abs=1e-6)

    out = tmp_path / "out"
    run_identification(cfg, out)

    master = out / "dfxm_identify.h5"
    assert master.is_file()
    # 1 plane x 1 b x 1 angle = 1 scan
    assert (out / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    scan_ids = _assert_oblique_scan_attrs(master)
    assert scan_ids == ["1.1"]

    with h5py.File(master, "r") as f:
        # detector frame actually computed (non-degenerate physics)
        img = f["/1.1/instrument/dfxm_sim_detector/data"][0].astype(np.float64)
        assert img.max() > 0.0
        # embedded config TOML round-trips oblique (pre-M2 the ONLY provenance)
        toml_str = f["/dfxm_geo/config_toml"][()].decode()
    parsed = tomllib.loads(toml_str)
    assert parsed["geometry"]["mode"] == "oblique"
    assert np.isclose(parsed["geometry"]["eta"], _OBLIQUE_ETA, atol=1e-6)
```

NOTE for the implementer: the detector dataset path `/1.1/instrument/dfxm_sim_detector/data` is the ExternalLink layout asserted in `tests/test_pipeline_identification_hdf5.py:78` — if h5py can't resolve it from the master alone, open the per-scan file directly like that test file does.

- [ ] **Step 2: Run the test**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_identification_oblique_e2e.py -m slow -v`
Expected: PASS (after Tasks 1+2; this test needs no new src code — it is the G1.1 acceptance test for the single mode). Budget ~1 min. If it fails on `_validate_eta_against_compute_omega_eta` or kernel lookup, re-check landmines 1-2.

- [ ] **Step 3: Commit**

```bash
git add tests/test_identification_oblique_e2e.py
git commit -m "test(oblique): G1.1 identify-single oblique e2e on the analytic backend"
```

---

### Task 4: G1.1 — oblique identify e2e, multi mode

**Files:**
- Modify: `tests/test_identification_oblique_e2e.py`

- [ ] **Step 1: Write the test**

Append (programmatic config — landmine 5 applies: set `eta` on `ReciprocalConfig` explicitly):

```python
@pytest.mark.slow
def test_identify_multi_oblique_e2e(tmp_path: Path) -> None:
    """G1.1 (multi): one Monte-Carlo scene under oblique geometry; per-scan
    oblique attrs + per-dislocation sample metadata both present."""
    from dfxm_geo.crystal.oblique import CrystalMount
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        GeometryConfig,
        IdentificationConfig,
        IdentificationCrystalConfig,
        IdentificationMonteCarloConfig,
        IdentificationNoiseConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
    )

    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, -1)),
        scan=ScanConfig(phi=AxisScanConfig(value=0.46e-3)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=1, pos_std_um=5.0),
        geometry=GeometryConfig(
            mode="oblique",
            eta=_OBLIQUE_ETA,
            theta_validated=float(_OBLIQUE_THETA),
            mount=mount,
        ),
        reciprocal=ReciprocalConfig(
            hkl=(-1, -1, 3),
            keV=19.1,
            lattice_a=4.0493e-10,
            eta=_OBLIQUE_ETA,
            beamstop=False,
            aperture=False,
        ),
    )

    out = tmp_path / "out"
    run_identification(cfg, out)

    master = out / "dfxm_identify.h5"
    assert master.is_file()
    scan_ids = _assert_oblique_scan_attrs(master)
    assert scan_ids == ["1.1"]
    with h5py.File(master, "r") as f:
        assert f["1.1"].attrs["identify_mode"] == "multi"
        assert "dislocations" in f["1.1/sample"]
```

NOTE: verify `ReciprocalConfig` accepts `aperture` as a field (it is a `[reciprocal]` TOML key in `al_oblique_figure3.toml`; if the dataclass field is named differently, match it — `grep -n "aperture" src/dfxm_geo/pipeline.py`). If `CrystalMount` kwargs differ, mirror `tests/test_oblique_crystal_mount.py`.

- [ ] **Step 2: Run the test**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_identification_oblique_e2e.py::test_identify_multi_oblique_e2e -m slow -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_identification_oblique_e2e.py
git commit -m "test(oblique): G1.1 identify-multi oblique e2e"
```

---

### Task 5: G1.2 — z-scan + oblique combination e2e

**Files:**
- Modify: `tests/test_identification_oblique_e2e.py`

- [ ] **Step 1: Write the test**

Append. The non-zero `z_offsets_um` entry is the point of G1.2: it drives the `Z_shift(z, xl_range=ctx.geometry.xl_range)` call with the OBLIQUE xl_range (the chain unit-pinned in `tests/test_find_hg_oblique_zshift.py` — this test closes the e2e loop over it):

```python
@pytest.mark.slow
def test_identify_zscan_oblique_e2e(tmp_path: Path) -> None:
    """G1.2: z-scan + oblique combination runs end-to-end; the non-zero z
    layer exercises Z_shift with the oblique ctx.geometry.xl_range, and the
    /N.1 attrs carry oblique provenance."""
    from dfxm_geo.crystal.oblique import CrystalMount
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        GeometryConfig,
        IdentificationConfig,
        IdentificationCrystalConfig,
        IdentificationNoiseConfig,
        IdentificationZScanConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
    )

    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    cfg = IdentificationConfig(
        mode="z-scan",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, -1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(phi=AxisScanConfig(value=0.46e-3)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        zscan=IdentificationZScanConfig(z_offsets_um=[5.0], include_secondary=False),
        geometry=GeometryConfig(
            mode="oblique",
            eta=_OBLIQUE_ETA,
            theta_validated=float(_OBLIQUE_THETA),
            mount=mount,
        ),
        reciprocal=ReciprocalConfig(
            hkl=(-1, -1, 3),
            keV=19.1,
            lattice_a=4.0493e-10,
            eta=_OBLIQUE_ETA,
            beamstop=False,
            aperture=False,
        ),
    )

    out = tmp_path / "out"
    run_identification(cfg, out)

    master = out / "dfxm_identify.h5"
    assert master.is_file()
    scan_ids = _assert_oblique_scan_attrs(master)
    # 1 z-layer x 1 b x 1 angle = 1 scan
    assert scan_ids == ["1.1"]
    with h5py.File(master, "r") as f:
        assert f["1.1"].attrs["identify_mode"] == "z-scan"
```

- [ ] **Step 2: Run the test**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_identification_oblique_e2e.py::test_identify_zscan_oblique_e2e -m slow -v`
Expected: PASS. (If `include_secondary=False` plus `z_offsets_um=[5.0]` trips a validation in `IdentificationZScanConfig.__post_init__` or the zscan positioner code, read the error — the config dataclass at `pipeline.py:588-613` documents the constraints.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_identification_oblique_e2e.py
git commit -m "test(oblique): G1.2 z-scan + oblique combination e2e"
```

---

### Task 6: G1.6 — remount (wall mode) × oblique interaction test

**Files:**
- Create: `tests/test_oblique_remount_wall.py`

Recon audit conclusion (2026-06-11) — encode it in the test docstring: the S1-S4 remount matrices (`crystal/remount.py`) operate purely in the crystal frame inside `Find_Hg`, mechanically independent of the lab-frame Bragg geometry (θ/η), so remount + oblique COMPOSES rather than conflicts — but zero tests covered the combination, so it was silently unverified. This test pins it as ALLOWED + working.

- [ ] **Step 1: Write the test (new file)**

```python
"""G1.6 (roadmap M2): sample remount x oblique geometry interaction.

The S1-S4 remount matrices (crystal/remount.py) act in the CRYSTAL frame
inside Find_Hg; the oblique machinery acts on the lab-frame Bragg geometry
(theta/eta) via the ForwardContext. They are mechanically orthogonal, so the
combination is allowed — but until M2 it was completely untested. This test
pins: a wall-mode forward with a non-trivial remount under the paper oblique
geometry runs end-to-end on the analytic backend and records BOTH the oblique
attrs and the remount in provenance. Marked `slow` (one real analytic frame
at the module-default grid, same budget as Gate C).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.data import configs_root
from dfxm_geo.pipeline import (
    CrystalConfig,
    ScanConfig,
    SimulationConfig,
    WallCrystalConfig,
    run_simulation,
)


@pytest.mark.slow
def test_wall_remount_oblique_forward_e2e(tmp_path: Path) -> None:
    cfg = SimulationConfig.from_toml(configs_root() / "al_oblique_figure3.toml")
    cfg = replace(
        cfg,
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S2"),
        ),
        scan=ScanConfig(),  # rocking peak, single frame (Gate C pattern)
        io=replace(
            cfg.io, write_strain_provenance=False, include_perfect_crystal=False
        ),
    )

    run_simulation(cfg, tmp_path)

    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        attrs = f["/1.1"].attrs
        # oblique provenance intact under wall+remount
        assert attrs["geometry_mode"] == "oblique"
        assert np.isclose(float(attrs["eta"]), 0.353140, atol=1e-4)
        assert np.isclose(float(attrs["theta"]), np.deg2rad(15.416), atol=1e-3)
        # remount provenance recorded alongside it
        samp = f["/1.1/sample"]
        assert samp["remount"][()].decode() == "S2"
        # the wall actually diffracts (not an all-zero frame)
        det = f["/1.1/instrument"]
        img = None
        for key in det:
            if "detector" in key and "data" in det[key]:
                img = det[key]["data"][0].astype(np.float64)
                break
        assert img is not None
        assert img.max() > 0.0
```

NOTE: verify the remount dataset name under `/1.1/sample/` — `write_simulation_h5` takes `sample_remount: str`; grep `sample_remount` in `src/dfxm_geo/io/hdf5.py` for the dataset key it writes (`remount` assumed above; existing forward tests `tests/test_hdf5_pipeline.py:75` show the convention). Adjust the assertion to the actual key, do not skip it — recording remount provenance under oblique is half the point.

- [ ] **Step 2: Run the test**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest tests/test_oblique_remount_wall.py -m slow -v`
Expected: PASS, ~tens of seconds (11 wall dislocations, one analytic frame).

- [ ] **Step 3: Commit**

```bash
git add tests/test_oblique_remount_wall.py
git commit -m "test(oblique): G1.6 wall remount x oblique composes (e2e pin)"
```

---

### Task 7: G1.4/G1.5 decision doc for Sina

**Files:**
- Create: `docs/superpowers/specs/2026-06-11-m2-pixel-eta-decision.md`

- [ ] **Step 1: Write the doc**

```markdown
# M2 decision: pixel-level R_x(eta) in forward() — implement or accept?

**Status: OPEN — Sina's call.** This is roadmap gaps G1.4/G1.5; the decision
changes M2's scope by ~a week and is required by the M2 DoD ("decision
recorded ... either implement pixel-level eta or document the ~4 %
approximation as accepted").

## Context

Since the v2.3.0 oblique Phase A arc, eta enters the simulation ONLY through
the resolution kernel (both backends: analytic_resolution.py and the MC LUT).
The direct-space `forward()` ray grid is NOT rotated by R_x(eta) at the pixel
level. This is an intentional Phase-A scope cut, validated at the time
against darkmod: the residual is ~3.9 % on the thinnest covariance
eigenvalue of the resolution function (see
docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md and the
darkmod parity work in auto-memory darkmod_analytic_resolution_parity).

## Option A — implement pixel-level eta (G1.4), optionally eta-weave (G1.5)

- Apply R_x(eta) in the direct-space projection (forward_model.py, the
  ForwardContext geometry block makes this a contained change post-#16).
- Effort: 3-5 days including darkmod parity re-validation (G1.4); +2-3 days
  if the eta-weave refinement (G1.5) is wanted instead of post-hoc R_x.
- Buys: closes the last known physics gap vs darkmod for oblique
  reflections; needed if quantitative pixel-fidelity to paper Fig 3 (or any
  oblique pixel-level comparison against darkmod/experiment) becomes a goal.
- Risk: touches the hot forward kernel path right after the W3 fusion work;
  full parity gates required.

## Option B — accept + document (recommended)

- Record the ~4 % thinnest-eigenvalue approximation as ACCEPTED in the
  oblique spec; no code change.
- Rationale: (1) M3 multi-reflection sweeps consume oblique geometry only
  through theta/eta kernel selection — first-order B-prime model, already
  deliberately approximate at the same order; (2) the ML-training-data goal
  (perf arc) is contrast-statistics-driven, not pixel-fidelity-driven;
  (3) the gap is confined to the resolution function's thinnest axis, the
  direction least constrained by experiment.
- Revisit trigger: any future quantitative oblique comparison against
  darkmod or measured Fig-3-class data (e.g. a paper figure), or M4/CIF-era
  non-cubic reflections where the approximation is unvalidated.

## Decision

- [ ] Option A (implement G1.4; G1.5 yes/no separately)
- [x] **Option B (accept + document) — recommended** _(pending Sina)_

Once decided: if B, copy the "accept" paragraph into
docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md as a
closing note and tick the M2 DoD box; if A, write a dedicated spec + plan
(it is NOT part of the current M2 test arc).

## G1.6 audit note (closed by this arc)

Remount (S1-S4, crystal frame) x oblique (lab frame) was confirmed
mechanically orthogonal and is now pinned by
tests/test_oblique_remount_wall.py. viz/mosaicity.py is geometry-agnostic
(consumes ctx-derived xl_start); no oblique-specific code needed there.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-11-m2-pixel-eta-decision.md
git commit -m "docs: file G1.4/G1.5 pixel-eta decision for Sina (M2 DoD item)"
```

---

### Task 8: Gates

- [ ] **Step 1: Full default suite (kernel present)**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest -q`
Expected: failure set identical to the branch baseline (empty, modulo the pre-existing main-side `test_render_readme_examples_smoke` bench exclusion — that one is `bench`-marked and deselected anyway). Compare failure SETS, not green counts ([[preexisting-test-failures-2026-05-28]] lesson).

- [ ] **Step 2: The new slow oblique tests, explicitly**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pytest -m slow tests/test_identification_oblique_e2e.py tests/test_oblique_remount_wall.py tests/test_oblique_forward_contrast.py -v`
Expected: ALL PASS (including the pre-existing Gate C as a control).

- [ ] **Step 3: mypy + pre-commit**

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: 0 errors.

Run: `& "...\wt-identify-fanout\.venv\Scripts\python.exe" -m pre_commit run --all-files`
Expected: all hooks pass.

- [ ] **Step 4: Commit any gate fallout, then stop**

Merge/push/version-bump is NOT part of this plan — per house rules, confirm with Sina before pushing or opening PRs. The M2 DoD boxes that remain open after this arc: the G1.4/G1.5 decision itself (Sina) — everything else (identify oblique tests all three modes, provenance round-trip, find-reflections CLI [closed by M3 branch], remount tested) is closed.
