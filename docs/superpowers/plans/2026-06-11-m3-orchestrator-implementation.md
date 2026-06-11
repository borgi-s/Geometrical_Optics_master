# M3 Orchestrator (multi-reflection sweeps, plan 2 — B′ decided) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the foundation's NotImplementedError guards with a working per-reflection orchestrator under the **B′ ω-handling decision** (Sina, 2026-06-11): forward + identify loop over `config.reflections`, one standard master per reflection in `reflection_NNN/` subdirs plus a thin super-master, per-scan g·b visibility labels, all-reflections invisibility semantics, and the invisibility physics regression test.

**Architecture:** B′ = ω applied ONLY at the projection step (`Us_eff = R_z(ω) @ Us` inside `precompute_forward_static`, guarded so ω=0 is bit-identical); Hg paths keep the un-rotated `Us` (Hg stays reflection-independent → shared within a (θ,η) group). The orchestrator loop is caller-side: per `ReflectionRun`, build a per-run `(ReciprocalConfig, GeometryConfig)` pair → `_load_resolution` → `build_forward_context(run.theta, res, run.hkl, omega=run.omega)` → existing single-reflection machinery writing into `output_dir/reflection_NNN/`. Same RNG seed across reflections (same crystal — the scientific requirement); per-reflection independent Poisson noise via SeedSequence.

**Tech Stack:** Python 3.11, numpy, h5py, pytest, mypy, smoke-scale tests on the **analytic backend** (no kernel npz needed → CI-safe; `backend="analytic"` requires `beamstop=false`).

**Spec:** `docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md` (§3 DECIDED B′; §7 HDF5 option 2; §8 labels).
**Baseline:** branch `feature/m3-multi-reflection-sweeps` @ `ac0e358`; full suite 815 passed / 4 skipped / 1 xfail / **0 failures**; mypy 0/36.
**Worktree:** `C:\Users\borgi\Documents\GM-reworked\wt-multi-reflection`; python `.\.venv\Scripts\python.exe` (NEVER bare `python`). NO full-scale runs (cluster row pending); smoke scale = ≤9 scan frames, Npixels ≤ 128, ndis ≤ 2.

**Physics definition of B′ (write into code docstrings verbatim):** per reflection, the diffraction vector and projection rotate with the crystal's goniometer ω about lab ẑ: `base_qc = (R_z(ω) @ Us) @ Hg @ q_hkl`. The ray grid `rl`, beam profile, and Hg field stay shared (same probed volume — the B′ approximation). Goniometer scan offsets (φ, χ, 2θ) remain lab-frame and are added to the rotated `base_qc` (the goniometer is lab-mounted). Full-ω (rotating `rl` itself) is the documented upgrade path, isolated behind this one seam.

---

### Task 1: B′ core — `omega` through GeometryContext into `precompute_forward_static`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` — `GeometryContext` (~line 57), `build_geometry_context` (~624), `build_forward_context` (~659), `precompute_forward_static` (~682)
- Test: `tests/test_bprime_projection.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
"""B-prime omega seam: R_z(omega) @ Us at the projection step only."""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.oblique import _R_z


def _dummy_resolution() -> fm.ResolutionContext:
    # Fields per ResolutionContext dataclass; analytic_eval/Resq_i None is fine —
    # precompute_forward_static never touches resolution.
    return fm.ResolutionContext(
        Resq_i=None, qi1_start=0.0, qi1_step=1.0, qi2_start=0.0, qi2_step=1.0,
        qi3_start=0.0, qi3_step=1.0, npoints1=None, npoints2=None, npoints3=None,
        analytic_eval=None, loaded_kernel_path=None,
    )


@pytest.fixture
def small_hg() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(scale=1e-4, size=(50, 3, 3))


def test_omega_zero_is_bit_identical(small_hg):
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    ctx0_explicit = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=0.0)
    a = fm.precompute_forward_static(small_hg, ctx0)
    b = fm.precompute_forward_static(small_hg, ctx0_explicit)
    assert np.array_equal(a, b)  # bit-identical, not approx


def test_omega_default_field_is_zero():
    ctx = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    assert ctx.geometry.omega == 0.0


def test_omega_rotates_projection(small_hg):
    omega = 0.7
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    ctxw = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=omega)
    base0 = fm.precompute_forward_static(small_hg, ctx0)   # (3, N)
    basew = fm.precompute_forward_static(small_hg, ctxw)
    # B': base_qc(omega) == R_z(omega) applied to the un-rotated projection
    np.testing.assert_allclose(basew, _R_z(omega) @ base0, rtol=1e-12, atol=1e-18)


def test_hg_paths_unaffected_by_omega():
    """ctx.instrument.Us must stay UN-rotated — Hg computation reads it."""
    ctxw = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=0.7)
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    assert np.array_equal(ctxw.instrument.Us, ctx0.instrument.Us)
```

NOTE: `ResolutionContext` field names above are from a survey — open the dataclass and match the real constructor exactly before running. If `build_geometry_context` is the public seam instead of `build_forward_context` for extra args, follow the existing call chain.

- [ ] **Step 2: Run to verify failure** — `.\.venv\Scripts\python.exe -m pytest tests/test_bprime_projection.py -q` → TypeError (unexpected kwarg `omega`).

- [ ] **Step 3: Implement**

3a. `GeometryContext` gains a field (keep frozen):

```python
    omega: float = 0.0  # B' goniometer omega (rad); consumed ONLY by precompute_forward_static
```

3b. `build_geometry_context(theta_run, instrument)` gains `omega: float = 0.0` and passes it through. `build_forward_context(theta_run, resolution, hkl, instrument=None, omega: float = 0.0)` forwards it.

3c. `precompute_forward_static` (the ONLY consumer — keep it that way; this is the documented full-ω upgrade seam):

```python
    Us = ctx.instrument.Us
    if ctx.geometry.omega != 0.0:
        # B' (spec §3, decided 2026-06-11): the diffraction vector and projection
        # rotate with goniometer omega about lab z; rl/Hg stay shared (same
        # probed volume). Guarded so omega=0 keeps v2.5.1 float ops bit-identical.
        from dfxm_geo.crystal.oblique import _R_z

        Us = _R_z(ctx.geometry.omega) @ Us
    qs = Us @ Hg @ ctx.q_hkl
    return qs.squeeze().T
```

(Module-level import of `_R_z` if forward_model already imports from crystal.oblique without cycles — check; otherwise keep the local import with a comment.)

- [ ] **Step 4: Run the new tests + the forward bit-identity neighborhood**

`.\.venv\Scripts\python.exe -m pytest tests/test_bprime_projection.py -q` → 4 pass.
`.\.venv\Scripts\python.exe -m pytest tests -q -k "forward or static or golden"` → failure set unchanged (empty).

- [ ] **Step 5: mypy + commit**

`.\.venv\Scripts\python.exe -m mypy src/dfxm_geo/` → 0.
`git add src/dfxm_geo/direct_space/forward_model.py tests/test_bprime_projection.py && git commit -m "feat: B-prime omega seam in precompute_forward_static (omega=0 bit-identical)"`

---

### Task 2: Per-run resolution loading + per-reflection context helper

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (next to `_load_resolution`, ~line 912)
- Modify: `src/dfxm_geo/direct_space/forward_model.py` ONLY IF the `expected_hkl` check in `_load_default_kernel` needs a bypass parameter (read it first)
- Test: `tests/test_reflection_run_context.py` (create)

The orchestrators need, per `ReflectionRun`: a `ResolutionContext` (analytic or group-kernel) and a `ForwardContext`. Centralize in two helpers:

```python
def _resolution_for_run(
    reciprocal: ReciprocalConfig, geometry: GeometryConfig, run: ReflectionRun
) -> fm.ResolutionContext:
    """Per-reflection resolution: analytic gets (theta(hkl), eta=run.eta); MC gets
    the group kernel by (run.theta, run.eta, keV) — kernel metadata hkl is the
    bootstrap group REPRESENTATIVE's hkl, so the per-hkl metadata check is
    relaxed (the (theta, eta, keV) match is the contract; spec §6)."""
    recip_run = replace(reciprocal, hkl=run.hkl, eta=run.eta)
    geom_run = GeometryConfig(
        mode="oblique", eta=run.eta, theta_validated=run.theta,
        omega=run.omega, mount=geometry.mount,
    )
    return _load_resolution(recip_run, geom_run)


def _context_for_run(
    res: fm.ResolutionContext, run: ReflectionRun
) -> fm.ForwardContext:
    return fm.build_forward_context(run.theta, res, run.hkl, omega=run.omega)
```

- [ ] **Step 1: Write the failing tests**

```python
"""Per-ReflectionRun resolution + context helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.reflections import resolve_reflections
from dfxm_geo.pipeline import (
    GeometryConfig,
    ReciprocalConfig,
    _context_for_run,
    _resolution_for_run,
)

MOUNT = CrystalMount(
    lattice="cubic", a=4.0493e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1)
)


@pytest.fixture
def run113():
    return resolve_reflections([{"hkl": [1, 1, 3]}], MOUNT, 19.1)[0]


def _analytic_recip() -> ReciprocalConfig:
    return ReciprocalConfig(keV=19.1, backend="analytic", beamstop=False, lattice_a=4.0493e-10)


def test_analytic_resolution_for_run(run113):
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    res = _resolution_for_run(_analytic_recip(), geom, run113)
    assert res.analytic_eval is not None
    assert res.loaded_kernel_path is None
    # eta must be the RUN's eta, not the config-level placeholder
    assert res.analytic_eval.eta == pytest.approx(run113.eta)


def test_context_for_run_carries_run_geometry(run113):
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    res = _resolution_for_run(_analytic_recip(), geom, run113)
    ctx = _context_for_run(res, run113)
    assert ctx.geometry.theta_0 == pytest.approx(run113.theta)
    assert ctx.geometry.omega == pytest.approx(run113.omega)
    expected_q = np.asarray(run113.hkl, dtype=float)
    expected_q /= np.linalg.norm(expected_q)
    np.testing.assert_allclose(ctx.q_hkl, expected_q)
```

ADAPT: check `AnalyticResolution`'s attribute name for eta (read `fm._load_analytic_resolution` and the AnalyticResolution class) — if eta isn't introspectable, assert on whatever IS (e.g. a stored param dict), don't weaken to "is not None" only. `ReciprocalConfig(...)` constructor args: verify field names/requireds; `backend="analytic"` requires `beamstop=False` or it raises.

- [ ] **Step 2: failure run** → ImportError on `_resolution_for_run`.

- [ ] **Step 3: Implement.** Place both helpers right after `_load_resolution`. For the MC branch: READ `fm._load_default_kernel(target, expected_hkl=..., expected_keV=...)` first — if it hard-fails on hkl mismatch, thread an `expected_hkl=None` (skip-check) through `_lookup_and_load_kernel` for the multi-reflection path with a comment citing spec §6 ("LUT covers the group, not one hkl"). Do NOT remove the check for single-reflection mode. Add an MC-path test only if you can fake it cheaply (a tiny real npz written by `np.savez` with the metadata keys `_load_default_kernel` requires — read what it needs; if that's >25 lines of fixture, skip the MC test and note it in your report: e2e coverage comes via Task 6's monkeypatch).

- [ ] **Step 4: tests pass; Step 5: mypy 0 + commit** `git commit -m "feat: per-ReflectionRun resolution + forward-context helpers"`

---

### Task 3: Forward orchestrator — per-reflection loop + super-master

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` — `run_simulation` (~947, replace the guard), `_run_simulation_inner` (~970, optional reflection kwarg), `cli_main` postprocess block (~1416)
- Modify: `src/dfxm_geo/io/hdf5.py` — new `write_multi_reflection_master`; `write_simulation_h5` gains optional `reflection_attrs: dict[str, object] | None = None`
- Test: `tests/test_forward_multi_reflection_e2e.py` (create)

- [ ] **Step 1: Write the failing test** (smoke, analytic backend, no kernel needed)

```python
"""2-reflection forward e2e smoke (analytic backend, tiny grid)."""

from __future__ import annotations

import h5py
import pytest

from dfxm_geo.pipeline import SimulationConfig, run_simulation

MULTI_FORWARD_TOML = """
[reciprocal]
keV = 19.1
backend = "analytic"
beamstop = false

[geometry]
mode = "oblique"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "centered"
[crystal.centered]

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 3

[io]
include_perfect_crystal = false

[postprocess]
enabled = false

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
eta = 0.3531
"""


@pytest.fixture
def multi_cfg(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(MULTI_FORWARD_TOML, encoding="utf-8")
    return SimulationConfig.from_toml(p)


def test_two_reflection_forward_writes_per_reflection_masters(multi_cfg, tmp_path):
    out = tmp_path / "out"
    result = run_simulation(multi_cfg, out)
    assert result["n_reflections"] == 2
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_geo.h5"
        assert master.is_file()
        with h5py.File(master, "r") as fh:
            assert "1.1" in fh
            attrs = dict(fh["1.1"].attrs)
            assert attrs["reflection_index"] == idx
            assert attrs["n_reflections"] == 2
            assert "omega" in attrs


def test_super_master_links_and_table(multi_cfg, tmp_path):
    out = tmp_path / "out"
    run_simulation(multi_cfg, out)
    super_master = out / "dfxm_geo_multi.h5"
    assert super_master.is_file()
    with h5py.File(super_master, "r") as fh:
        # ExternalLinks resolve to each reflection's root
        assert "1.1" in fh["reflection_001"]
        assert "1.1" in fh["reflection_002"]
        table = fh["reflections"]
        assert table["hkl"].shape == (2, 3)
        assert table["omega"].shape == (2,)
        assert table["eta"].shape == (2,)
        assert table["theta"].shape == (2,)
        assert fh.attrs["n_reflections"] == 2


def test_per_reflection_images_differ(multi_cfg, tmp_path):
    """Different q_hkl/omega must produce different contrast (B' is live)."""
    import numpy as np

    out = tmp_path / "out"
    run_simulation(multi_cfg, out)
    stacks = []
    for idx in (1, 2):
        det = out / f"reflection_{idx:03d}" / "scan0001" / "dfxm_sim_detector_0000.h5"
        with h5py.File(det, "r") as fh:
            stacks.append(fh["entry_0000/dfxm_sim_detector/image"][...])
    assert not np.allclose(stacks[0], stacks[1])
```

ADAPT before running: the `[crystal]` block needs BOTH mount keys and the forward layout (`mode`/`[crystal.centered]` — copy the minimal valid shape from `tests/test_reflections_config.py` which already solved this); the `[scan]` schema (`[scan.phi]` value/range/steps) must match ScanConfig — copy from an existing tiny forward test (grep `tests/` for a small phi-scan TOML). Detector dataset path inside the LIMA file: verify against `io/hdf5.py` constants. Smoke speed: default Npixels may be 510 — if SimulationConfig has an instrument/pixels knob, set it small; if not, check what existing smoke tests do (e.g. `test_pipeline_multi_reflection.py` crystal-mode smokes) and mirror their size-control mechanism. If analytic-backend + centered-mode needs `[reciprocal] eta` absent (multi rule), it is — per-run eta comes from the resolver.

- [ ] **Step 2: failure run** → NotImplementedError (the foundation guard).

- [ ] **Step 3: Implement**

3a. `run_simulation` — replace the guard with the loop:

```python
    if config.reflections:
        runs = config.reflections
        results: list[dict[str, Any]] = []
        for idx, run in enumerate(runs, start=1):
            res_run = _resolution_for_run(config.reciprocal, config.geometry, run)
            sub_dir = output_dir / f"reflection_{idx:03d}"
            results.append(
                _run_simulation_inner(
                    config, sub_dir, res_run,
                    reflection=run, reflection_index=idx, n_reflections=len(runs),
                )
            )
        write_multi_reflection_master(
            output_dir, runs, master_name="dfxm_geo.h5",
            mount=config.geometry.mount, keV=config.reciprocal.keV,
        )
        return {"n_reflections": len(runs), "reflections": results}
```

3b. `_run_simulation_inner(config, output_dir, res, *, reflection: ReflectionRun | None = None, reflection_index: int = 0, n_reflections: int = 0)`:
- ctx build becomes: `ctx = _context_for_run(res, reflection) if reflection is not None else fm.build_forward_context(run_theta(config), res, config.reciprocal.hkl)`.
- The `write_simulation_h5` call gains `eta=(reflection.eta if reflection else config.geometry.eta)` and `reflection_attrs={"hkl_reflection": np.asarray(reflection.hkl), "omega": reflection.omega, "reflection_index": reflection_index, "n_reflections": n_reflections} if reflection else None`.
- Everything else (Hg_provider, frames, positioners) is untouched — single-reflection path must be byte-identical (reflection=None default).

3c. `write_simulation_h5` gains `reflection_attrs: dict[str, object] | None = None`; merge into the per-scan `attrs` dict (read where attrs is assembled, ~hdf5.py:766) — write `reflection_index`/`n_reflections`/`omega` as attrs and the reflection hkl under the existing attr naming style.

3d. New in `io/hdf5.py`:

```python
def write_multi_reflection_master(
    output_dir: Path,
    runs: Sequence["ReflectionRun"],
    *,
    master_name: str,
    mount: "CrystalMount | None",
    keV: float,
) -> Path:
    """Thin super-master: ExternalLinks to each reflection's standard master +
    the resolved reflection table (spec §7 option 2). Each per-reflection master
    is a fully standard single-reflection file; this adds zero constraints on them."""
    path = output_dir / (Path(master_name).stem + "_multi.h5")
    with h5py.File(path, "w") as fh:
        fh.attrs["n_reflections"] = len(runs)
        fh.attrs["keV"] = float(keV)
        if mount is not None:
            fh.attrs["lattice"] = mount.lattice
            fh.attrs["a"] = float(mount.a)
            for name in ("mount_x", "mount_y", "mount_z"):
                fh.attrs[name] = np.asarray(getattr(mount, name), dtype=np.int64)
        grp = fh.create_group("reflections")
        grp.create_dataset("hkl", data=np.asarray([r.hkl for r in runs], dtype=np.int64))
        grp.create_dataset("omega", data=np.asarray([r.omega for r in runs]))
        grp.create_dataset("eta", data=np.asarray([r.eta for r in runs]))
        grp.create_dataset("theta", data=np.asarray([r.theta for r in runs]))
        grp.create_dataset("group", data=np.asarray([r.group for r in runs], dtype=np.int64))
        for idx in range(1, len(runs) + 1):
            fh[f"reflection_{idx:03d}"] = h5py.ExternalLink(
                f"reflection_{idx:03d}/{master_name}", "/"
            )
    return path
```

(Relative ExternalLink paths — resolve relative to the super-master's directory; that's how the existing per-scan links work. Import `ReflectionRun` under TYPE_CHECKING if needed to avoid cycles.)

3e. `cli_main`: postprocess for multi-reflection configs — loop `run_postprocess(args.output / f"reflection_{idx:03d}", config)` per reflection when `config.reflections` and postprocess enabled. READ `run_postprocess` first: if it needs per-run ctx/Hg that the loop can't supply cheaply, instead print a clear stderr note "postprocess: skipped for multi-reflection runs (per-reflection figures land with plan-2 polish)" and skip — choose based on what the code actually needs, and SAY which you chose in your report.

- [ ] **Step 4: tests green; then full suite** `.\.venv\Scripts\python.exe -m pytest -q` → failure set unchanged (empty). **Step 5: mypy 0 + commit** `git commit -m "feat: per-reflection forward orchestrator + super-master (B-prime)"`

---

### Task 4: Identify orchestrator — loop, RNG policy, all-reflections invisibility

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` — `run_identification` (~2311, replace guard), the three `_run_identification_*` dispatchers + `_iter_identification_single`/`_iter_identification_zscan` (visibility gate), `_maybe_apply_poisson_noise` (~2076)
- Test: `tests/test_identify_multi_reflection_e2e.py` (create)

**RNG policy (write into the code as a comment):** the dislocation parameter stream uses `config.noise.rng_seed` UNCHANGED for every reflection — all reflections image the SAME crystal realization; that is the entire point of multi-reflection data. Poisson noise must be independent per reflection: seed the noise stream with `np.random.default_rng([rng_seed, reflection_index])` when `reflection_index > 0` (0 = single-reflection → exactly today's `default_rng(seed).spawn(2)[1]`, bit-compatible).

**Invisibility policy (spec §8):** in multi-reflection runs the deterministic sweep grid must be IDENTICAL across reflections (scan k = same (plane, b, α, z) in every reflection master — ML labels align by index). So the `exclude_invisibility` gate uses the WHOLE reflection list: keep a config only if `gb_visible(q, b)` for AT LEAST ONE reflection q. Implement by threading `visibility_qs: list[np.ndarray] | None = None` into `_iter_identification_single` and `_iter_identification_zscan` (None → today's single-q behavior via `ctx.q_hkl`); the gate becomes `any(_passes_invisibility(q, b, thr) for q in qs)`. Multi mode has no deterministic gate (unchanged).

- [ ] **Step 1: Write the failing tests**

```python
"""2-reflection identify e2e smoke: per-reflection masters, aligned grids, RNG policy."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import load_identification_config, run_identification

MULTI_IDENTIFY_TOML = """
[reciprocal]
keV = 19.1
backend = "analytic"
beamstop = false

[geometry]
mode = "oblique"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0, 1]
sweep_all_slip_planes = false
exclude_invisibility = false

[identification]
mode = "single"

[identification.single]
n_samples = 2

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 3

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
eta = 0.3531
"""


@pytest.fixture
def multi_cfg_path(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(MULTI_IDENTIFY_TOML, encoding="utf-8")
    return p


def test_two_reflection_identify_layout_and_alignment(multi_cfg_path, tmp_path):
    cfg = load_identification_config(multi_cfg_path)
    out = tmp_path / "out"
    result = run_identification(cfg, out)
    assert result["n_reflections"] == 2
    samples = []
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_identify.h5"
        assert master.is_file()
        with h5py.File(master, "r") as fh:
            scans = sorted(k for k in fh.keys() if k[0].isdigit())
            sample_meta = [dict(fh[s]["sample"].attrs) for s in scans] if False else []
            # grid alignment: same scan COUNT and same per-scan sample params
            samples.append((len(scans), [fh[s]["title"][()] for s in scans]))
    assert samples[0][0] == samples[1][0]          # same number of scans
    assert samples[0][1] == samples[1][1]          # same sweep grid order
    # super-master
    assert (out / "dfxm_identify_multi.h5").is_file()
```

This is a SKETCH for layout/alignment — the implementer must adapt the introspection to the real identification TOML schema (copy the minimal valid blocks from `configs/identification_single.toml` and `tests/test_reflections_config.py`) and to what `write_identification_h5` actually stores per scan (read `ScanSpec` and the `/N.1` layout: sample dict carries the (plane, b, α) draw — assert THOSE are equal across reflections, e.g. `/N.1/sample` attrs or datasets). Keep the title-list comparison only if titles encode the sweep point. Add a second test: identical `/N.1/sample` content across the two reflection masters for every scan index (the real alignment guarantee, since param RNG is shared).

- [ ] **Step 2: failure** → NotImplementedError.

- [ ] **Step 3: Implement** — mirror Task 3's loop in `run_identification`:

```python
    if config.reflections:
        runs = config.reflections
        results = []
        for idx, run in enumerate(runs, start=1):
            res_run = _resolution_for_run(config.reciprocal, config.geometry, run)
            ctx_run = _context_for_run(res_run, run)
            sub_dir = output_dir / f"reflection_{idx:03d}"
            results.append(_dispatch_identification(
                config, sub_dir, ctx_run,
                reflection=run, reflection_index=idx, n_reflections=len(runs),
                visibility_qs=[_q_unit(r.hkl) for r in runs],
            ))
        write_multi_reflection_master(
            output_dir, runs, master_name="dfxm_identify.h5",
            mount=config.geometry.mount, keV=config.reciprocal.keV,
        )
        return {"n_reflections": len(runs), "reflections": results}
```

where `_dispatch_identification` is the existing 3-way mode dispatch factored into a helper so single-reflection `run_identification` and the loop share it (single path passes reflection=None — byte-identical behavior). Thread `reflection_index` into `_maybe_apply_poisson_noise` (new optional param, 0 default → today's stream; >0 → `np.random.default_rng([seed, reflection_index])`). Thread `visibility_qs` into the single/zscan iterators' gates; `reflection_attrs` into `write_identification_h5` (same optional-kwarg pattern as Task 3 — per-scan attrs `reflection_index`/`n_reflections`/`omega`/reflection hkl).

ScanSpec/attrs plumbing: `write_identification_h5` builds attrs per ScanSpec — find where and merge. Keep the diff minimal and the single-reflection call path byte-identical (all new params default-off).

- [ ] **Step 4: e2e tests green; full suite failure set unchanged. Step 5: mypy 0 + commit** `git commit -m "feat: per-reflection identify orchestrator — shared param RNG, per-reflection noise, all-reflections invisibility"`

---

### Task 5: g·b labels in identify HDF5

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` — the three iterators (where ScanSpec dicts are built)
- Modify: `src/dfxm_geo/io/hdf5.py` — only if attrs plumbing needs widening (prefer reusing the ScanSpec dfxm_geo/sample dicts)
- Test: `tests/test_gb_labels_in_identify.py` (create)

Every identify scan that knows its Burgers vector(s) gets per-scan labels (single/zscan: scalars `gb_cos`, `gb_visible`; multi mode: arrays, one per dislocation). Computed against the RUN's q_hkl (`ctx.q_hkl`), via the Task-1-foundation helpers `gb_cos`/`gb_visible` and the config's `invisibility_threshold_deg`. Labels are written for single-reflection runs too (purely additive — new keys, no existing key changes; the bit-identity contract is about existing bytes, so put labels in NEW datasets/attrs only and verify the existing golden-comparison tests still pass — if any byte-compares the whole attrs set, gate labels on multi-reflection-only and note it).

- [ ] **Step 1: failing test** — run a tiny single-mode identify (reuse Task 4's TOML minus `[[reflections]]`, plus `[geometry]` removed → simplified mode, `[reciprocal] hkl = [1,1,3]`), open each `/N.1`, assert `gb_cos` present and equals `gb_cos(q_hkl, b)` recomputed from the scan's stored b, and `gb_visible` consistent with the threshold. Then a 2-reflection variant: labels differ across the two reflection masters for the same scan index (different q).
- [ ] **Step 2-3:** Implement in the iterators where each ScanSpec's `dfxm_geo` dict is assembled (b is in scope there); arrays for multi mode (order = the specs list order, matching the existing per-dislocation file naming).
- [ ] **Step 4: full suite — failure set unchanged. Step 5: mypy + commit** `git commit -m "feat: per-scan gb_cos/gb_visible labels in identify HDF5"`

---

### Task 6: g·b invisibility physics regression test

**Files:**
- Test: `tests/test_gb_invisibility_physics.py` (create); no src changes expected.

The classic criterion as an executable physics gate, at smoke scale on the analytic backend:

- Forward, centered mode, ONE dislocation with `b ∥ [-1, 1, 0]` (in-plane of (111); `b·[1,1,1] = 0` → invisible for g=111; `b·[2,0,0] = -2 ≠ 0` → visible for g=200).
- `[[reflections]]` = `[1,1,1]` and `[2,0,0]` (resolver picks each reflection's solution-1; they are different groups — that's fine).
- Contrast metric per reflection: relative image modulation of the strained scan vs its own perfect-crystal reference, e.g. `np.std(im_dis - im_perfect) / np.mean(im_perfect)` with `include_perfect_crystal = true` (both scans are in each master), or simpler `np.std(im)/np.mean(im)` if the perfect reference is flat — pick whichever is robust at smoke scale and ASSERT a strong ordering: `contrast(g=200) > 5 * contrast(g=111)`.
- CAUTION: the centered-mode default dislocation may not have `b=[-1,1,0]` — read `CenteredCrystalConfig` (b, n, t fields, validated b·n=0, t ∥ n×b) and set explicitly: `b = [-1, 1, 0]`, `n = [1, 1, 1]`, `t` accordingly. Verify with `gb_cos` one-liners before writing the assert.
- If at smoke scale the 5× margin is flaky, drop to 3× but report the measured ratio; do NOT invert or weaken to ≠.

Commit: `git commit -m "test: g.b invisibility reproduced across reflections (physics gate)"`

---

### Task 7: Polish — final-review cosmetics + spec/CLAUDE.md sync

**Files:**
- Modify: `src/dfxm_geo/crystal/reflections.py` (ASCII error strings: η₁/η₂/ω → eta1/eta2/omega in exception MESSAGES only — docstrings may keep glyphs; update any test matching them)
- Modify: `src/dfxm_geo/find_reflections_cmd.py` (hkl column width: negative-index rows overflow 10 chars — widen to 12 and keep header aligned; update the column-parsing test slice accordingly)
- Modify: `docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md` — §7/§8: record what plan 2 actually built (super-master name `*_multi.h5`, per-scan label names, RNG policy, all-reflections invisibility as implemented)
- Test: existing suites adjusted only where messages/columns changed.

Commit: `git commit -m "polish: ASCII CLI errors, table alignment, spec sync (plan-2 close-out)"`

---

### Task 8: Gates

- [ ] Full suite: `.\.venv\Scripts\python.exe -m pytest -q` — failure set unchanged (empty); record the new totals.
- [ ] `.\.venv\Scripts\python.exe -m mypy src/dfxm_geo/` → 0 errors.
- [ ] `.\.venv\Scripts\python.exe -m pre_commit run --all-files` → all pass.
- [ ] Final whole-branch review (controller dispatches; not this plan's job).

---

## Deferred (explicitly out of this plan)

- Hg hoist across reflections within a (θ,η) group (perf; valid under B′ — Hg depends on θ via Theta, so sharing is per-group only). Roadmap P2 territory; needs the cluster baseline first.
- `--reflections-toml` flag on the sweep generators (cheap, low value until users ask).
- `dfxm-migrate-h5` re-pack of legacy single-reflection masters into a super-master.
- Full-ω upgrade (the documented seam: `precompute_forward_static` + rotating `rl`).
- M3 DoD quantitative runs (blocked on the cluster LSF row — NO full runs in this arc).
