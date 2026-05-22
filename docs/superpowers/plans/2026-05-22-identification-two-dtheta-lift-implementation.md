# Lift the Identification `[scan.two_dtheta]` Guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the eager `ValueError` in `run_identification` that today rejects `[scan.two_dtheta]`, then verify identification single + multi + 4-axis configs all run end-to-end, then bump to v1.3.1.

**Architecture:** Pure guard removal. The frame-generation plumbing for `two_dtheta` is already in place from v1.3.0-A: `_scan_frames_args` emits 5-tuples `(idx, Hg, phi, chi, two_dtheta)`, `_compute_frame` calls `forward(Hg, phi, chi, TwoDeltaTheta=two_dtheta)`, `_iter_identification_*` already use `_build_scan_frames_at_z`. `two_dtheta` is a within-scan frame axis in identification (multiplies n_frames per `/N.1/`, not the scan count — matches forward mode and the physics).

**Tech Stack:** Same as v1.3.0-A — numpy, h5py, existing `dfxm_geo` machinery.

**Spec:** `docs/superpowers/specs/2026-05-22-identification-two-dtheta-lift-design.md`.

---

## File touch list (locked in here, drives task decomposition)

| Path | Role |
|---|---|
| `src/dfxm_geo/pipeline.py` | Delete the `if config.scan.is_scanned("two_dtheta"):` block in `run_identification` (~lines 1727-1735). |
| `tests/test_identification_scan_modes.py` | Add 3 tests: single + two_dtheta, multi + two_dtheta, 4-axis single. |
| `tests/test_identification_config_changes.py` | Delete `test_run_identification_eager_guards_unwired_axes` (it has nothing left to assert). |
| `docs/output-format.md` | Append one clarifying sentence to the identification scan-count paragraph. |
| `pyproject.toml` | Bump 1.3.0 → 1.3.1. |
| `tests/test_version_is_1_3_0.py` → `tests/test_version_is_1_3_1.py` | Rename + update literal. |

---

## Task 1: Remove the eager guard + smoke test

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (delete guard block at ~lines 1727-1735)
- Modify: `tests/test_identification_scan_modes.py` (add a smoke test)

- [ ] **Step 1: Write the failing test** at the end of `tests/test_identification_scan_modes.py`:

```python
def test_single_with_two_dtheta_scanned_does_not_raise(tmp_path: Path) -> None:
    """[scan.two_dtheta] in identification single mode runs to completion."""
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1.5e-4),
            two_dtheta=AxisScanConfig(range=1e-4, steps=3),
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    # Must not raise. (Pre-v1.3.0-B this raised ValueError eagerly.)
    run_identification(cfg, tmp_path)
    assert (tmp_path / "dfxm_identify.h5").is_file()
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_single_with_two_dtheta_scanned_does_not_raise -v`

Expected: FAIL with `ValueError: scan axis two_dtheta is configured but not yet wired into identification.`

- [ ] **Step 3: Delete the guard block** in `src/dfxm_geo/pipeline.py`. Find the block (currently ~lines 1727-1735):

```python
    # two_dtheta is not yet wired into the identify forward path. Raise eagerly
    # so users don't get silently-wrong output. z is now wired for single + multi
    # (v1.3.0-A); two_dtheta lifting is tracked as a future follow-up.
    if config.scan.is_scanned("two_dtheta"):
        raise ValueError(
            "scan axis two_dtheta is configured but not yet wired into "
            "identification. For now, set range+steps only on "
            "[scan.phi], [scan.chi], and/or [scan.z]."
        )
```

Delete the entire block (the comment AND the if). The `_lookup_and_load_kernel(...)` call on the next line stays.

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_single_with_two_dtheta_scanned_does_not_raise -v`

Expected: PASS.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`

Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_identification_scan_modes.py
git commit -m "pipeline: lift identification [scan.two_dtheta] guard (v1.3.0-B task 1)"
```

---

## Task 2: Single-mode scan-count regression test

**Files:**
- Modify: `tests/test_identification_scan_modes.py` (add a second test)

- [ ] **Step 1: Write the failing-then-passing test** at the end of `tests/test_identification_scan_modes.py`:

```python
def test_single_with_two_dtheta_scanned_keeps_one_scan_per_plane(tmp_path: Path) -> None:
    """`[scan.two_dtheta]` is a within-scan axis in identification single mode.

    Scan count is unchanged (1 plane * 1 b * 1 alpha = 1 scan); the n_frames
    per scan is multiplied by n_two_dtheta.
    """
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1.5e-4),
            two_dtheta=AxisScanConfig(range=1e-4, steps=3),  # 3 two_dtheta values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 1 z * 1 plane * 1 b * 1 alpha = 1 scan (two_dtheta is within-scan)
        assert scan_keys == ["1.1"]
        # n_frames = 3 (n_two_dtheta)
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 3
        # two_dtheta is a per-frame positioner array of length 3
        assert f["/1.1/instrument/positioners/two_dtheta"].shape == (3,)
        # Angular axis stored in degrees
        assert f["/1.1/instrument/positioners/two_dtheta"].attrs["units"] == "degree"
```

- [ ] **Step 2: Run the test to confirm it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_single_with_two_dtheta_scanned_keeps_one_scan_per_plane -v`

Expected: PASS. (The guard removal in Task 1 unblocked the path; this test now exercises and locks in the within-scan semantics.)

- [ ] **Step 3: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_identification_scan_modes.py
git commit -m "tests: identification single with [scan.two_dtheta] keeps scan count (v1.3.0-B task 2)"
```

---

## Task 3: Multi-mode regression test

**Files:**
- Modify: `tests/test_identification_scan_modes.py` (add a third test)

- [ ] **Step 1: Write the test**:

```python
def test_multi_with_two_dtheta_scanned_keeps_one_scan_per_mc_sample(tmp_path: Path) -> None:
    """`[scan.two_dtheta]` in identification multi mode multiplies n_frames, not scan count."""
    _require_kernel()
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1e-4),
            two_dtheta=AxisScanConfig(range=1e-4, steps=2),  # 2 two_dtheta values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=2, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 1 z * 2 n_samples = 2 scans (two_dtheta is within-scan)
        assert scan_keys == ["1.1", "2.1"]
        # n_frames = 2 (n_two_dtheta) per scan
        for sid in scan_keys:
            assert f[f"/{sid}/instrument/dfxm_sim_detector/data"].shape[0] == 2
            assert f[f"/{sid}/instrument/positioners/two_dtheta"].shape == (2,)
```

- [ ] **Step 2: Run the test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_multi_with_two_dtheta_scanned_keeps_one_scan_per_mc_sample -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_identification_scan_modes.py
git commit -m "tests: identification multi with [scan.two_dtheta] keeps scan count (v1.3.0-B task 3)"
```

---

## Task 4: 4-axis identification integration test

**Files:**
- Modify: `tests/test_identification_scan_modes.py` (add a fourth test)

- [ ] **Step 1: Write the test**:

```python
def test_single_with_all_four_axes_scanned(tmp_path: Path) -> None:
    """All 4 axes scanned in identification single mode.

    z is between-scan (multiplies scan count). phi, chi, two_dtheta are
    within-scan (multiply n_frames).
    """
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            chi=AxisScanConfig(range=2e-3, steps=2),
            two_dtheta=AxisScanConfig(range=1e-4, steps=2),
            z=AxisScanConfig(range=1.0, steps=2),  # 2 z values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 2 z * 1 plane * 1 b * 1 alpha = 2 scans
        assert scan_keys == ["1.1", "2.1"]
        # n_frames per scan = 2 phi * 2 chi * 2 two_dtheta = 8
        for sid in scan_keys:
            assert f[f"/{sid}/instrument/dfxm_sim_detector/data"].shape[0] == 8
            positioners = f[f"/{sid}/instrument/positioners"]
            for axis in ("phi", "chi", "two_dtheta"):
                assert positioners[axis].shape == (8,), (
                    f"{axis} should be a per-frame array of length 8; "
                    f"got shape {positioners[axis].shape}"
                )
            # z is fixed within a scan (scalar)
            assert positioners["z"].shape == ()
            assert positioners["z"].attrs["units"] == "micrometer"
        # The two scans have different z values.
        z0 = f["/1.1/instrument/positioners/z"][()]
        z1 = f["/2.1/instrument/positioners/z"][()]
        assert z0 != z1
```

- [ ] **Step 2: Run the test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_single_with_all_four_axes_scanned -v`

Expected: PASS. (Takes ~30 s — 2 z × 8 frames per scan; Hg is recomputed per z but shared across the within-scan trajectory.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_identification_scan_modes.py
git commit -m "tests: identification single with all 4 axes scanned (v1.3.0-B task 4)"
```

---

## Task 5: Delete the stale eager-guard test

**Files:**
- Modify: `tests/test_identification_config_changes.py`

The current `test_run_identification_eager_guards_unwired_axes` is parametrized over `["two_dtheta"]`. With the guard gone (Task 1), `pytest.raises(ValueError, match="two_dtheta")` no longer matches — the test would fail. Delete it.

- [ ] **Step 1: Run the test to confirm it now fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_config_changes.py::test_run_identification_eager_guards_unwired_axes -v`

Expected: FAIL — `DID NOT RAISE <class 'ValueError'>`. (The configured `[scan.two_dtheta]` is no longer rejected; the test asserted a raise that no longer happens.)

- [ ] **Step 2: Delete the test function from `tests/test_identification_config_changes.py`**

Find the entire block starting at the `@pytest.mark.parametrize("axis_name", ["two_dtheta"])` decorator and including the `def test_run_identification_eager_guards_unwired_axes(...)` function body. Delete it including the explanatory comment block inside.

Also check whether removing this test orphans any imports — if `IdentificationConfig`, `IdentificationCrystalConfig`, `IdentificationNoiseConfig`, `IOConfig`, `ReciprocalConfig`, `ScanConfig`, `AxisScanConfig`, `run_identification`, or `pytest` are now unused, remove only the truly unused ones (ruff will flag if you miss any).

- [ ] **Step 3: Run the remaining tests in the file to confirm they still pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_config_changes.py -v`

Expected: 5 passed (the 5 other tests in the file; previously there were 6).

- [ ] **Step 4: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_identification_config_changes.py
git commit -m "tests: drop stale eager-guard test now that two_dtheta is wired (v1.3.0-B task 5)"
```

---

## Task 6: Update `docs/output-format.md`

**Files:**
- Modify: `docs/output-format.md` (identification scan-count paragraph)

The v1.3.0-A doc update (`b694829`) edited the identification scan-count paragraph to say:

> Identification mode writes one `/N.1/` per config drawn from the runner generator. For `mode="single"`: `n_z × n_planes × n_b × n_alpha` entries (where `n_z` = `[scan.z].steps` if set, else 1). For `mode="multi"`: `n_z × n_samples`. For `mode="z-scan"`: `len(z_offsets_um) × phi_chi_configs` (legacy z-scan mode, distinct from `[scan.z]` axis scanning). New in v1.3.0: `[scan.z]` in single/multi multiplies the scan count by `n_z` and recomputes the strain field at each z-slice.

- [ ] **Step 1: Append a clarifying sentence to that paragraph**

Search for the exact paragraph (`grep -n "Identification mode writes one" docs/output-format.md`). Append at the end of the paragraph:

> `[scan.two_dtheta]` is a within-scan axis in identification (multiplies `n_frames`-per-scan, not the scan count) — it differs from `[scan.z]` because the deformation-gradient `Hg` is two_dtheta-independent, so `forward()` shifts the Bragg angle internally without recomputing `Hg`.

- [ ] **Step 2: Spot-check the rendered doc reads cleanly**

Open `docs/output-format.md` and read the identification scan-count paragraph. Confirm it scans naturally end-to-end after the new sentence.

- [ ] **Step 3: Commit**

```bash
git add docs/output-format.md
git commit -m "docs: clarify [scan.two_dtheta] is within-scan in identification (v1.3.0-B task 6)"
```

---

## Task 7: Release commit — bump to 1.3.1

**Files:**
- Modify: `pyproject.toml` (1.3.0 → 1.3.1)
- Rename: `tests/test_version_is_1_3_0.py` → `tests/test_version_is_1_3_1.py` (update literal)

- [ ] **Step 1: Bump the version in `pyproject.toml`**

Find the `[project]` block and change `version = "1.3.0"` to `version = "1.3.1"`. No other changes to the file.

- [ ] **Step 2: Rename the version test via `git mv`**

```bash
git mv tests/test_version_is_1_3_0.py tests/test_version_is_1_3_1.py
```

- [ ] **Step 3: Update the literal in the renamed test**

The current file content is:

```python
"""Pin the project version to 1.3.0 for the 4-axis scan trajectory release."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_1_3_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "1.3.0"
```

Replace with:

```python
"""Pin the project version to 1.3.1 for the identification two_dtheta lift release."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_1_3_1() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "1.3.1"
```

- [ ] **Step 4: Run the version test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_version_is_1_3_1.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_version_is_1_3_1.py
git commit -m "release: bump to 1.3.1 (identification two_dtheta wired)"
```

---

## Wrap-up (after Task 7)

These are not numbered tasks — they're the merge/tag/push handoff and are user-driven (Sina pushes the buttons or approves each step).

- [ ] **Step 1: Full pytest run** to confirm no regressions across the whole suite.

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`

Expected: ≥ 476 passed (473 from v1.3.0 + 3 new identification two_dtheta tests; minus 1 deleted eager-guard test = net +2 from v1.3.0). 2 xfailed (pre-existing Find_Hg seed). 0 failed.

- [ ] **Step 2: mypy across the package**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`

Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 3: Merge to main (no-ff)**

```bash
git checkout main
git merge --no-ff feature/v131-identification-two-dtheta -m "Merge feature/v131-identification-two-dtheta for v1.3.1"
```

- [ ] **Step 4: Tag v1.3.1**

```bash
git tag -a v1.3.1 -m "v1.3.1 — Lift identification [scan.two_dtheta] guard"
```

- [ ] **Step 5: Push main + tag**

```bash
git push origin main
git push origin v1.3.1
```

The tag push fires `.github/workflows/publish.yml` → TestPyPI auto, PyPI gated by `pypi` Environment manual approval.

- [ ] **Step 6: Update auto-memory + CLAUDE.md**

- Write `session_handoff_<date>_v131-shipped.md`.
- Update `MEMORY.md` index.
- Update `CLAUDE.md`: mark v1.3.0-B / v1.3.1 as shipped in the v1.3.0 follow-on arc table; strike through the "two_dtheta lifting in identification" follow-up.

---

## Self-review

**Spec coverage check:**

- Spec Task 1 (Remove the guard) → Plan Task 1. ✓
- Spec Task 2 (Single + two_dtheta scan count test) → Plan Task 2. ✓
- Spec Task 3 (Multi + two_dtheta scan count test) → Plan Task 3. ✓
- Spec Task 4 (4-axis identification test) → Plan Task 4. ✓
- Spec Task 5 (Delete stale eager-guard test) → Plan Task 5. ✓
- Spec Task 6 (Doc update) → Plan Task 6. ✓
- Spec Task 7 (Release commit + tag) → Plan Task 7 + wrap-up. ✓
- Acceptance criteria mapping:
  - AC1 (smoke runs to completion) → Task 1 covers single mode; Task 3 covers multi; spec says single/multi/zscan but we test single + multi only — zscan is covered by Task 4 indirectly (the zscan dispatcher uses the same `_build_scan_frames_at_z` + `_scan_frames_args` machinery; explicit zscan test omitted to avoid runtime cost; if Sina wants explicit zscan coverage, that's a Task-4.5 addition).
  - AC2 (scan-count semantics) → Tasks 2, 3, 4 each pin a different combination.
  - AC3 (suite passes; old test removed) → wrap-up Step 1 + Task 5.
  - AC4 (mypy clean) → every task ends with mypy.
  - AC5 (v1.3.1 tag on origin/main) → Task 7 + wrap-up Steps 3-5.

**Placeholder scan:** No "TBD" / "TODO" / "fill in" left. The acknowledgement that "zscan is covered indirectly" in the spec-coverage note is explicit and offers an opt-in test.

**Type consistency check:**

- All 4 new tests use `IdentificationConfig`, `IdentificationCrystalConfig`, `IdentificationNoiseConfig`, `IdentificationMonteCarloConfig`, `ScanConfig`, `AxisScanConfig`, `IOConfig`, `ReciprocalConfig`, `run_identification` — all exist in `dfxm_geo.pipeline` and are already imported at the top of `tests/test_identification_scan_modes.py` (verified via grep before plan-writing).
- `_require_kernel()` and `h5py` import are already conventions in `tests/test_identification_scan_modes.py`.
- `tests/test_version_is_1_3_0.py` → `tests/test_version_is_1_3_1.py` rename mirrors the v1.2.0→v1.3.0 rename in the v1.3.0 release commit (`c0d50f4`).

**Risk scan:**

- R3 from the spec (memory for 4-axis 61×61×5 stacks) → mitigated in Task 4 by using 2×2×2×2 = 16 frames per scan, 2 scans = 32 frames total. Fits well within pytest-of-borgi headroom even with 3-session retention.
