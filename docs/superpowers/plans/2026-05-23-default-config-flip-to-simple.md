# Sub-project F: Default config flip to "simple" — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an empty TOML produce a valid 1-dislocation, 1-image, Al 111 @ 17 keV run for both `dfxm-forward` and `dfxm-identify`; strip `WallCrystalConfig`'s publication defaults; tag v2.0.0.

**Architecture:** Additive defaults are added first (suite stays green throughout). The breaking `WallCrystalConfig` strip + test-fixture sweep is the penultimate code task. Release-engineering (version bump, rename, notes, merge, tag) is the tail. Per `[[dfxm-no-backcompat-constraint]]`, no compat shim or warning cycle — break loudly via `TypeError`.

**Tech Stack:** Python 3.11+, `dataclasses`, `tomllib`, pytest, mypy, h5py. Venv at `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe` (per CLAUDE.md — repo's `python` is Python 2.7).

**Spec:** `docs/superpowers/specs/2026-05-23-default-config-flip-to-simple-design.md`

**Branch:** `feature/v200-default-simple` (matches the `feature/v131-…` naming used by v1.3.1).

---

## File Structure

**Modified:**
- `src/dfxm_geo/pipeline.py` — every dataclass change lives here
- `configs/default.toml` — header rewrite
- `configs/identification_single.toml` — header rewrite
- `configs/identification_multi.toml` — header rewrite
- `configs/identification_zscan.toml` — header rewrite
- `pyproject.toml` — `1.3.1` → `2.0.0`
- `tests/test_pipeline.py` — 6 implicit-default `WallCrystalConfig` calls made explicit
- `tests/test_pipeline_crystal_modes.py` — `WallCrystalConfig()` test inverted

**Created:**
- `tests/test_defaults_simple.py` — unit tests for the new default factories
- `tests/test_empty_toml_runs.py` — end-to-end empty-TOML smoke (forward + identify)
- `tests/test_partial_reciprocal_override.py` — partial-override semantics for `[reciprocal]`
- `tests/test_wall_no_defaults.py` — `WallCrystalConfig()` raises `TypeError`
- `tests/test_version_is_2_0_0.py` — pin to `2.0.0` (renamed from `_1_3_1`)
- `docs/release-notes-2.0.0.md` — release notes

**Deleted:**
- `tests/test_version_is_1_3_1.py` (via `git mv` → `_2_0_0`)

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 0.1: Confirm clean tree and current HEAD**

```bash
cd C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master
git status -sb
git log --oneline -3
```

Expected: branch `main` clean (untracked artifacts OK; no staged/unstaged tracked changes); HEAD is `21f39e7 docs(spec): fix v1.2.0 -> v1.3.1 version-test filename in F spec` or a later doc-only commit.

- [ ] **Step 0.2: Create branch from main**

```bash
git checkout -b feature/v200-default-simple
```

Expected: `Switched to a new branch 'feature/v200-default-simple'`.

- [ ] **Step 0.3: Confirm baseline tests pass before touching code**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 476 passed, 2 xfailed (the long-standing `Find_Hg`-seed bit-equivalence tests), 0 failed.

---

## Task 1: `CenteredCrystalConfig` gains canonical FCC defaults

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:170-174` (the `b/n/t` field declarations)
- Test: `tests/test_defaults_simple.py` (new)

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_defaults_simple.py`:

```python
"""Sub-project F: dataclass-level defaults for the empty-TOML path."""

from __future__ import annotations

from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    ReciprocalConfig,
    SimulationConfig,
)


class TestCenteredCrystalDefaults:
    def test_bare_construction_uses_canonical_fcc_primary(self) -> None:
        cfg = CenteredCrystalConfig()
        assert cfg.b == (1, 0, -1)
        assert cfg.n == (1, 1, 1)
        assert cfg.t == (1, -2, 1)

    def test_canonical_defaults_satisfy_validators(self) -> None:
        # b · n == 0 and t ∥ (n × b) — the __post_init__ checks.
        # Construction passing without ValueError is the assertion.
        CenteredCrystalConfig()
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestCenteredCrystalDefaults -v
```

Expected: FAIL — `TypeError: CenteredCrystalConfig.__init__() missing 3 required positional arguments: 'b', 'n', and 't'`.

- [ ] **Step 1.3: Add defaults to the dataclass**

In `src/dfxm_geo/pipeline.py`, edit the `CenteredCrystalConfig` field declarations (currently around lines 171-173) from:

```python
    b: tuple[int, int, int]
    n: tuple[int, int, int]
    t: tuple[int, int, int]
```

to:

```python
    b: tuple[int, int, int] = (1, 0, -1)
    n: tuple[int, int, int] = (1, 1, 1)
    t: tuple[int, int, int] = (1, -2, 1)
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestCenteredCrystalDefaults -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Run full suite to confirm no regressions**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 478 passed, 2 xfailed (was 476 + 2 new tests).

- [ ] **Step 1.6: mypy clean**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 1.7: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py
git commit -m "feat(pipeline): CenteredCrystalConfig defaults to canonical FCC primary

Adds (b, n, t) = ((1, 0, -1), (1, 1, 1), (1, -2, 1)) as class-level
defaults. Satisfies b·n=0 and t∥(n×b). First step of sub-project F's
empty-TOML cascade: CenteredCrystalConfig() now constructible bare.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `ReciprocalConfig` gains Al 111 @ 17 keV defaults + partial-override

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:361-386` (`ReciprocalConfig` definition + `from_dict`)
- Test: `tests/test_defaults_simple.py` (extend), `tests/test_partial_reciprocal_override.py` (new)

- [ ] **Step 2.1: Write failing tests for bare-construction default**

Append to `tests/test_defaults_simple.py`:

```python
class TestReciprocalDefaults:
    def test_bare_construction_uses_al_111_17kev(self) -> None:
        cfg = ReciprocalConfig()
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0

    def test_from_dict_none_returns_default(self) -> None:
        cfg = ReciprocalConfig.from_dict(None)
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0

    def test_from_dict_empty_returns_default(self) -> None:
        cfg = ReciprocalConfig.from_dict({})
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0
```

- [ ] **Step 2.2: Write failing tests for partial overrides**

Create `tests/test_partial_reciprocal_override.py`:

```python
"""Sub-project F: ReciprocalConfig.from_dict accepts partial overrides."""

from __future__ import annotations

from dfxm_geo.pipeline import ReciprocalConfig


class TestPartialReciprocalOverride:
    def test_only_keV_provided_keeps_default_hkl(self) -> None:
        cfg = ReciprocalConfig.from_dict({"keV": 21.0})
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 21.0

    def test_only_hkl_provided_keeps_default_keV(self) -> None:
        cfg = ReciprocalConfig.from_dict({"hkl": [1, 1, 1]})
        assert cfg.hkl == (1, 1, 1)
        assert cfg.keV == 17.0

    def test_both_provided_uses_both(self) -> None:
        cfg = ReciprocalConfig.from_dict({"hkl": [2, 0, 0], "keV": 19.5})
        assert cfg.hkl == (2, 0, 0)
        assert cfg.keV == 19.5
```

- [ ] **Step 2.3: Run tests to verify they fail**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestReciprocalDefaults tests/test_partial_reciprocal_override.py -v
```

Expected: FAIL — `TypeError: ReciprocalConfig.__init__() missing 2 required positional arguments` AND `ValueError: missing [reciprocal] block` (from `from_dict(None)`).

- [ ] **Step 2.4: Update `ReciprocalConfig` dataclass + `from_dict`**

Replace the entire `ReciprocalConfig` definition in `src/dfxm_geo/pipeline.py` (currently approx lines 352-386) with:

```python
@dataclass
class ReciprocalConfig:
    """Sub-project D: reflection identity for kernel lookup.

    The TOML ``[reciprocal]`` block carries both this (small, consumed by
    forward + identify) and bootstrap's MC params (large, consumed only by
    `dfxm-bootstrap`). This dataclass holds only the lookup-relevant keys.

    Sub-project F: both fields default to the IUCrJ-canonical Al 111 @ 17 keV.
    `from_dict(None)` / `from_dict({})` returns the default; partial dicts
    (one of hkl/keV) fall back to the default for the missing key.
    """

    hkl: tuple[int, int, int] = (-1, 1, -1)
    keV: float = 17.0

    def __post_init__(self) -> None:
        # Normalize hkl to a tuple in case a list slipped through programmatic
        # construction; from_dict already does this for TOML callers.
        if not isinstance(self.hkl, tuple):
            self.hkl = tuple(self.hkl)  # type: ignore[assignment]
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # TODO(non-Al materials): hardcoded Al lattice parameter; revisit if/when
        # the codebase supports other crystals. Tracked as deferred work in the
        # sub-project A spec ("materials other than Al") and in the sub-project D
        # spec ("out of scope").
        _validate_reflection(self.hkl, self.keV, 4.0495e-10)

    @classmethod
    def from_dict(cls, data: dict | None) -> ReciprocalConfig:
        if not data:
            return cls()
        kwargs: dict = {}
        if "hkl" in data:
            kwargs["hkl"] = tuple(data["hkl"])
        if "keV" in data:
            kwargs["keV"] = float(data["keV"])
        return cls(**kwargs)
```

Key changes:
- Defaults on both fields.
- `__post_init__` now runs `_validate_reflection` (was inside `from_dict`); this means programmatic `ReciprocalConfig()` also validates. Same Al lattice constant.
- `from_dict({})` and `from_dict(None)` short-circuit to `cls()`.
- Partial dicts only override the keys provided.

- [ ] **Step 2.5: Run tests to verify they pass**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestReciprocalDefaults tests/test_partial_reciprocal_override.py -v
```

Expected: 6 passed.

- [ ] **Step 2.6: Run full suite**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 484 passed, 2 xfailed. Watch for regressions in `tests/test_pipeline.py` (the `from_dict(None)` raise is gone).

If any test previously asserted `ReciprocalConfig.from_dict(None)` raises `ValueError` — find it and flip the assertion to check the default. Likely none, but grep:

```bash
grep -rn "ReciprocalConfig.from_dict(None)" tests/
```

If a match prescribed a raise, edit that test in this same commit.

- [ ] **Step 2.7: mypy clean**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors.

- [ ] **Step 2.8: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py tests/test_partial_reciprocal_override.py
git commit -m "feat(pipeline): ReciprocalConfig defaults to Al 111 @ 17 keV + partial overrides

Sub-project F: bare ReciprocalConfig() now constructible (Al 111, 17 keV).
ReciprocalConfig.from_dict(None) and from_dict({}) return the default
instead of raising. Partial dicts ({'keV': 21.0}) only override the
provided keys; missing keys fall back to defaults. Moves _validate_reflection
to __post_init__ so programmatic construction also validates.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: `CrystalConfig.default()` + `from_dict` softening

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:253-324` (`CrystalConfig` + `from_dict`)
- Test: `tests/test_defaults_simple.py` (extend)

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_defaults_simple.py`:

```python
class TestCrystalConfigDefault:
    def test_default_classmethod_returns_centered_canonical(self) -> None:
        cfg = CrystalConfig.default()
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.wall is None
        assert cfg.random_dislocations is None
        assert cfg.centered.b == (1, 0, -1)
        assert cfg.centered.n == (1, 1, 1)
        assert cfg.centered.t == (1, -2, 1)

    def test_from_dict_none_returns_default(self) -> None:
        cfg = CrystalConfig.from_dict(None)
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.centered.b == (1, 0, -1)

    def test_from_dict_empty_returns_default(self) -> None:
        cfg = CrystalConfig.from_dict({})
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.centered.b == (1, 0, -1)

    def test_from_dict_mode_only_still_requires_sub_block(self) -> None:
        # Explicit `mode = "centered"` with no `[crystal.centered]` still raises:
        # if the user wrote `[crystal]`, they intended to specify something.
        import pytest

        with pytest.raises(ValueError, match=r"\[crystal\.centered\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "centered"})

        with pytest.raises(ValueError, match=r"\[crystal\.wall\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "wall"})
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestCrystalConfigDefault -v
```

Expected: FAIL — `AttributeError: type object 'CrystalConfig' has no attribute 'default'` AND `ValueError: missing [crystal] block` (from `from_dict(None)`).

- [ ] **Step 3.3: Add `.default()` classmethod and soften `from_dict`**

In `src/dfxm_geo/pipeline.py`, add a `.default()` classmethod to `CrystalConfig` (insert after the `from_dict` method, around line 324):

```python
    @classmethod
    def default(cls) -> CrystalConfig:
        """Sub-project F: canonical empty-TOML default.

        Returns mode='centered' with CenteredCrystalConfig() (canonical FCC
        primary). Used as the SimulationConfig.crystal default factory and
        as the empty-TOML fallback in from_dict.
        """
        return cls(mode="centered", centered=CenteredCrystalConfig())
```

Modify the `from_dict` head (currently approx lines 287-293):

```python
    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig:
        if data is None:
            raise ValueError(
                "missing [crystal] block — forward/identify require explicit "
                "crystal layout; see configs/default.toml."
            )
        if "mode" not in data:
            raise ValueError("missing `mode` in [crystal] — required to pick a layout.")
```

to:

```python
    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig:
        # Sub-project F: empty/missing [crystal] → canonical centered default.
        # Explicit `[crystal] mode = "<m>"` without `[crystal.<m>]` still raises
        # below; "default" is reached only by omission, not declaration.
        if not data:
            return cls.default()
        if "mode" not in data:
            raise ValueError("missing `mode` in [crystal] — required to pick a layout.")
```

(Rest of `from_dict` unchanged — the sibling/sub-block validation continues to fire.)

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestCrystalConfigDefault -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Run full suite**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 488 passed, 2 xfailed.

Watch for: any existing test that asserted `CrystalConfig.from_dict(None)` raises. Grep:

```bash
grep -rn "CrystalConfig.from_dict(None)" tests/
```

Flip if found.

- [ ] **Step 3.6: mypy clean**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors.

- [ ] **Step 3.7: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py
git commit -m "feat(pipeline): CrystalConfig.default() + from_dict accepts empty input

Sub-project F: CrystalConfig.from_dict(None) and from_dict({}) now
return cls.default() = mode='centered' + canonical FCC primary. Explicit
[crystal] mode = '<m>' without [crystal.<m>] still raises — the default
path is *omission*, not declaration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: `SimulationConfig` field defaults flip

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:389-412` (`SimulationConfig` + `from_toml`)
- Modify: `src/dfxm_geo/pipeline.py:run_simulation` head (the `if reciprocal is None: raise` guard, if any)
- Test: `tests/test_defaults_simple.py` (extend)

- [ ] **Step 4.1: Locate the `reciprocal is None` runtime guard**

```bash
grep -n "reciprocal is None\|reciprocal=None" src/dfxm_geo/pipeline.py
```

Expected: a couple of hits — the field declaration, and at least one runtime check inside `run_simulation` (or `_lookup_and_load_kernel` invoker). Note the line numbers; you'll edit them in 4.4.

- [ ] **Step 4.2: Write failing test**

Append to `tests/test_defaults_simple.py`:

```python
class TestSimulationConfigDefaults:
    def test_bare_construction_succeeds(self) -> None:
        cfg = SimulationConfig()
        # Crystal cascades to default
        assert cfg.crystal.mode == "centered"
        assert cfg.crystal.centered is not None
        # Reciprocal cascades to default
        assert cfg.reciprocal is not None
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 17.0
        # Scan defaults to all-axes-fixed (single mode)
        assert cfg.scan.scanned_axes() == ()
        assert cfg.scan.derived_mode_name() == "single"
```

- [ ] **Step 4.3: Run test to verify it fails**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestSimulationConfigDefaults -v
```

Expected: FAIL — `TypeError: SimulationConfig.__init__() missing 1 required positional argument: 'crystal'`.

- [ ] **Step 4.4: Flip the field defaults**

In `src/dfxm_geo/pipeline.py`, replace the `SimulationConfig` dataclass (approx lines 389-398) with:

```python
@dataclass
class SimulationConfig:
    # Sub-project F: crystal cascades to canonical-centered default (was: required).
    crystal: CrystalConfig = field(default_factory=CrystalConfig.default)
    scan: ScanConfig = field(default_factory=ScanConfig)
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Sub-project F: reciprocal cascades to Al 111 @ 17 keV (was: Optional[None]).
    reciprocal: ReciprocalConfig = field(default_factory=ReciprocalConfig)
```

Then delete the runtime `if config.reciprocal is None: raise` guard inside `run_simulation` (line number located in 4.1). The type tightened to non-Optional; mypy will surface anything you missed.

- [ ] **Step 4.5: Run test to verify it passes**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestSimulationConfigDefaults -v
```

Expected: 1 passed.

- [ ] **Step 4.6: mypy clean**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors. If mypy reports `unreachable` warnings, those are the `is None` branches you missed in 4.4 — delete them.

- [ ] **Step 4.7: Run full suite**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 489 passed, 2 xfailed.

- [ ] **Step 4.8: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py
git commit -m "feat(pipeline): SimulationConfig() constructible bare

Sub-project F: crystal field default_factory=CrystalConfig.default;
reciprocal field type tightens from ReciprocalConfig|None to
ReciprocalConfig with default_factory. run_simulation's None-guard
deleted (collapsed by the type flip). Empty TOML now produces a
fully-resolved SimulationConfig via from_toml's existing
from_dict(raw.get('crystal')) etc. calls.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: `IdentificationCrystalConfig.slip_plane_normal` default

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:419` (`slip_plane_normal` field)
- Test: `tests/test_defaults_simple.py` (extend)

- [ ] **Step 5.1: Write failing test**

Append to `tests/test_defaults_simple.py`:

```python
class TestIdentificationCrystalDefaults:
    def test_bare_construction_uses_slip_plane_111(self) -> None:
        cfg = IdentificationCrystalConfig()
        assert cfg.slip_plane_normal == (1, 1, 1)
        # Other fields already had defaults — spot-check:
        assert cfg.sweep_all_slip_planes is True
        assert cfg.exclude_invisibility is True
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestIdentificationCrystalDefaults -v
```

Expected: FAIL — `TypeError: IdentificationCrystalConfig() missing 1 required keyword-only argument: 'slip_plane_normal'`.

- [ ] **Step 5.3: Add default**

In `src/dfxm_geo/pipeline.py`, change the `slip_plane_normal` field (line 419) from:

```python
    slip_plane_normal: tuple[int, int, int]
```

to:

```python
    slip_plane_normal: tuple[int, int, int] = (1, 1, 1)
```

- [ ] **Step 5.4: Run test to verify it passes**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestIdentificationCrystalDefaults -v
```

Expected: 1 passed.

- [ ] **Step 5.5: Run full suite + mypy**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q && & "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 490 passed, 2 xfailed; mypy 0 errors.

- [ ] **Step 5.6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py
git commit -m "feat(pipeline): IdentificationCrystalConfig.slip_plane_normal defaults to (1,1,1)

Sub-project F identification cascade: bare IdentificationCrystalConfig()
now constructible. (1,1,1) is the canonical {111} starting plane;
sweep_all_slip_planes=True (existing default) means the full {111} family
is still swept when the user opts into the default.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: `IdentificationConfig` defaults flip + `load_identification_config` soften

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:481-519` (`IdentificationConfig` dataclass)
- Modify: `src/dfxm_geo/pipeline.py:522-567` (`load_identification_config` function)
- Test: `tests/test_defaults_simple.py` (extend)

- [ ] **Step 6.1: Write failing test**

Append to `tests/test_defaults_simple.py`:

```python
class TestIdentificationConfigDefaults:
    def test_bare_construction_succeeds(self) -> None:
        cfg = IdentificationConfig()
        assert cfg.mode == "single"
        assert cfg.crystal.slip_plane_normal == (1, 1, 1)
        assert cfg.scan.derived_mode_name() == "single"
        assert cfg.noise.poisson_noise is True
        assert cfg.reciprocal is not None
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        # multi/zscan blocks stay None outside their modes
        assert cfg.multi is None
        assert cfg.zscan is None
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestIdentificationConfigDefaults -v
```

Expected: FAIL — `TypeError: IdentificationConfig.__init__() missing N required keyword-only arguments`.

- [ ] **Step 6.3: Flip field defaults on `IdentificationConfig`**

In `src/dfxm_geo/pipeline.py`, replace the `IdentificationConfig` field declarations (approx lines 487-496) with:

```python
    # Sub-project F: every field cascades to the empty-TOML default.
    mode: Literal["single", "multi", "z-scan"] = "single"
    crystal: IdentificationCrystalConfig = field(default_factory=IdentificationCrystalConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    noise: IdentificationNoiseConfig = field(default_factory=IdentificationNoiseConfig)
    io: IOConfig = field(default_factory=IOConfig)
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None
    # Sub-project F: reciprocal tightens to non-Optional + default.
    reciprocal: ReciprocalConfig = field(default_factory=ReciprocalConfig)
```

Important: the dataclass is currently `@dataclass(frozen=True, kw_only=True)`. Keep `kw_only=True` so the order of `field(default_factory=...)` vs. plain defaults doesn't matter for the constructor signature. Leave the decorator as-is.

The `__post_init__` validation block stays unchanged.

- [ ] **Step 6.4: Soften `load_identification_config`**

In `src/dfxm_geo/pipeline.py`, replace the head of `load_identification_config` (approx lines 536-548):

```python
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    if "mode" not in data:
        raise ValueError(f"{path}: missing top-level 'mode' field")

    crystal_data = data.get("crystal", {})
    if "slip_plane_normal" in crystal_data:
        crystal_data = {
            **crystal_data,
            "slip_plane_normal": tuple(crystal_data["slip_plane_normal"]),
        }
    crystal = IdentificationCrystalConfig(**crystal_data)
```

with:

```python
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    # Sub-project F: 'mode' is now optional in TOML; defaults to 'single'.
    mode = data.get("mode", "single")

    crystal_data = data.get("crystal", {})
    if "slip_plane_normal" in crystal_data:
        crystal_data = {
            **crystal_data,
            "slip_plane_normal": tuple(crystal_data["slip_plane_normal"]),
        }
    crystal = IdentificationCrystalConfig(**crystal_data)
```

Then update the final `IdentificationConfig(...)` constructor call (approx line 559) to use the new `mode` local instead of `data["mode"]`:

```python
    return IdentificationConfig(
        mode=mode,
        crystal=crystal,
        scan=scan,
        noise=noise,
        io=io,
        multi=multi,
        zscan=zscan,
        reciprocal=reciprocal,
    )
```

- [ ] **Step 6.5: Run test to verify it passes**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_defaults_simple.py::TestIdentificationConfigDefaults -v
```

Expected: 1 passed.

- [ ] **Step 6.6: Run full suite + mypy**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q && & "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 491 passed, 2 xfailed; mypy 0 errors.

Watch for: any test that asserted `load_identification_config` raises `ValueError: missing top-level 'mode' field`. Grep:

```bash
grep -rn "missing top-level 'mode'" tests/
```

Flip if found.

- [ ] **Step 6.7: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_defaults_simple.py
git commit -m "feat(pipeline): IdentificationConfig() constructible bare; mode optional in TOML

Sub-project F identification cascade. IdentificationConfig fields all
gain defaults: mode='single', crystal=IdentificationCrystalConfig(),
scan=ScanConfig(), noise=IdentificationNoiseConfig(),
io=IOConfig(), reciprocal=ReciprocalConfig(). multi/zscan stay None
outside their respective modes (existing gating preserved).

load_identification_config now treats top-level 'mode' as optional
(defaults to 'single').

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Empty-TOML smoke for `dfxm-forward`

**Files:**
- Create: `tests/test_empty_toml_runs.py`

- [ ] **Step 7.1: Write failing test (and confirm forward path is already smoke-testable)**

Create `tests/test_empty_toml_runs.py`:

```python
"""Sub-project F: an empty TOML file produces a valid forward + identify run."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import (
    IdentificationConfig,
    SimulationConfig,
    load_identification_config,
)


class TestEmptyTomlForward:
    def test_empty_toml_parses_to_default_simulation_config(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = SimulationConfig.from_toml(empty)
        # Cascades to canonical defaults across the board:
        assert cfg.crystal.mode == "centered"
        assert cfg.crystal.centered is not None
        assert cfg.crystal.centered.b == (1, 0, -1)
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 17.0
        assert cfg.scan.scanned_axes() == ()
        assert cfg.scan.derived_mode_name() == "single"

    def test_empty_toml_runs_end_to_end(self, tmp_path: Path) -> None:
        """Smoke: from_toml -> run_simulation on a 1-image config produces
        a non-empty HDF5 file that load_h5_scan can round-trip.
        """
        # Skip if the bundled Al 111 @ 17 keV kernel isn't present (CI / fresh
        # checkout); the empty-TOML semantics tested above already cover the
        # parse path.
        try:
            import dfxm_geo.direct_space.forward_model as fm

            fm._lookup_kernel_path((-1, 1, -1), 17.0, fm.pkl_fpath)
        except FileNotFoundError:
            pytest.skip("bundled Al 111 @ 17 keV kernel missing; run dfxm-bootstrap first")

        from dfxm_geo.pipeline import run_simulation

        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = SimulationConfig.from_toml(empty)
        # Constrain io fields to tmp_path so the test doesn't pollute the
        # working directory:
        cfg.io.dislocs_dirname = "images"
        cfg.io.perfect_dirname = "perfect"
        cfg.io.include_perfect_crystal = False  # speed
        out = run_simulation(cfg, tmp_path)
        assert isinstance(out, dict)
        # One scan position (single mode); one HDF5 master file exists.
        master_h5 = tmp_path / "simulation.h5"
        assert master_h5.exists()
```

- [ ] **Step 7.2: Run tests to verify the parse test passes immediately and the e2e either passes or skips**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_empty_toml_runs.py::TestEmptyTomlForward -v
```

Expected: `test_empty_toml_parses_to_default_simulation_config` PASS; `test_empty_toml_runs_end_to_end` PASS or SKIP (depending on whether the laptop has the Al 111 kernel — per CLAUDE.md it does post-v1.2.0).

- [ ] **Step 7.3: If the e2e fails with `KeyError` / unexpected exception**

Inspect the output file path the pipeline actually writes; it may differ from `tmp_path / "simulation.h5"`. Look at any existing pipeline smoke test (`tests/test_hdf5_run_simulation_end_to_end.py` or `tests/test_hdf5_pipeline.py`) for the canonical output path and fix the assertion to match.

- [ ] **Step 7.4: Commit**

```bash
git add tests/test_empty_toml_runs.py
git commit -m "test(pipeline): empty-TOML produces valid SimulationConfig + end-to-end run

Sub-project F smoke. Parse test confirms an empty .toml file produces a
fully-resolved SimulationConfig with canonical centered crystal +
Al 111 @ 17 keV reciprocal + single scan. End-to-end test runs the
forward pipeline on the resolved config and confirms the master HDF5
file is written (skips if the bundled kernel is missing).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Empty-TOML smoke for `dfxm-identify`

**Files:**
- Modify: `tests/test_empty_toml_runs.py` (extend)

- [ ] **Step 8.1: Write the failing test (parse only — full e2e is heavy)**

Append to `tests/test_empty_toml_runs.py`:

```python
class TestEmptyTomlIdentify:
    def test_empty_toml_parses_to_default_identification_config(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = load_identification_config(empty)
        # mode defaults to 'single':
        assert cfg.mode == "single"
        # crystal hypothesis sweep cascades:
        assert cfg.crystal.slip_plane_normal == (1, 1, 1)
        assert cfg.crystal.sweep_all_slip_planes is True
        # noise + reciprocal cascade:
        assert cfg.noise.poisson_noise is True
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        # multi/zscan blocks stay None outside their modes:
        assert cfg.multi is None
        assert cfg.zscan is None

    def test_partial_identify_toml_overrides_only_specified_keys(self, tmp_path: Path) -> None:
        partial = tmp_path / "partial.toml"
        partial.write_text("""
[reciprocal]
keV = 21.0
""")
        cfg = load_identification_config(partial)
        # Mode still defaults:
        assert cfg.mode == "single"
        # Reciprocal: hkl defaults, keV overridden:
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 21.0
```

- [ ] **Step 8.2: Run tests to verify they pass**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_empty_toml_runs.py::TestEmptyTomlIdentify -v
```

Expected: 2 passed.

- [ ] **Step 8.3: Commit**

```bash
git add tests/test_empty_toml_runs.py
git commit -m "test(pipeline): empty-TOML produces valid IdentificationConfig

Sub-project F identification smoke. Empty .toml → mode='single' +
canonical {111} hypothesis sweep + default noise + Al 111 reciprocal.
Partial .toml ([reciprocal] keV = 21.0 only) overrides just keV.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Strip `WallCrystalConfig` defaults — THE BREAKING CHANGE

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:204-221` (`WallCrystalConfig` definition)
- Modify: `tests/test_pipeline.py` (6 call sites: lines 37, 126, 133, 184, 236, 338, 351 — verify with grep in 9.2)
- Modify: `tests/test_pipeline_crystal_modes.py` (lines 37-49 — the `TestWallCrystalConfig` class)
- Create: `tests/test_wall_no_defaults.py`

- [ ] **Step 9.1: Write the failing new test**

Create `tests/test_wall_no_defaults.py`:

```python
"""Sub-project F: WallCrystalConfig has no defaults (breaking change for v2.0.0)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import WallCrystalConfig


class TestWallNoDefaults:
    def test_bare_construction_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 3 required"):
            WallCrystalConfig()  # type: ignore[call-arg]

    def test_missing_two_fields_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 2 required"):
            WallCrystalConfig(dis=4.0)  # type: ignore[call-arg]

    def test_missing_one_field_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 1 required"):
            WallCrystalConfig(dis=4.0, ndis=151)  # type: ignore[call-arg]

    def test_all_three_provided_succeeds(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
        assert cfg.dis == 4.0
        assert cfg.ndis == 151
        assert cfg.sample_remount == "S1"
```

- [ ] **Step 9.2: Catalog every existing bare-or-partial `WallCrystalConfig(...)` call site**

```bash
grep -rn "WallCrystalConfig(" src/ tests/
```

Expected sites (verified during plan-writing):
- `src/dfxm_geo/pipeline.py:321` — `WallCrystalConfig(**sub_data)` (TOML callers, OK — sub_data has all 3)
- `tests/test_pipeline.py:37, 126, 184, 236, 338, 351` — `WallCrystalConfig(dis=..., ndis=...)` (missing sample_remount)
- `tests/test_pipeline.py:133, 140` — `WallCrystalConfig(dis=..., ndis=..., sample_remount=...)` (already explicit — OK)
- `tests/test_pipeline_crystal_modes.py:38` — bare `WallCrystalConfig()` (test_constructs_with_defaults — to be REWRITTEN)
- `tests/test_pipeline_crystal_modes.py:44, 49` — already explicit (kept) and bare-with-sample_remount-only (test_invalid_remount_rejected, needs update)
- `tests/test_forward_dispatch.py:111, 121` — already explicit ✓
- `tests/test_hdf5_pipeline.py:133` — already explicit ✓
- `tests/test_hdf5_run_simulation_end_to_end.py:39` — already explicit ✓

If the grep reveals additional sites not listed above (e.g. someone added one between plan-writing and execution), patch them in this same task following the same rule: add the missing `sample_remount="S1"` (the IUCrJ default the call was previously inheriting).

- [ ] **Step 9.3: Update `WallCrystalConfig` to strip defaults + make kw-only**

In `src/dfxm_geo/pipeline.py`, replace the `WallCrystalConfig` dataclass (approx lines 204-221) with:

```python
@dataclass(kw_only=True)
class WallCrystalConfig:
    """Dis-spaced grid of dislocations (sub-project C, mode='wall').

    The current Borgi/Purdue IUCrJ 2024 layout. Sub-project F strips the
    publication-grade defaults: dis/ndis/sample_remount must be specified
    explicitly. This is the v2.0.0 breaking change. `kw_only=True` ensures
    the strip surfaces as a clear "missing N required keyword-only argument"
    TypeError rather than positional-arg confusion.
    """

    dis: float
    ndis: int
    sample_remount: str

    def __post_init__(self) -> None:
        if self.sample_remount not in SAMPLE_REMOUNT_OPTIONS:
            valid = ", ".join(SAMPLE_REMOUNT_OPTIONS.keys())
            raise ValueError(
                f"sample_remount must be one of: {valid} (got {self.sample_remount!r})"
            )
```

- [ ] **Step 9.4: Fix the 6 implicit-default call sites in `tests/test_pipeline.py`**

Edit each of these lines to add `sample_remount="S1"`:
- Line 37
- Line 126
- Line 184
- Line 236
- Line 338
- Line 351

Use sed/grep to find them precisely, but the edits look like:

Before:
```python
WallCrystalConfig(dis=4.0, ndis=151)
```

After:
```python
WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
```

Before:
```python
WallCrystalConfig(dis=1.0, ndis=2)
```

After:
```python
WallCrystalConfig(dis=1.0, ndis=2, sample_remount="S1")
```

- [ ] **Step 9.5: Rewrite `tests/test_pipeline_crystal_modes.py::TestWallCrystalConfig`**

Replace the `TestWallCrystalConfig` class (lines 36-49) with:

```python
class TestWallCrystalConfig:
    def test_explicit_construction_succeeds(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
        assert cfg.dis == 4.0
        assert cfg.ndis == 151
        assert cfg.sample_remount == "S1"

    def test_custom_remount(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S2")
        assert cfg.sample_remount == "S2"

    def test_invalid_remount_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_remount must be one of"):
            WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S9")
```

(The old `test_constructs_with_defaults` is inverted by `tests/test_wall_no_defaults.py::TestWallNoDefaults::test_bare_construction_raises_type_error`.)

- [ ] **Step 9.6: Run the new test file**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_wall_no_defaults.py -v
```

Expected: 4 passed.

- [ ] **Step 9.7: Run the modified test files**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline.py tests/test_pipeline_crystal_modes.py -v
```

Expected: all pass. If any fail with `TypeError: WallCrystalConfig() missing N required` — you missed a call site in 9.4 or 9.5; grep again.

- [ ] **Step 9.8: Run full suite + mypy**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q && & "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 499 passed (or 498 + 1 skipped if the bundled Al 111 kernel is missing and Task 7's e2e test skipped), 2 xfailed; mypy 0 errors.

- [ ] **Step 9.9: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py tests/test_pipeline_crystal_modes.py tests/test_wall_no_defaults.py
git commit -m "feat(pipeline)!: strip WallCrystalConfig defaults (v2.0.0 breaking change)

BREAKING CHANGE: WallCrystalConfig.dis, .ndis, .sample_remount are now
required keyword-only fields. Bare WallCrystalConfig() (or partial
construction) raises TypeError. Removes the silent IUCrJ-publication
default substitution.

All 5 in-repo wall-mode TOMLs already specify dis/ndis/sample_remount
explicitly; no config migration needed. 7 test call sites updated to
add explicit sample_remount='S1' (the IUCrJ default the calls were
previously inheriting). test_pipeline_crystal_modes.TestWallCrystalConfig.
test_constructs_with_defaults inverted to
test_wall_no_defaults.TestWallNoDefaults.test_bare_construction_raises_type_error.

This is the change that drives the v2.0.0 bump.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Rewrite `configs/default.toml` with override-only framing

**Files:**
- Modify: `configs/default.toml`

- [ ] **Step 10.1: Replace the header comment block**

In `configs/default.toml`, replace the top comment (lines 1-4) with:

```toml
# Default DFXM forward-simulation config.
#
# Every block below shows the DEFAULT value the pipeline would use if
# the block were omitted entirely. A literally-empty .toml file is a
# valid input: `dfxm-forward --config <empty.toml>` produces a single
# detector image of a single canonical FCC dislocation at the origin,
# Al 111 reflection at 17 keV. (The empty case uses the dataclass
# defaults: b=(1, 0, -1), n=(1, 1, 1), t=(1, -2, 1).)
#
# Edit any block to override; delete any block to fall back to the
# default shown here.
#
# The active [crystal] + [scan] blocks below are tuned to produce the
# recognizable 2D mosa rocking grid out-of-the-box for users running
# `dfxm-forward --config configs/default.toml` — this is NOT what an
# empty .toml produces (empty = single image, IUCrJ-canonical slip system).
```

- [ ] **Step 10.2: Add per-block "default" markers**

Add a `# Default. Override or omit.` line as the first content of each top-level block. Sample (apply pattern to all 5 blocks):

```toml
[reciprocal]
# Default. Override or omit.
hkl        = [-1, 1, -1]   # Al 111 reflection (default)
...

[scan.phi]
# Override-only: omit for fixed-at-0. Together with [scan.chi] this gives
# the 2D "mosa" rocking grid the IUCrJ paper used.
range = 6e-4
steps = 61
```

For `[crystal.centered]` add an extra clarifying comment:

```toml
[crystal.centered]
# This file uses the IUCrJ b/n/t (1,-1,0)/(1,1,1)/(1,1,-2) to keep
# `dfxm-forward --config configs/default.toml` output bit-stable.
# The empty-TOML fall-through uses different canonical values:
# b=(1,0,-1), n=(1,1,1), t=(1,-2,1).
b = [1, -1, 0]
n = [1, 1, 1]
t = [1, 1, -2]
```

- [ ] **Step 10.3: Smoke-load default.toml under the new schema**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "from dfxm_geo.pipeline import SimulationConfig; cfg = SimulationConfig.from_toml('configs/default.toml'); print('OK:', cfg.crystal.mode, cfg.crystal.centered.b, cfg.scan.derived_mode_name())"
```

Expected: `OK: centered (1, -1, 0) mosa`.

- [ ] **Step 10.4: Run any tests that load default.toml**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q -k "default"
```

Expected: all pass. (Existing tests that load this file should be unaffected — same data, only comments changed.)

- [ ] **Step 10.5: Commit**

```bash
git add configs/default.toml
git commit -m "docs(configs): default.toml header rewrite — override-only framing

Sub-project F: makes the empty-TOML semantics discoverable. Header
explains that every block below shows the default the pipeline would
use if the block were omitted; the file as shipped is tuned to
reproduce the IUCrJ 2D mosa rocking grid (not what an empty .toml
produces).

Per-block 'Default. Override or omit.' markers added. [crystal.centered]
gets an explicit note that this file's b/n/t differs from the
empty-TOML dataclass defaults.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Identification config headers (3 files)

**Files:**
- Modify: `configs/identification_single.toml`
- Modify: `configs/identification_multi.toml`
- Modify: `configs/identification_zscan.toml`

- [ ] **Step 11.1: Add override-only header to `identification_single.toml`**

In `configs/identification_single.toml`, add at the top (above the existing comment block):

```toml
# Default DFXM identification config (mode='single').
#
# Every block below shows the DEFAULT value `dfxm-identify` would use if
# the block were omitted. A literally-empty .toml file is a valid input —
# the result is mode='single' + canonical {111} slip-plane sweep + default
# noise + Al 111 @ 17 keV.
#
# Edit any block to override; delete any block to fall back to the
# default shown here.
#
```

Keep the existing "Reproduces the test set of Borgi 2025..." comment beneath.

- [ ] **Step 11.2: Add the same header to `identification_multi.toml`**

Same pattern. Adjust the "(mode='single')" → "(mode='multi')" in the first comment line, and the "result is mode='single' +" → "result for an empty .toml is mode='single' +" (since empty defaults to single, not multi).

- [ ] **Step 11.3: Add the same header to `identification_zscan.toml`**

Same pattern. Adjust "(mode='single')" → "(mode='z-scan')". Mention: `mode='z-scan'` additionally requires `[zscan].z_offsets_um` (no default).

- [ ] **Step 11.4: Smoke-load all three under the new schema**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "from dfxm_geo.pipeline import load_identification_config; [print(p, load_identification_config(p).mode) for p in ['configs/identification_single.toml', 'configs/identification_multi.toml', 'configs/identification_zscan.toml']]"
```

Expected: each path prints with its corresponding mode (`single`, `multi`, `z-scan`).

- [ ] **Step 11.5: Commit**

```bash
git add configs/identification_single.toml configs/identification_multi.toml configs/identification_zscan.toml
git commit -m "docs(configs): identification_*.toml header rewrite — override-only framing

Sub-project F: same treatment as default.toml. Each identification
config gains a header explaining the empty-TOML semantics
(mode='single' + canonical sweep + default noise + Al 111 reciprocal).
identification_zscan.toml header additionally notes that
[zscan].z_offsets_um stays required.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Bump `pyproject.toml` + rename version test

**Files:**
- Modify: `pyproject.toml:7`
- Rename: `tests/test_version_is_1_3_1.py` → `tests/test_version_is_2_0_0.py`

- [ ] **Step 12.1: Bump version**

In `pyproject.toml`, change line 7 from:

```toml
version = "1.3.1"
```

to:

```toml
version = "2.0.0"
```

- [ ] **Step 12.2: Rename the version test via `git mv`**

```bash
git mv tests/test_version_is_1_3_1.py tests/test_version_is_2_0_0.py
```

- [ ] **Step 12.3: Update the renamed test's contents**

Replace the entire contents of `tests/test_version_is_2_0_0.py` with:

```python
"""Pin the project version to 2.0.0 for the default-config-flip-to-simple release."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_0_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.0.0"
```

- [ ] **Step 12.4: Run the renamed test**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_version_is_2_0_0.py -v
```

Expected: 1 passed.

- [ ] **Step 12.5: Run full suite**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 499 passed (or 498 + 1 skipped — see Task 9.8 note), 2 xfailed. The renamed test offsets the deleted one, so net count vs. Task 9 is unchanged.

- [ ] **Step 12.6: Commit**

```bash
git add pyproject.toml tests/test_version_is_2_0_0.py tests/test_version_is_1_3_1.py
git commit -m "release: bump version 1.3.1 -> 2.0.0

Sub-project F release commit. Renames tests/test_version_is_1_3_1.py
to tests/test_version_is_2_0_0.py (single-line version-string change).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(Note: the `git add` includes both the new file and the deleted-by-mv old file path; `git mv` already staged the rename. The `add` is a no-op for the deletion but ensures everything is bundled.)

---

## Task 13: Release notes

**Files:**
- Create: `docs/release-notes-2.0.0.md`

- [ ] **Step 13.1: Write the release notes file**

Create `docs/release-notes-2.0.0.md`:

```markdown
# DFXM v2.0.0 — Default config flip to "simple"

Released: 2026-05-23.

## Headline: BREAKING CHANGE — `WallCrystalConfig` requires explicit fields

`WallCrystalConfig` no longer ships with the IUCrJ-2024 publication-grade
defaults (`dis=4.0`, `ndis=151`, `sample_remount="S1"`). Calling
`WallCrystalConfig()` bare — or with fewer than three keyword arguments —
now raises `TypeError`.

**Migration:** specify all three fields explicitly.

```python
# Before (v1.3.1 and earlier):
from dfxm_geo.pipeline import WallCrystalConfig
cfg = WallCrystalConfig()                          # silently IUCrJ-default

# After (v2.0.0+):
from dfxm_geo.pipeline import WallCrystalConfig
cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
```

**TOML configs are unaffected.** All wall-mode configs shipped in
`configs/variants/` already specify `dis`/`ndis`/`sample_remount`
explicitly.

## New: empty TOML now produces a valid run

A literally-empty `.toml` file is now a valid input to both
`dfxm-forward` and `dfxm-identify`. The empty case resolves to:

- **`dfxm-forward`**: `mode="centered"` single dislocation at origin
  with canonical FCC primary slip system `(b, n, t) = ((1, 0, -1),
  (1, 1, 1), (1, -2, 1))`, scan trajectory `"single"` (no axes scanned,
  one detector image), Al 111 reflection at 17 keV.
- **`dfxm-identify`**: `mode="single"` + canonical {111} hypothesis sweep
  + default noise (Poisson on, `intensity_scale=7.0`) + Al 111 reciprocal.

Every dataclass in the config hierarchy that previously had no default
now does: `CenteredCrystalConfig`, `ReciprocalConfig`, `CrystalConfig`,
`SimulationConfig`, `IdentificationCrystalConfig`, `IdentificationConfig`.

## New: partial `[reciprocal]` overrides

`ReciprocalConfig.from_dict` now accepts partial dicts. The following
TOML works (keeps default `hkl`, overrides only `keV`):

```toml
[reciprocal]
keV = 21.0
```

Symmetric for `hkl`-only.

## Migrated / updated

- `configs/default.toml`: header rewritten with override-only framing.
  Active blocks still produce the recognizable IUCrJ mosa grid for
  `dfxm-forward --config configs/default.toml`; only an explicitly
  empty TOML drops to single-image behavior.
- `configs/identification_{single,multi,zscan}.toml`: same header
  treatment.
- 7 test call sites updated to add explicit `sample_remount="S1"`.

## Unchanged

- Bit-equivalence safety net (`tests/data/golden/Fd_find_smoke.npy`)
  — covers the wall path, unaffected by F.
- All 7 `configs/variants/*.toml` — already explicit, untouched.
- HDF5 schema (B+C's `scan_mode` / `scanned_axes` / `crystal_mode`
  attrs populate identically; empty-TOML runs write `scan_mode="single"`,
  `scanned_axes=[]`, `crystal_mode="centered"`).
- Collaborator branches (`Beam_Stop`, `CDD_inc`, `ESRF_DTU`,
  `Purdue_Paper`, `dislocation_identification`) — ground-truth
  references, never touched.

## Out of scope (deferred)

- Module-level FOV constants (`Npixels`, `psize`, `zl_rms`) migrating to
  a `[detector]` config block — still tracked as a v2.1+ follow-up.
- Find_Hg seeding (the 2 long-standing xfailed bit-equivalence tests
  stay xfailed).
- darling 2.0.0 external-link traversal — separate follow-up.

## Upgrade

```bash
pip install --upgrade dfxm-geo==2.0.0
```

If your code constructs `WallCrystalConfig()` bare or with partial args,
audit and fix per the migration snippet above before upgrading.
```

- [ ] **Step 13.2: Commit**

```bash
git add docs/release-notes-2.0.0.md
git commit -m "docs(release): v2.0.0 release notes

Sub-project F. Headline: WallCrystalConfig defaults stripped (breaking).
Documents the empty-TOML semantics for both dfxm-forward and dfxm-identify,
partial-override semantics for [reciprocal], and the list of touched
config files. Migration snippet for bare WallCrystalConfig() callers.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: Final verification gate

**Files:** none

- [ ] **Step 14.1: Full suite from clean state**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q
```

Expected: 499 passed (or 498 + 1 skipped if Task 7's e2e test couldn't find the bundled Al 111 @ 17 keV kernel), 2 xfailed, 0 failed.

Math, in case the count comes in different: started at 476 passed; net additions per task — Task 1 +2, Task 2 +6 (3+3), Task 3 +4, Task 4 +1, Task 5 +1, Task 6 +1, Task 7 +2, Task 8 +2, Task 9 +4 (the TestWallCrystalConfig class in test_pipeline_crystal_modes.py keeps 3 tests via the inversion rewrite). Total: 476 + 23 = 499. The rename in Task 12 is net 0. If the count differs by more than ±2 from 499, grep the new test files to find the missing/extra one.

- [ ] **Step 14.2: mypy clean**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 14.3: Confirm release notes render**

```bash
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "from pathlib import Path; print(Path('docs/release-notes-2.0.0.md').read_text()[:500])"
```

Expected: the first 500 characters of the release notes print successfully (no `FileNotFoundError`).

- [ ] **Step 14.4: Branch state**

```bash
git log --oneline main..HEAD
```

Expected: ~13 commits, one per task, ordered Task 0 → Task 13.

---

## Task 15: Merge to main (LOCAL ONLY)

**Files:** none (git only)

Per CLAUDE.md: "**Confirm before pushing or opening PRs. Local work is fine to do freely.**" This task does a local merge + tag; pushing is gated on user approval in Task 16.

- [ ] **Step 15.1: Switch to main**

```bash
git checkout main
git status -sb
```

Expected: `## main` (no uncommitted changes).

- [ ] **Step 15.2: Merge feature branch with `--no-ff`**

```bash
git merge --no-ff feature/v200-default-simple -m "Merge sub-project F: default config flip to simple (v2.0.0)

Pipeline-features arc complete. Empty TOML now produces a valid
1-dislocation, 1-image run for both dfxm-forward and dfxm-identify.
WallCrystalConfig defaults stripped — the v2.0.0 breaking change.

See docs/release-notes-2.0.0.md for the full changelog and migration
guide.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Expected: merge commit created. Use `--no-ff` to preserve the per-task commit history (same pattern as v1.2.0/E's merge).

- [ ] **Step 15.3: Tag v2.0.0 (annotated)**

```bash
git tag -a v2.0.0 -m "v2.0.0 — Default config flip to simple

Sub-project F. Headline: empty TOML produces a valid 1-image,
1-dislocation, Al 111 @ 17 keV run for both forward and identify.

BREAKING: WallCrystalConfig requires explicit dis/ndis/sample_remount.

See docs/release-notes-2.0.0.md."
```

- [ ] **Step 15.4: Confirm tag + commit graph**

```bash
git log --oneline --decorate -5
```

Expected: top line shows the merge commit tagged `v2.0.0`; next lines are the task commits on the feature branch.

---

## Task 16: Push (GATED on user approval)

**Files:** none (git only)

**STOP. Confirm with the user before this task.** Per CLAUDE.md: "Confirm before pushing or opening PRs." Do not run these commands until the user gives explicit go-ahead.

- [ ] **Step 16.1: Push main**

```bash
git push origin main
```

Expected: pushes the merge commit. (publish.yml does NOT fire on commit pushes — only on tag pushes.)

- [ ] **Step 16.2: Push the v2.0.0 tag**

```bash
git push origin v2.0.0
```

Expected: pushes the tag. publish.yml fires on `tags: ["v*"]` → TestPyPI auto-publishes, PyPI publish queued behind the `pypi` Environment manual-approval gate.

- [ ] **Step 16.3: Confirm GitHub Actions status**

```bash
gh run list --workflow=publish.yml --limit 3
```

Expected: the most-recent run for `v2.0.0` is `in_progress` or `completed`. TestPyPI step should auto-succeed; PyPI step is `waiting` until manual approval.

---

## Task 17: Update CLAUDE.md + auto-memory

**Files:**
- Modify: `C:\Users\borgi\Documents\GM-reworked\CLAUDE.md` (multiple sections)
- Create/Modify: `C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\session_handoff_2026-05-23_v200-shipped.md`
- Modify: `C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\MEMORY.md` (add session-handoff link, mark previous as superseded)
- Modify: `C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\cleanup_session_state.md`

- [ ] **Step 17.1: Update CLAUDE.md "Latest release tag" line**

In `CLAUDE.md`, find the line starting `- **Latest release tag**:` and update it to reflect v2.0.0, today's date, the merge commit SHA, the tag SHA, and the commit count from the merge log. Pattern matches the existing v1.2.0 / v1.3.0 / v1.3.1 entries above.

- [ ] **Step 17.2: Update CLAUDE.md "main HEAD" line**

Update `- **main HEAD**:` to the new merge SHA (from `git log --oneline -1 main`) with the tag annotation.

- [ ] **Step 17.3: Close the F row in the pipeline-features arc table**

In the table currently showing `**F** | Default config flip to "simple" | Pending; v2.0.0 candidate`, change to `**F** | Default config flip to "simple" | ✅ SHIPPED (merge <SHA>, tag v2.0.0); local-only until push` (then update the trailing clause when the user approves the push in Task 16).

- [ ] **Step 17.4: Add an "Already resolved" entry**

Append to the "Already resolved (no action needed)" section in CLAUDE.md:

```markdown
- Default config flip to "simple" + WallCrystalConfig defaults strip —
  shipped in v2.0.0 (sub-project F, merge `<SHA>`, tag `v2.0.0`). Empty
  TOML produces 1-image, 1-dislocation, Al 111 @ 17 keV runs for both
  forward and identify. `WallCrystalConfig()` bare now raises TypeError;
  in-repo wall TOMLs unaffected (all already explicit). See
  `docs/release-notes-2.0.0.md`.
```

- [ ] **Step 17.5: Write the session handoff**

Create `C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\session_handoff_2026-05-23_v200-shipped.md`:

```markdown
---
name: session-handoff-2026-05-23-v200-shipped
description: Sub-project F shipped — v2.0.0 tagged locally. Empty TOML produces valid runs for both forward and identify; WallCrystalConfig defaults stripped (breaking).
metadata:
  type: project
---

# Session handoff — 2026-05-23 (v2.0.0 SHIPPED)

## Where we are

Sub-project F (default config flip to "simple") shipped on
`feature/v200-default-simple` (~13 commits), merged into main with
`git merge --no-ff` (merge SHA <FILL IN>), tagged **v2.0.0**
(annotated, <FILL IN>). Suite at HEAD: 499 passed / 2 xfailed (long-
standing Find_Hg-seed bit-equivalence). mypy: 0 errors across 28
source files.

**Pipeline-features arc complete.** A, D, B+C, E, v1.3.0-A, v1.3.0-B (v1.3.1),
and F all shipped. The cleanup is at its natural close-out point.

## What changed in v2.0.0

- Empty TOML now produces a valid `dfxm-forward` and `dfxm-identify`
  run: mode=centered (b=(1,0,-1), n=(1,1,1), t=(1,-2,1)) /
  mode=single + canonical {111} sweep, scan=single (no axes), Al 111
  reciprocal @ 17 keV.
- `WallCrystalConfig` no longer has default fields. Bare construction
  raises TypeError. THE breaking change.
- `[reciprocal]` accepts partial overrides.
- `configs/default.toml` + 3 identification configs gained override-only
  header framing.

## Open follow-ups

- **Push gate** (Task 16): tag pushed YET? <FILL IN — YES/NO>. If no,
  push when user gives go-ahead. publish.yml fires on tag push;
  TestPyPI auto, PyPI manual-approval-gated.
- **Phase 11 (collaborator branches)** — CLOSED per Sina 2026-05-23:
  Beam_Stop, CDD_inc, ESRF_DTU, Purdue_Paper, dislocation_identification
  stay frozen as ground-truth references. Never to be rebased.
- Long-tail items (Find_Hg seeding, darling external-link, CDD_inc wire
  chord physics, etc.) unchanged from [[session-handoff-2026-05-22-v131-shipped]].

## How to verify

```powershell
cd C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: 499 passed, 2 xfailed; mypy 0 errors.
```

- [ ] **Step 17.6: Add session-handoff link to MEMORY.md and mark prior as superseded**

In `MEMORY.md`:
- Add a new line above the existing latest-session-handoff entry pointing to the new file.
- Edit the previous `**Session handoff 2026-05-22 LATE (v1.3.1 SHIPPED)**` line to remove the `**` bold (no longer the latest).

Pattern:
```markdown
- [**Session handoff 2026-05-23 (v2.0.0 SHIPPED)**](session_handoff_2026-05-23_v200-shipped.md) — `main` HEAD <merge SHA>, tag `v2.0.0` <push status>. Sub-project F shipped: empty TOML works, WallCrystalConfig defaults stripped.
- [Session handoff 2026-05-22 LATE (v1.3.1 SHIPPED)](session_handoff_2026-05-22_v131-shipped.md) — `main` HEAD `92126ce`, tag `v1.3.1` pushed. ...
```

- [ ] **Step 17.7: Update `cleanup_session_state.md`**

Edit the top of the file to add a "Round 39" entry summarizing F's shipment. Pattern matches Rounds 38 / 37 / etc. The MEMORY.md description line for `cleanup_session_state.md` should also update to "Round 39 (latest): v2.0.0 shipped — sub-project F (default config flip to 'simple') complete. Pipeline-features arc done."

- [ ] **Step 17.8: Commit the CLAUDE.md change (only — auto-memory is not in git)**

```bash
# CLAUDE.md is one level up from the repo (per CLAUDE.md's own front matter:
# "Private file, not in git. Lives one level above the repo so it can never
# be accidentally committed.")
# So no git add needed — just confirm the edits saved.
echo "CLAUDE.md updated in place; not in git, no commit needed."
```

- [ ] **Step 17.9: Final state confirmation**

```bash
git -C "C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master" log --oneline --decorate -3
```

Expected: top commit shows `v2.0.0` tag annotation; next commits are the per-task work; everything pushed or local-only per Task 16 status.

---

## Done

Sub-project F is shipped. Pipeline-features arc (A, D, B+C, E, v1.3.0-A, v1.3.1, F) complete. The DFXM cleanup is at its natural close-out point; the remaining work is either deferred (`out of scope` items in each spec) or in the long-tail follow-up bucket.
