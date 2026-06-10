# M3 Foundation (multi-reflection sweeps, decision-independent half) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land every M3 (v2.6.0) component that does not depend on the open ω-handling decision: the `gb_visibility` helper, the `dfxm-find-reflections` CLI, `[[reflections]]`/`[reflections_auto]` config schema + validation + kernel grouping, the multi-reflection `dfxm-bootstrap` loop + kernel manifest, and the sweep-generator reflection axis.

**Architecture:** New pure-logic module `crystal/reflections.py` resolves TOML reflection lists into frozen `ReflectionRun` records (validated via the existing `compute_omega_eta`, grouped by (θ, η) for kernel sharing). `pipeline.py` config loaders carry the resolved list but the orchestrators **refuse it with a clear NotImplementedError** until the ω-decision arc. Bootstrap loops unique groups. Everything is additive; configs without the new tables are bit-identical to v2.5.1.

**Tech Stack:** Python 3.11, dataclasses, tomllib, numpy, pytest, mypy --strict-ish (repo config), pre-commit (ruff).

**Spec:** `docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md`
**Deferred to plan 2 (ω-decision-blocked):** per-reflection orchestrator loop, B′ projection, HDF5 super-master, g·b labels *in HDF5*, invisibility physics smoke test, `--reflections-toml` generator flag.

**Worktree:** `C:\Users\borgi\Documents\GM-reworked\wt-multi-reflection`, branch `feature/m3-multi-reflection-sweeps`. Python: `.\.venv\Scripts\python.exe` (NEVER bare `python` — that's py2.7 on this machine). Run tests as `.\.venv\Scripts\python.exe -m pytest -q ...`. NO full-scale simulation runs in this arc (cluster row pending); smoke scale only.

---

### Task 1: `gb_cos` / `gb_visible` helpers (bit-identical extraction)

**Files:**
- Modify: `src/dfxm_geo/crystal/burgers.py` (append after `burgers_vectors`, line 57)
- Modify: `src/dfxm_geo/pipeline.py:1575-1587` (`_passes_invisibility` delegates)
- Test: `tests/test_gb_visibility.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""g·b visibility helpers — extraction of pipeline._passes_invisibility math."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import gb_cos, gb_visible


def test_gb_cos_perpendicular_is_zero():
    assert gb_cos(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_gb_cos_parallel_is_one():
    assert gb_cos(np.array([1.0, 1.0, 0.0]), np.array([2.0, 2.0, 0.0])) == pytest.approx(1.0)


def test_gb_cos_normalization_invariant():
    rng = np.random.default_rng(42)
    for _ in range(20):
        q, b = rng.normal(size=3), rng.normal(size=3)
        assert gb_cos(q, b) == pytest.approx(gb_cos(3.7 * q, 0.2 * b))


def test_gb_visible_threshold_semantics():
    # angle(G,b)=90° → invisible at any positive threshold
    assert not gb_visible(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 10.0)
    # angle(G,b)=0° → visible
    assert gb_visible(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10.0)


def test_gb_visible_matches_pipeline_passes_invisibility():
    """The pipeline guard must delegate: identical verdicts on random input."""
    from dfxm_geo.pipeline import _passes_invisibility

    rng = np.random.default_rng(7)
    for _ in range(50):
        q, b = rng.normal(size=3), rng.normal(size=3)
        thr = float(rng.uniform(0.0, 45.0))
        assert _passes_invisibility(q, b, thr) == gb_visible(q, b, thr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gb_visibility.py -q`
Expected: FAIL — `ImportError: cannot import name 'gb_cos'`

- [ ] **Step 3: Implement in `burgers.py`**

Append (numpy is already imported in this module):

```python
def gb_cos(q_hkl: np.ndarray, b_vec: np.ndarray) -> float:
    """Normalized |cos∠(G, b)| = |G·b| / (|G||b|) — the g·b visibility scalar.

    0.0 means G ⊥ b (classic invisibility criterion g·b = 0 for screw
    dislocations); 1.0 means G ∥ b (maximum contrast).
    """
    q = np.asarray(q_hkl, dtype=float)
    b = np.asarray(b_vec, dtype=float)
    return float(abs(np.dot(q, b)) / (np.linalg.norm(q) * np.linalg.norm(b)))


def gb_visible(q_hkl: np.ndarray, b_vec: np.ndarray, threshold_deg: float) -> bool:
    """True when the dislocation is NOT within `threshold_deg` of invisibility.

    Bit-identical to the historical inline criterion in
    ``pipeline._passes_invisibility``: visible ⇔ cos∠(G,b) ≥ cos(90° − threshold).
    """
    return bool(gb_cos(q_hkl, b_vec) >= np.cos(np.deg2rad(90.0 - threshold_deg)))
```

Then replace the **body** of `pipeline._passes_invisibility` (keep its signature and docstring; it has call sites at `pipeline.py:1665` and `:2068`):

```python
def _passes_invisibility(q_hkl, b_vec, threshold_deg):
    # (existing docstring stays)
    from dfxm_geo.crystal.burgers import gb_visible

    return gb_visible(q_hkl, b_vec, threshold_deg)
```

Match the existing import style at the top of `pipeline.py` — if `dfxm_geo.crystal.burgers` is already imported module-level (it is, for `_burgers_vectors`), extend that import instead of a local import.

- [ ] **Step 4: Run the new test + the invisibility-adjacent existing tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gb_visibility.py -q && .\.venv\Scripts\python.exe -m pytest tests -q -k "invisib"`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/burgers.py src/dfxm_geo/pipeline.py tests/test_gb_visibility.py
git commit -m "refactor: extract gb_cos/gb_visible helpers from _passes_invisibility (bit-identical)"
```

---

### Task 2: `ReflectionRun` + resolver module `crystal/reflections.py`

**Files:**
- Create: `src/dfxm_geo/crystal/reflections.py`
- Test: `tests/test_reflections_resolver.py` (create)

The resolver is pure logic over `compute_omega_eta` — no I/O, no pipeline imports (avoids cycles; `pipeline.py` will import *it*).

- [ ] **Step 1: Write the failing tests**

```python
"""[[reflections]] / [reflections_auto] resolution — pure-logic layer."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.reflections import ReflectionRun, resolve_reflections

PAPER_MOUNT = CrystalMount(
    lattice="cubic", a=4.0493e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1)
)
KEV = 19.1
# Paper Table A.2 group 1: four {113}-family reflections share (θ=15.417°, η=20.233°)
GROUP1 = [(1, 1, 3), (-1, -1, 3), (1, -1, 3), (-1, 1, 3)]


def _entries(hkls, **common):
    return [dict(hkl=list(h), **common) for h in hkls]


def test_single_entry_defaults_to_solution_1():
    runs = resolve_reflections(_entries([(1, 1, 3)]), PAPER_MOUNT, KEV)
    assert len(runs) == 1
    r = runs[0]
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    assert r.hkl == (1, 1, 3)
    assert r.eta == pytest.approx(geom.eta_1)
    assert r.omega == pytest.approx(geom.omega_1)
    assert r.theta == pytest.approx(geom.theta_1)
    assert r.group == 0


def test_omega_solution_2_selects_second_branch():
    runs = resolve_reflections(
        _entries([(1, 1, 3)], omega_solution=2), PAPER_MOUNT, KEV
    )
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    assert runs[0].omega == pytest.approx(geom.omega_2)
    assert runs[0].eta == pytest.approx(geom.eta_2)


def test_explicit_eta_must_match_a_solution():
    with pytest.raises(ValueError, match="does not match"):
        resolve_reflections(_entries([(1, 1, 3)], eta=0.123456), PAPER_MOUNT, KEV)


def test_explicit_eta_picks_matching_solution():
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    runs = resolve_reflections(_entries([(1, 1, 3)], eta=geom.eta_2), PAPER_MOUNT, KEV)
    assert runs[0].omega == pytest.approx(geom.omega_2)


def test_group1_reflections_share_one_group():
    """The paper's η=20.233° quadruple must dedup to a single kernel group."""
    entries = []
    for h in GROUP1:
        geom = compute_omega_eta(PAPER_MOUNT, h, KEV)
        # pick whichever solution carries the shared η=0.3531 rad
        target = 0.3531
        sol = 1 if abs(geom.eta_1 - target) < abs(geom.eta_2 - target) else 2
        entries.append(dict(hkl=list(h), omega_solution=sol))
    runs = resolve_reflections(entries, PAPER_MOUNT, KEV)
    assert {r.group for r in runs} == {0}


def test_mixed_groups_get_distinct_ids():
    runs = resolve_reflections(_entries([(1, 1, 1), (2, 0, 0)]), PAPER_MOUNT, KEV)
    assert runs[0].group != runs[1].group
    # group ids are dense, ordered by first appearance
    assert sorted({r.group for r in runs}) == [0, 1]


def test_unreachable_reflection_raises_with_cli_hint():
    # (5,5,5) at 19.1 keV: sin θ > 1 → no Laue solution
    with pytest.raises(ValueError, match="dfxm-find-reflections"):
        resolve_reflections(_entries([(5, 5, 5)]), PAPER_MOUNT, KEV)


def test_empty_list_raises():
    with pytest.raises(ValueError, match=r"\[\[reflections\]\]"):
        resolve_reflections([], PAPER_MOUNT, KEV)


def test_default_eta_applies_to_entries_without_eta():
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    runs = resolve_reflections(
        _entries([(1, 1, 3)]), PAPER_MOUNT, KEV, default_eta=geom.eta_2
    )
    assert runs[0].omega == pytest.approx(geom.omega_2)


def test_reflection_run_is_frozen():
    runs = resolve_reflections(_entries([(1, 1, 3)]), PAPER_MOUNT, KEV)
    with pytest.raises(AttributeError):
        runs[0].eta = 0.0  # type: ignore[misc]
```

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_reflections_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm_geo.crystal.reflections'`

- [ ] **Step 3: Implement `src/dfxm_geo/crystal/reflections.py`**

```python
"""Resolve [[reflections]] / [reflections_auto] TOML tables into ReflectionRun records.

Pure logic over crystal.oblique — no I/O, no pipeline imports. See
docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta, find_reflections

# Matches the Phase-A bootstrap tolerance (kernel._validate_eta_against_compute_omega_eta):
# loose enough for paper-quoting precision, strict enough to catch typos.
ETA_MATCH_TOL = 1e-3
# Two solver outputs for a genuinely shared geometry agree to ~1e-12; 1e-6 rad
# groups them while keeping physically distinct geometries apart.
GROUP_TOL = 1e-6


@dataclass(frozen=True)
class ReflectionRun:
    """One fully resolved reflection of a multi-reflection run. Angles in radians."""

    hkl: tuple[int, int, int]
    keV: float
    eta: float
    theta: float
    omega: float
    group: int  # kernel-sharing group: same (theta, eta) within GROUP_TOL


def _resolve_entry(
    entry: dict,
    mount: CrystalMount,
    keV: float,
    default_eta: float | None,
) -> tuple[tuple[int, int, int], float, float, float]:
    """Resolve one [[reflections]] entry → (hkl, eta, theta, omega)."""
    if "hkl" not in entry:
        raise ValueError(f"[[reflections]] entry missing 'hkl': {entry!r}.")
    hkl = tuple(int(c) for c in entry["hkl"])
    if len(hkl) != 3:
        raise ValueError(f"[[reflections]] hkl must have 3 components, got {entry['hkl']!r}.")

    geom = compute_omega_eta(mount, hkl, keV)  # type: ignore[arg-type]
    if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
        raise ValueError(
            f"Laue condition unsatisfiable for hkl={hkl} at keV={keV} with this mount. "
            "Use 'dfxm-find-reflections' to enumerate reachable reflections."
        )
    solutions = {
        1: (geom.eta_1, geom.theta_1, geom.omega_1),
        2: (geom.eta_2, geom.theta_2, geom.omega_2),
    }

    eta_requested = entry.get("eta", default_eta)
    if eta_requested is not None:
        eta_requested = float(eta_requested)
        if "omega_solution" in entry:
            raise ValueError(
                f"[[reflections]] entry for hkl={hkl}: give 'eta' or 'omega_solution', "
                "not both (eta already selects the solution branch)."
            )
        for eta_i, theta_i, omega_i in solutions.values():
            if not np.isnan(eta_i) and abs(eta_i - eta_requested) <= ETA_MATCH_TOL:
                return hkl, float(eta_i), float(theta_i), float(omega_i)
        raise ValueError(
            f"eta={eta_requested:.6f} rad does not match either solution for hkl={hkl} "
            f"(η₁={geom.eta_1:.6f}, η₂={geom.eta_2:.6f}). No auto-correct; use "
            "'dfxm-find-reflections' to list valid (η, ω) pairs."
        )

    sol = int(entry.get("omega_solution", 1))
    if sol not in (1, 2):
        raise ValueError(f"omega_solution must be 1 or 2, got {sol} (hkl={hkl}).")
    eta_i, theta_i, omega_i = solutions[sol]
    if np.isnan(omega_i):
        raise ValueError(
            f"omega_solution={sol} does not exist for hkl={hkl} at keV={keV} "
            "(no real Laue branch). Use 'dfxm-find-reflections'."
        )
    return hkl, float(eta_i), float(theta_i), float(omega_i)


def _assign_groups(resolved: list[tuple[tuple[int, int, int], float, float, float]]) -> list[int]:
    """Dense group ids by (theta, eta) proximity, ordered by first appearance."""
    reps: list[tuple[float, float]] = []  # (theta, eta) representative per group
    groups: list[int] = []
    for _hkl, eta, theta, _omega in resolved:
        for gid, (t_rep, e_rep) in enumerate(reps):
            if abs(theta - t_rep) <= GROUP_TOL and abs(eta - e_rep) <= GROUP_TOL:
                groups.append(gid)
                break
        else:
            reps.append((theta, eta))
            groups.append(len(reps) - 1)
    return groups


def resolve_reflections(
    entries: list[dict],
    mount: CrystalMount,
    keV: float,
    *,
    default_eta: float | None = None,
) -> list[ReflectionRun]:
    """Resolve explicit [[reflections]] entries. See spec §5 for the rules."""
    if not entries:
        raise ValueError("[[reflections]] must contain at least one entry.")
    resolved = [_resolve_entry(e, mount, keV, default_eta) for e in entries]
    groups = _assign_groups(resolved)
    return [
        ReflectionRun(hkl=hkl, keV=float(keV), eta=eta, theta=theta, omega=omega, group=gid)
        for (hkl, eta, theta, omega), gid in zip(resolved, groups)
    ]


def resolve_reflections_auto(
    auto: dict,
    mount: CrystalMount,
    keV: float,
) -> list[ReflectionRun]:
    """Expand [reflections_auto] via find_reflections. One shared group by construction."""
    if "eta_target" not in auto:
        raise ValueError("[reflections_auto] requires 'eta_target' (radians).")
    eta_target = float(auto["eta_target"])
    kwargs: dict = {"eta_target": eta_target, "eta_tol": ETA_MATCH_TOL}
    if "theta_max" in auto:
        kwargs["theta_range"] = (0.0, float(auto["theta_max"]))
    if "hkl_max" in auto:
        kwargs["hkl_max"] = int(auto["hkl_max"])
    geoms = find_reflections(mount, keV, **kwargs)
    if not geoms:
        raise ValueError(
            f"[reflections_auto] eta_target={eta_target:.6f} rad matched no reflections. "
            "Run 'dfxm-find-reflections' to inspect the accessible groups."
        )
    entries = []
    for g in geoms:
        # pick the solution branch whose eta matched the target
        sol = 1 if (not np.isnan(g.eta_1) and abs(g.eta_1 - eta_target) <= ETA_MATCH_TOL) else 2
        entries.append({"hkl": list(g.hkl), "omega_solution": sol})
    return resolve_reflections(entries, mount, keV)
```

- [ ] **Step 4: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_reflections_resolver.py -q`
Expected: all PASS. If `test_unreachable_reflection_raises_with_cli_hint` fails because (5,5,5) IS reachable at 19.1 keV, pick a higher-order hkl, e.g. `(7, 7, 7)` — verify by calling `compute_omega_eta` in a scratch one-liner first.

- [ ] **Step 5: mypy + commit**

Run: `.\.venv\Scripts\python.exe -m mypy src/dfxm_geo/crystal/reflections.py`
Expected: 0 errors

```bash
git add src/dfxm_geo/crystal/reflections.py tests/test_reflections_resolver.py
git commit -m "feat: ReflectionRun resolver for [[reflections]]/[reflections_auto] (M3 schema core)"
```

---

### Task 3: Auto-expansion tests for `resolve_reflections_auto`

**Files:**
- Test: `tests/test_reflections_resolver.py` (extend)

- [ ] **Step 1: Write the failing/passing tests** (function exists from Task 2 — these pin its behavior)

```python
def test_auto_expands_paper_group1():
    """[reflections_auto] with the paper's η must recover the 4-reflection group."""
    runs = resolve_reflections_auto(
        {"eta_target": 0.3531, "hkl_max": 3}, PAPER_MOUNT, KEV
    )
    assert {r.hkl for r in runs} >= set(GROUP1)
    assert {r.group for r in runs} == {0}


def test_auto_zero_matches_raises():
    with pytest.raises(ValueError, match="matched no reflections"):
        resolve_reflections_auto({"eta_target": 1.5, "hkl_max": 2}, PAPER_MOUNT, KEV)


def test_auto_requires_eta_target():
    with pytest.raises(ValueError, match="eta_target"):
        resolve_reflections_auto({}, PAPER_MOUNT, KEV)
```

(Add `resolve_reflections_auto` to the import at the top of the test file.)

Note: `find_reflections` brute-forces (2·hkl_max+1)³ solver calls — keep `hkl_max ≤ 3` in tests for speed.

- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_reflections_resolver.py -q` — Expected: all PASS (fix the eta_target for the zero-match test if 1.5 rad accidentally matches something; any absurd η works).

- [ ] **Step 3: Commit**

```bash
git add tests/test_reflections_resolver.py
git commit -m "test: pin [reflections_auto] paper-group expansion + error paths"
```

---

### Task 4: Config-loader wiring (parse, validate, refuse-at-run)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` — `SimulationConfig` (line ~507), `SimulationConfig.from_toml` (~519), `load_identification_config` (~673), `_build_geometry_config` (~472), `run_simulation` (~861), `run_identification` (~2196)
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` — `_parse_geometry_block` (line 51) gains an `allow_missing_eta` escape
- Test: `tests/test_reflections_config.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
"""[[reflections]] config-loader wiring: parse, validate, refuse-at-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import SimulationConfig, load_identification_config

MOUNT_TOML = """
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def _multi_toml(extra_reciprocal: str = "", geometry_eta: str = "") -> str:
    return f"""
[reciprocal]
keV = 19.1
{extra_reciprocal}

[geometry]
mode = "oblique"
{geometry_eta}

[crystal]
{MOUNT_TOML}

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
"""


def test_reflections_parse_to_runs(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    assert len(cfg.reflections) == 2
    assert cfg.reflections[0].hkl == (1, 1, 3)
    assert cfg.reflections[0].theta == pytest.approx(cfg.reflections[1].theta)


def test_no_reflections_block_is_empty_list(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, "[reciprocal]\nkeV = 17.0\n"))
    assert cfg.reflections == []


def test_reflections_with_reciprocal_hkl_rejected(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        SimulationConfig.from_toml(
            _write(tmp_path, _multi_toml(extra_reciprocal="hkl = [1, 1, 1]"))
        )


def test_reflections_require_oblique(tmp_path):
    body = _multi_toml().replace('mode = "oblique"', 'mode = "simplified"')
    with pytest.raises(ValueError, match="oblique"):
        SimulationConfig.from_toml(_write(tmp_path, body))


def test_reflections_and_auto_mutually_exclusive(tmp_path):
    body = _multi_toml() + "\n[reflections_auto]\neta_target = 0.3531\n"
    with pytest.raises(ValueError, match="reflections_auto"):
        SimulationConfig.from_toml(_write(tmp_path, body))


def test_geometry_eta_optional_with_reflections(tmp_path):
    # No [geometry] eta — per-entry solution-1 defaults apply.
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    assert all(r.eta != 0.0 for r in cfg.reflections)


def test_geometry_eta_acts_as_default(tmp_path):
    cfg = SimulationConfig.from_toml(
        _write(tmp_path, _multi_toml(geometry_eta="eta = 0.3531"))
    )
    for r in cfg.reflections:
        assert r.eta == pytest.approx(0.3531, abs=1e-3)


def test_run_simulation_refuses_multi_reflection(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    from dfxm_geo.pipeline import run_simulation

    with pytest.raises(NotImplementedError, match="multi-reflection"):
        run_simulation(cfg)


def test_identification_loader_parses_reflections(tmp_path):
    body = _multi_toml() + """
[identification]
mode = "single"

[identification.single]
n_samples = 1
"""
    cfg = load_identification_config(_write(tmp_path, body))
    assert len(cfg.reflections) == 2
```

NOTE for the implementer: the `[identification]` stanza above is a sketch — open `configs/identification_single.toml` and copy the *minimal* valid identification block from it (the loader validates mode/sub-block consistency; smoke-size values only). Same for `run_identification` refusal: add a `test_run_identification_refuses_multi_reflection` mirroring the forward one once the loader test passes.

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_reflections_config.py -q`
Expected: FAIL — `AttributeError: 'SimulationConfig' object has no attribute 'reflections'` (and ValueError-matches failing)

- [ ] **Step 3: Implement loader wiring in `pipeline.py`**

3a. Imports: `from dfxm_geo.crystal.reflections import ReflectionRun, resolve_reflections, resolve_reflections_auto` (module level, near the existing `crystal` imports).

3b. New field on BOTH `SimulationConfig` and `IdentificationConfig`:

```python
    # v2.6.0 (M3): resolved multi-reflection runs; empty = single-reflection config.
    reflections: list[ReflectionRun] = field(default_factory=list)
```

3c. New shared parse helper (place next to `_build_geometry_config`):

```python
def _parse_reflections_tables(raw: dict, geometry: GeometryConfig, reciprocal: ReciprocalConfig) -> list[ReflectionRun]:
    """Resolve [[reflections]] / [reflections_auto] from raw TOML. Empty list when absent.

    Validation rules per spec §5: oblique-only, mutually exclusive with
    [reciprocal] hkl and with each other.
    """
    has_list = "reflections" in raw
    has_auto = "reflections_auto" in raw
    if not has_list and not has_auto:
        return []
    if has_list and has_auto:
        raise ValueError("[[reflections]] and [reflections_auto] are mutually exclusive.")
    if geometry.mode != "oblique":
        raise ValueError(
            "[[reflections]] / [reflections_auto] require [geometry] mode='oblique' "
            "(sweep simplified-mode reflections by emitting one config per hkl via "
            "the gen-sweep scripts instead)."
        )
    if "hkl" in raw.get("reciprocal", {}):
        raise ValueError(
            "[reciprocal] hkl and [[reflections]]/[reflections_auto] are mutually "
            "exclusive: list every reflection in the reflections table."
        )
    mount = geometry.mount
    assert mount is not None  # oblique mode guarantees a mount
    keV = reciprocal.keV
    if has_auto:
        return resolve_reflections_auto(raw["reflections_auto"], mount, keV)
    default_eta = raw.get("geometry", {}).get("eta")
    return resolve_reflections(
        raw["reflections"], mount, keV,
        default_eta=float(default_eta) if default_eta is not None else None,
    )
```

3d. **The eta-requirement wrinkle**: `_build_geometry_config` → `_parse_geometry_block` raises when oblique mode lacks `eta`, and `_validate_eta_against_compute_omega_eta` validates `eta` against `[reciprocal] hkl` — both wrong for multi-reflection configs (per-entry η, no single hkl). Extend `_build_geometry_config` with a `multi_reflection: bool = False` keyword: when True, parse mode + mount but skip the single-eta validation, returning `GeometryConfig(mode="oblique", eta=0.0, theta_validated=None, omega=0.0, mount=mount)` (per-reflection values live on the `ReflectionRun`s instead). In `kernel._parse_geometry_block`, add the matching `allow_missing_eta: bool = False` parameter (when True and `eta` missing in oblique mode, return `("oblique", float("nan"))` instead of raising); `_build_geometry_config` passes `allow_missing_eta=multi_reflection` and ignores the NaN. Callers determine the flag *before* building geometry: `multi = ("reflections" in raw) or ("reflections_auto" in raw)`.

3e. In `SimulationConfig.from_toml` and `load_identification_config`:

```python
        multi = ("reflections" in raw) or ("reflections_auto" in raw)
        geometry = _build_geometry_config(raw, reciprocal, multi_reflection=multi)
        reflections = _parse_reflections_tables(raw, geometry, reciprocal)
```

…and pass `reflections=reflections` into the constructed config. (In `load_identification_config` the raw dict variable is named `data`.) Keep the existing `reciprocal.eta = geometry.eta` propagation under `if geometry.mode == "oblique" and not reflections:` — the analytic-backend eta handoff stays single-reflection-only for now.

3f. Refuse-at-run guards (top of `run_simulation` and `run_identification`):

```python
    if config.reflections:
        raise NotImplementedError(
            "multi-reflection orchestration is not wired yet (M3 plan 2, pending "
            "the omega-handling decision in docs/superpowers/specs/"
            "2026-06-10-m3-multi-reflection-sweeps-design.md §3). Bootstrap and "
            "config validation work; forward/identify of [[reflections]] configs "
            "land in the next arc."
        )
```

- [ ] **Step 4: Run the new tests, then the FULL suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_reflections_config.py -q` → all PASS
Run: `.\.venv\Scripts\python.exe -m pytest -q` → failure set unchanged vs the baseline recorded at branch start (GATE ON THE FULL SUITE, not -k subsets — see [[session-handoff-2026-06-03-v240-forwardcontext-paused-before-deletion]]).

- [ ] **Step 5: mypy + commit**

Run: `.\.venv\Scripts\python.exe -m mypy src/dfxm_geo/` → 0 errors

```bash
git add src/dfxm_geo/pipeline.py src/dfxm_geo/reciprocal_space/kernel.py tests/test_reflections_config.py
git commit -m "feat: parse+validate [[reflections]]/[reflections_auto] in both config loaders; orchestrators refuse until plan 2"
```

---

### Task 5: `dfxm-find-reflections` CLI

**Files:**
- Create: `src/dfxm_geo/find_reflections_cmd.py`
- Modify: `pyproject.toml:71-77` (`[project.scripts]`)
- Test: `tests/test_find_reflections_cli.py` (create)

CLI shape mirrors `init_cmd.py` (small argparse main). Tests call `cli_main(argv)` directly — no subprocess, no entry-point install dependency.

- [ ] **Step 1: Write the failing tests**

```python
"""dfxm-find-reflections CLI — Table A.2-style enumeration."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.find_reflections_cmd import cli_main

PAPER_CONFIG = """
[reciprocal]
keV = 19.1

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


@pytest.fixture
def paper_config(tmp_path):
    p = tmp_path / "paper.toml"
    p.write_text(PAPER_CONFIG, encoding="utf-8")
    return p


def test_table_a2_group1_rows_present(paper_config, capsys):
    rc = cli_main(["--config", str(paper_config), "--hkl-max", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    # Paper Table A.2 group 1: η = 20.233°, θ = 15.417° — all four {113} variants
    for hkl_str in ("1 1 3", "-1 -1 3", "1 -1 3", "-1 1 3"):
        assert hkl_str in out
    assert "20.23" in out  # eta, degrees
    assert "15.41" in out  # theta, degrees


def test_eta_target_filters(paper_config, capsys):
    rc = cli_main([
        "--config", str(paper_config),
        "--eta-target-deg", "20.233", "--hkl-max", "3",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 1 3" in out
    assert "1 1 1" not in out  # different group, filtered out


def test_missing_config_errors(tmp_path, capsys):
    rc = cli_main(["--config", str(tmp_path / "nope.toml")])
    assert rc != 0


def test_default_mount_when_no_crystal_block(tmp_path, capsys):
    p = tmp_path / "min.toml"
    p.write_text("[reciprocal]\nkeV = 19.1\n", encoding="utf-8")
    rc = cli_main(["--config", str(p), "--hkl-max", "2"])
    assert rc == 0
    assert "keV" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: No module named 'dfxm_geo.find_reflections_cmd'`

- [ ] **Step 3: Implement `src/dfxm_geo/find_reflections_cmd.py`**

```python
"""dfxm-find-reflections: enumerate accessible reflections for a crystal mount.

Reproduces the paper's Table A.2 (Detlefs et al. 2025 / arXiv:2503.22022
Appendix A) for the mount + keV given in a config TOML. Wires the
(Phase-A-tested) crystal.oblique.find_reflections solver to the command line.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

from dfxm_geo.crystal.oblique import find_reflections
from dfxm_geo.crystal.reflections import GROUP_TOL
from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dfxm-find-reflections",
        description="Enumerate Laue-accessible reflections (theta, eta, omega) for "
        "the crystal mount and beam energy in a config TOML.",
    )
    parser.add_argument("--config", required=True, help="TOML with [crystal] mount + [reciprocal] keV")
    parser.add_argument("--keV", type=float, default=None, help="override [reciprocal] keV")
    parser.add_argument("--hkl-max", type=int, default=5, help="max |Miller index| (default 5)")
    parser.add_argument("--theta-max-deg", type=float, default=16.25, help="max Bragg angle (default 16.25)")
    parser.add_argument("--eta-target-deg", type=float, default=None, help="keep only the group at this eta")
    parser.add_argument("--eta-tol-deg", type=float, default=0.06, help="eta-target tolerance (default 0.06)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2
    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

    mount = _crystal_mount_from_toml(raw.get("crystal"))
    keV = args.keV if args.keV is not None else float(raw.get("reciprocal", {}).get("keV", 17.0))

    kwargs: dict = {
        "theta_range": (0.0, float(np.deg2rad(args.theta_max_deg))),
        "hkl_max": args.hkl_max,
    }
    if args.eta_target_deg is not None:
        kwargs["eta_target"] = float(np.deg2rad(args.eta_target_deg))
        kwargs["eta_tol"] = float(np.deg2rad(args.eta_tol_deg))
    geoms = find_reflections(mount, keV, **kwargs)

    print(f"# mount: x={mount.mount_x} y={mount.mount_y} z={mount.mount_z}  a={mount.a:g} m  keV={keV:g}")
    print(f"{'hkl':>10} {'theta_deg':>10} {'eta1_deg':>10} {'omega1_deg':>11} {'eta2_deg':>10} {'omega2_deg':>11} {'group':>6}")
    reps: list[tuple[float, float]] = []
    for g in geoms:
        theta = g.theta_1 if not np.isnan(g.theta_1) else g.theta_2
        eta = g.eta_1 if not np.isnan(g.eta_1) else g.eta_2
        gid = -1
        for i, (t_rep, e_rep) in enumerate(reps):
            if abs(theta - t_rep) <= GROUP_TOL and abs(eta - e_rep) <= GROUP_TOL:
                gid = i
                break
        if gid < 0:
            reps.append((float(theta), float(eta)))
            gid = len(reps) - 1
        hkl_str = " ".join(str(c) for c in g.hkl)
        print(
            f"{hkl_str:>10} {np.degrees(theta):>10.3f} "
            f"{np.degrees(g.eta_1):>10.3f} {np.degrees(g.omega_1):>11.3f} "
            f"{np.degrees(g.eta_2):>10.3f} {np.degrees(g.omega_2):>11.3f} {gid:>6}"
        )
    print(f"# {len(geoms)} reflections, {len(reps)} kernel group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
```

(NaN values print as `nan` via numpy degrees — acceptable; do NOT crash on them.)

3b. `pyproject.toml` — add to `[project.scripts]` alphabetically:

```toml
dfxm-find-reflections = "dfxm_geo.find_reflections_cmd:cli_main"
```

3c. Re-run editable install so the entry point exists locally: `.\.venv\Scripts\python.exe -m pip install --quiet -e ".[dev]"`

- [ ] **Step 4: Run tests** — `.\.venv\Scripts\python.exe -m pytest tests/test_find_reflections_cli.py -q` → all PASS. If group-1 hkl strings mismatch (sign rendering), adjust the test to the actual `g.hkl` tuple rendering — verify against `find_reflections` output, do not weaken to substring-of-digits.

- [ ] **Step 5: mypy + commit**

```bash
git add src/dfxm_geo/find_reflections_cmd.py pyproject.toml tests/test_find_reflections_cli.py
git commit -m "feat: dfxm-find-reflections CLI (Table A.2 enumeration; closes roadmap G1.3)"
```

**RELEASE NOTE (do NOT do now):** at v2.6.0 release time the conda-forge feedstock recipe MUST gain this entry point in `build.python.entry_points` — Windows-launcher lesson, see CLAUDE.md recipe-sync checklist.

---

### Task 6: Multi-reflection `dfxm-bootstrap` loop + kernel manifest

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` — `cli_main` (lines 377-626)
- Test: `tests/test_bootstrap_multi_reflection.py` (create)

Read `cli_main` in full before editing — it already handles `--config/--output/--force/--if-missing/--seed` and single-reflection oblique validation; the multi-reflection branch loops *unique groups* and writes `kernel_manifest.toml`. Kernel generation is expensive — tests MUST monkeypatch `generate_kernel` (assert on calls, not real npz generation; pattern exists in `tests/test_kernel_cli_both_validated.py`).

- [ ] **Step 1: Write the failing tests**

```python
"""dfxm-bootstrap [[reflections]] loop: one kernel per unique (theta, eta, keV) group."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import dfxm_geo.reciprocal_space.kernel as kernel_mod

GROUP1_TOML = """
[reciprocal]
keV = 19.1

[geometry]
mode = "oblique"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
"""

MIXED_TOML = GROUP1_TOML + """
[[reflections]]
hkl = [1, 1, 1]
"""


@pytest.fixture
def fake_generate(monkeypatch, tmp_path):
    calls: list[dict] = []

    def _fake(output_path=None, **kwargs):
        calls.append({"output_path": output_path, **kwargs})
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-npz")
        return p

    monkeypatch.setattr(kernel_mod, "generate_kernel", _fake)
    return calls


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_same_group_bootstraps_one_kernel(fake_generate, tmp_path):
    rc = kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, GROUP1_TOML)), "--output", str(tmp_path / "k")]
    )
    assert rc in (None, 0)
    assert len(fake_generate) == 1  # ONE kernel for the shared (theta, eta) group


def test_mixed_groups_bootstrap_per_group(fake_generate, tmp_path):
    kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, MIXED_TOML)), "--output", str(tmp_path / "k")]
    )
    assert len(fake_generate) == 2  # {113}-group + 111


def test_manifest_written_next_to_kernels(fake_generate, tmp_path):
    kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, MIXED_TOML)), "--output", str(tmp_path / "k")]
    )
    manifest = tmp_path / "k" / "kernel_manifest.toml"
    assert manifest.is_file()
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["kernels"]) == 2
    hkls = {tuple(h) for entry in data["kernels"] for h in entry["reflections"]}
    assert (1, 1, 3) in hkls and (1, 1, 1) in hkls
    for entry in data["kernels"]:
        assert {"group", "theta", "eta", "keV", "filename", "reflections", "omegas"} <= set(entry)
```

ADAPT during implementation: `cli_main`'s exact return convention and `--output` semantics (directory vs file path) must be read from the existing code first; mirror the single-reflection behavior. If `--output` is a file path in single mode, multi mode treats it as a *directory* and errors if it names a file.

- [ ] **Step 2: Run to verify failure** — currently `cli_main` reads `[reciprocal] hkl` (absent here) and either defaults to Al-111 or errors; the calls list won't match. Expected: FAIL.

- [ ] **Step 3: Implement the multi-reflection branch in `cli_main`**

Structure (adapt names to the real function body):

```python
    # after raw TOML load + [geometry]/[crystal] parsing:
    multi = ("reflections" in raw) or ("reflections_auto" in raw)
    if multi:
        from dfxm_geo.crystal.reflections import (
            resolve_reflections,
            resolve_reflections_auto,
        )

        if mode != "oblique":
            raise ValueError(
                "[[reflections]] / [reflections_auto] require [geometry] mode='oblique'."
            )
        keV = float(raw.get("reciprocal", {}).get("keV", 17.0))
        if "reflections_auto" in raw:
            runs = resolve_reflections_auto(raw["reflections_auto"], mount, keV)
        else:
            default_eta = raw.get("geometry", {}).get("eta")
            runs = resolve_reflections(
                raw["reflections"], mount, keV,
                default_eta=float(default_eta) if default_eta is not None else None,
            )
        out_dir = Path(args.output) if args.output else _default_kernel_dir()  # mirror single-mode default
        manifest_entries = []
        for gid in sorted({r.group for r in runs}):
            members = [r for r in runs if r.group == gid]
            rep = members[0]
            # --if-missing: reuse the existing oblique lookup; skip group if it resolves
            kernel_path = out_dir / _build_kernel_filename(
                "oblique", rep.hkl, keV, theta=rep.theta, eta=rep.eta, date=date_str
            )
            generate_kernel(
                output_path=kernel_path, mode="oblique", eta=rep.eta,
                mount=mount, omega=rep.omega, hkl=rep.hkl, keV=keV,
                **reciprocal_kwargs,
            )
            manifest_entries.append({
                "group": gid,
                "theta": rep.theta,
                "eta": rep.eta,
                "keV": keV,
                "filename": kernel_path.name,
                "reflections": [list(m.hkl) for m in members],
                "omegas": [m.omega for m in members],
            })
        _write_kernel_manifest(out_dir / "kernel_manifest.toml", manifest_entries)
        return 0
```

…with a small `_write_kernel_manifest(path, entries)` helper that hand-renders TOML (the repo has no TOML *writer* dependency — render with f-strings; nested arrays-of-arrays are `[[1, 1, 3], [-1, -1, 3]]`):

```python
def _write_kernel_manifest(path: Path, entries: list[dict]) -> None:
    """Render kernel_manifest.toml (no tomli-w dependency; hand-rendered)."""
    lines = ["# generated by dfxm-bootstrap — one row per unique (theta, eta, keV) kernel group", ""]
    for e in entries:
        lines += [
            "[[kernels]]",
            f"group = {e['group']}",
            f"theta = {e['theta']!r}",
            f"eta = {e['eta']!r}",
            f"keV = {e['keV']!r}",
            f'filename = "{e["filename"]}"',
            "reflections = [" + ", ".join(str(list(h)) for h in e["reflections"]) + "]",
            "omegas = [" + ", ".join(repr(o) for o in e["omegas"]) + "]",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
```

(`{e['theta']!r}` on a float renders `0.2691...` — valid TOML. Verify by round-tripping through `tomllib.loads` in the test.)

Honor `--if-missing` and `--force` in the loop with the same semantics as single mode (read how single mode does the existence check and reuse it per group). The `generate_kernel` signature MUST be called exactly as single-reflection oblique mode calls it — copy that call and parameterize.

- [ ] **Step 4: Run tests** — new file passes; then `.\.venv\Scripts\python.exe -m pytest tests -q -k "kernel or bootstrap"` — failure set unchanged.

- [ ] **Step 5: mypy + commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_bootstrap_multi_reflection.py
git commit -m "feat: dfxm-bootstrap loops [[reflections]] groups, one kernel per (theta,eta,keV) + kernel_manifest.toml"
```

---

### Task 7: Sweep-generator reflection axis (`--hkl-list`, `--keV`)

**Files:**
- Modify: `scripts/gen_identify_sweep_configs.py` (argparse exists; extend)
- Modify: `scripts/gen_sweep_configs.py` (no argparse today; add, preserving no-arg behavior)
- Test: `tests/test_gen_sweep_reflection_axis.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
"""Reflection axis in the sweep generators (--hkl-list / --keV)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_identify_sweep_configs  # noqa: E402


def test_hkl_list_multiplies_configs(tmp_path):
    gen_identify_sweep_configs.main([
        "--n-configs", "2", "--out-dir", str(tmp_path),
        "--hkl-list", "1,1,1;2,0,0",
    ])
    tomls = sorted(tmp_path.glob("*.toml"))
    assert len(tomls) == 4  # 2 seeds x 2 reflections
    hkls = set()
    for p in tomls:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        hkls.add(tuple(data["reciprocal"]["hkl"]))
        assert "hkl" in p.stem  # filename carries the reflection token
    assert hkls == {(1, 1, 1), (2, 0, 0)}


def test_kev_override(tmp_path):
    gen_identify_sweep_configs.main([
        "--n-configs", "1", "--out-dir", str(tmp_path), "--keV", "19.1",
    ])
    p = next(iter(tmp_path.glob("*.toml")))
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["reciprocal"]["keV"] == pytest.approx(19.1)


def test_default_behavior_unchanged(tmp_path):
    """No --hkl-list → legacy filenames (no hkl token) and Al-111 defaults."""
    gen_identify_sweep_configs.main(["--n-configs", "1", "--out-dir", str(tmp_path)])
    p = next(iter(tmp_path.glob("*.toml")))
    assert "hkl" not in p.stem
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert tuple(data["reciprocal"]["hkl"]) == (-1, 1, -1)
    assert data["reciprocal"]["keV"] == pytest.approx(17.0)
```

PRE-CHECK: open `gen_identify_sweep_configs.py` — if `main()` doesn't take an argv parameter, change `def main():` → `def main(argv=None):` and `parser.parse_args()` → `parser.parse_args(argv)` (this is the only structural change the test forces; existing CLI behavior is unaffected). If existing generator tests exist (grep `tests/` for `gen_identify`), mirror their import style instead of the `sys.path` shim.

- [ ] **Step 2: Run to verify failure** — `unrecognized arguments: --hkl-list`

- [ ] **Step 3: Implement**

In `gen_identify_sweep_configs.py`:

```python
def _parse_hkl_list(spec: str) -> list[tuple[int, int, int]]:
    """'1,1,1;2,0,0' → [(1,1,1), (2,0,0)]."""
    out = []
    for token in spec.split(";"):
        parts = [int(x) for x in token.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--hkl-list entry must have 3 indices, got {token!r}")
        out.append(tuple(parts))
    return out
```

- argparse: `parser.add_argument("--hkl-list", default=None, help="semicolon-separated reflections, e.g. '1,1,1;2,0,0' — adds a reflection axis to the sweep")` and `parser.add_argument("--keV", type=float, default=17.0)`.
- `config_text(...)` gains `hkl: tuple[int,int,int]` and `keV: float` parameters; the hardcoded `hkl = [-1, 1, -1]` / `keV = 17.0` lines (currently ~58-59) become f-string interpolations: `hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]` / `keV = {keV}`. Defaults preserve today's output byte-for-byte.
- Emission loop: `reflections = _parse_hkl_list(args.hkl_list) if args.hkl_list else [(-1, 1, -1)]`; when `--hkl-list` given, filenames gain a token: `multi_hkl{h}{k}{l}_seed{s:05d}.toml` with negative indices rendered as `m` (e.g. `(-1,1,-1)` → `hklm11m1`) to stay filesystem-safe. No `--hkl-list` → legacy `multi_seed{s:05d}.toml` exactly.

Apply the same `--hkl-list`/`--keV` + filename-token pattern to `gen_sweep_configs.py` (it first needs a `main(argv=None)` + argparse wrapper around its current constants; running with no args must emit today's 8 files with identical names and contents — verify by diffing one generated file against a pre-change copy).

- [ ] **Step 4: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_gen_sweep_reflection_axis.py -q` → PASS; plus any existing fanout/gen tests: `.\.venv\Scripts\python.exe -m pytest tests -q -k "fanout or sweep"` — failure set unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_identify_sweep_configs.py scripts/gen_sweep_configs.py tests/test_gen_sweep_reflection_axis.py
git commit -m "feat: --hkl-list/--keV reflection axis in both sweep generators (cross-config path)"
```

---

### Task 8: Full-suite gates

**Files:** none (verification only)

- [ ] **Step 1:** `.\.venv\Scripts\python.exe -m pytest -q` — compare against the baseline failure SET recorded at branch start (expect: `test_render_readme_examples_smoke` pre-existing failure only, kernel-dependent tests green with the copied kernel npz).
- [ ] **Step 2:** `.\.venv\Scripts\python.exe -m mypy src/dfxm_geo/` — 0 errors.
- [ ] **Step 3:** `pre-commit run --all-files` (ruff/format) — clean.
- [ ] **Step 4:** Commit any gate fixes individually; do NOT bundle into feature commits.

---

## Plan 2 preview (NOT in this plan — blocked on spec §3 ω decision)

Orchestrator loop in `run_simulation`/`run_identification` (replacing the NotImplementedError), `_reflection_projection` (B′: `R_z(ω) @ Us` at the `precompute_forward_static` seam), per-reflection masters + `dfxm_geo_multi.h5` super-master, g·b labels in identify HDF5, invisibility physics smoke test, `--reflections-toml` generator flag, Hg-hoist perf pass. Write it via superpowers:writing-plans once Sina answers §3 (default B′ if no answer).
