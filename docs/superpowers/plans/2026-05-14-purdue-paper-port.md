# Purdue_Paper Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `S` sample-remount rotation matrix from `origin/Purdue_Paper` into `cleanup/main-modernization`. After this lands, `dfxm-forward` accepts `[crystal].sample_remount = "S1" | "S2" | "S3" | "S4"` in its TOML, and the strain field is rotated through the named sample-remount matrix before reaching the crystal frame.

**Architecture:** New `dfxm_geo.crystal.remount` module holds the four named constants and a `SAMPLE_REMOUNT_OPTIONS` dict. `Fd_find` / `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` / `load_or_generate_Hg` / `Find_Hg` all gain a keyword-only `S` parameter defaulting to `np.identity(3)` (backward-compatible). `CrystalConfig` gains a `sample_remount: str` field; `run_simulation` resolves the name → matrix and threads it down. Fg cache filenames gain `_remount{name}` to keep S1/S2/S3/S4 caches separate. Paper-figure code from the Purdue branch goes to `legacy/init_forward_purdue.py` as a frozen reference.

**Tech Stack:** Python 3.11+, NumPy, pytest (existing). No new deps.

**Spec:** `docs/superpowers/specs/2026-05-14-purdue-paper-port-design.md` (commit `cb3676d`).

---

## File map

**Create:**
- `src/dfxm_geo/crystal/remount.py` — constants + options dict (~35 LoC)
- `tests/test_remount.py` — 4 tests
- `legacy/init_forward_purdue.py` — verbatim copy of Purdue branch `init_forward.py` + header
- `configs/variants/sample_remount_S2.toml` — minimal reproducer / smoke fixture

**Modify:**
- `src/dfxm_geo/crystal/dislocations.py` — `Fd_find`, `Fd_find_mixed`, `Fd_find_multi_dislocs_mixed` signatures + body
- `src/dfxm_geo/io/strain_cache.py` — `load_or_generate_Hg` signature
- `src/dfxm_geo/direct_space/forward_model.py` — `Find_Hg` signature + cache filename
- `src/dfxm_geo/pipeline.py` — `CrystalConfig` field, `run_simulation` resolution
- `tests/test_dislocations.py` — 3 new tests, kwarg-only assertion
- `tests/test_dislocations_mixed.py` — 2 new tests
- `tests/test_io.py` — 1 new test
- `tests/test_pipeline.py` — 4 new tests, update `fake_find_hg` signature
- `configs/default.toml` — add `sample_remount = "S1"` under `[crystal]`
- `docs/architecture.md` — note on the goniometer frame
- `docs/physics.md` — note on S1–S4

---

### Task 1: New module `crystal/remount.py` with constants + tests

**Files:**
- Create: `src/dfxm_geo/crystal/remount.py`
- Create: `tests/test_remount.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_remount.py
"""Tests for the sample-remount rotation matrices (Purdue 2024 paper)."""

import numpy as np
import pytest

from dfxm_geo.crystal.remount import (
    S1,
    S2,
    S3,
    S4,
    SAMPLE_REMOUNT_OPTIONS,
)


def test_constants_are_proper_rotations() -> None:
    """Each S_i must be orthogonal with det=+1 (proper rotation)."""
    for name, S in [("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)]:
        np.testing.assert_allclose(
            S.T @ S, np.identity(3), atol=1e-10, err_msg=f"{name} not orthogonal"
        )
        assert np.isclose(np.linalg.det(S), 1.0, atol=1e-10), f"{name} det != +1"


def test_S1_is_identity() -> None:
    np.testing.assert_array_equal(S1, np.identity(3))


def test_S2_S3_S4_match_purdue_source_verbatim() -> None:
    """Pin the literal values from origin/Purdue_Paper direct_space/forward_model.py.

    Regression test: catches accidental edits to the constants.
    """
    np.testing.assert_array_equal(
        S2,
        np.array(
            [
                [1 / 3, -2 / 3, -2 / 3],
                [2 / 3, -1 / 3, 2 / 3],
                [-2 / 3, -2 / 3, 1 / 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        S3,
        np.array(
            [
                [1 / 3, -2 / 3, 2 / 3],
                [2 / 3, 2 / 3, 1 / 3],
                [-2 / 3, 1 / 3, 2 / 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        S4,
        np.array(
            [
                [1 / 3, 2 / 3, 2 / 3],
                [2 / 3, 1 / 3, -2 / 3],
                [-2 / 3, 2 / 3, -1 / 3],
            ]
        ),
    )


def test_sample_remount_options_map_is_complete() -> None:
    """Map keys exactly match the public names; values are the same array objects."""
    assert set(SAMPLE_REMOUNT_OPTIONS.keys()) == {"S1", "S2", "S3", "S4"}
    assert SAMPLE_REMOUNT_OPTIONS["S1"] is S1
    assert SAMPLE_REMOUNT_OPTIONS["S2"] is S2
    assert SAMPLE_REMOUNT_OPTIONS["S3"] is S3
    assert SAMPLE_REMOUNT_OPTIONS["S4"] is S4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_remount.py -v`
Expected: `ImportError: cannot import name 'S1' from 'dfxm_geo.crystal.remount'` (module not yet created).

- [ ] **Step 3: Create `src/dfxm_geo/crystal/remount.py`**

```python
"""Sample-remount rotation matrices (Purdue 2024 paper).

S_i is applied between the sample frame (after Theta) and the crystal frame
(before Us.T) in the Fd_find rotation chain. Operationally: S rotates the
entire sample relative to the goniometer, simulating physical remounting at
a symmetry-equivalent orientation.

S1 = identity (no remount); S2, S3, S4 are three specific cubic-symmetry
proper rotations from the Purdue 2024 paper. They are ported verbatim from
the branch source; their numerical traces are not all equal — S2 and S4
have trace 1/3 (~109.47 deg rotations) and S3 has trace 5/3 (~70.53 deg)
— so they are NOT three rotations about a single axis; they are independent
group elements selected for the paper's specific remount scenarios.
"""

import numpy as np

S1: np.ndarray = np.identity(3)

S2: np.ndarray = np.array(
    [
        [1 / 3, -2 / 3, -2 / 3],
        [2 / 3, -1 / 3, 2 / 3],
        [-2 / 3, -2 / 3, 1 / 3],
    ]
)

S3: np.ndarray = np.array(
    [
        [1 / 3, -2 / 3, 2 / 3],
        [2 / 3, 2 / 3, 1 / 3],
        [-2 / 3, 1 / 3, 2 / 3],
    ]
)

S4: np.ndarray = np.array(
    [
        [1 / 3, 2 / 3, 2 / 3],
        [2 / 3, 1 / 3, -2 / 3],
        [-2 / 3, 2 / 3, -1 / 3],
    ]
)

SAMPLE_REMOUNT_OPTIONS: dict[str, np.ndarray] = {
    "S1": S1,
    "S2": S2,
    "S3": S3,
    "S4": S4,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_remount.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/remount.py tests/test_remount.py
git commit -m "feat(crystal): sample-remount rotation matrices (S1-S4)"
```

---

### Task 2: Thread `S` through `Fd_find`

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (function `Fd_find` at line 109)
- Modify: `tests/test_dislocations.py`

- [ ] **Step 1: Add the failing tests to `tests/test_dislocations.py`**

Append at the end of the file:

```python
# --- Sample-remount (S) ---

class TestFdFindSampleRemount:
    """Tests for the S kwarg added by the Purdue_Paper port."""

    def _build_inputs(self, n: int = 8):
        lin = np.linspace(-1.0, 1.0, n)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
        rl = grid.reshape(3, -1)
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)
        return rl, Ud, Us, Theta

    def test_S_kwarg_default_matches_omitted(self) -> None:
        """Fd_find(..., S=identity) must equal Fd_find(...) with S omitted."""
        from dfxm_geo.crystal.dislocations import Fd_find

        rl, Ud, Us, Theta = self._build_inputs()
        without = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=3)
        with_explicit_I = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=3, S=np.identity(3))
        np.testing.assert_array_equal(without, with_explicit_I)

    def test_S2_yields_distinct_output(self) -> None:
        """S=S2 must produce a different Fg than S=identity on a non-trivial rl."""
        from dfxm_geo.crystal.dislocations import Fd_find
        from dfxm_geo.crystal.remount import S2

        rl, Ud, Us, Theta = self._build_inputs()
        with_I = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=3, S=np.identity(3))
        with_S2 = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=3, S=S2)
        # Difference must be non-zero somewhere.
        assert not np.allclose(with_I, with_S2)

    def test_S_kwarg_is_keyword_only(self) -> None:
        """Positional arg in the S slot must not silently rebind to S."""
        from dfxm_geo.crystal.dislocations import Fd_find

        rl, Ud, Us, Theta = self._build_inputs()
        # Eighth positional arg should be `b` (Burgers vector), not `S`.
        # Pass a non-matrix scalar that would error if mistaken for S.
        result = Fd_find(rl, Ud, Us, Theta, 1.0, 3, 3.0e-4)  # b=3.0e-4
        assert result.shape == (rl.shape[1], 3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dislocations.py::TestFdFindSampleRemount -v`
Expected: `TypeError: Fd_find() got an unexpected keyword argument 'S'` for the first two tests; the third should pass (b accepted positionally — pre-port behaviour).

- [ ] **Step 3: Modify `Fd_find` in `src/dfxm_geo/crystal/dislocations.py`**

Change the signature at line 109 from:

```python
def Fd_find(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float = 1,
    ndis: int = 1,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    misorientation: bool = False,
    t_vec: np.ndarray | None = None,
) -> np.ndarray:
```

to:

```python
def Fd_find(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float = 1,
    ndis: int = 1,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    misorientation: bool = False,
    t_vec: np.ndarray | None = None,
    *,
    S: np.ndarray = np.identity(3),
) -> np.ndarray:
```

Then change the rotation chain at lines 155-157 from:

```python
    rs = Theta @ rl
    rc = Us.T @ rs
    rd = Ud.T @ rc
```

to:

```python
    rs = Theta @ rl
    rgon = S.T @ rs  # sample-remount (Purdue 2024); S = identity → rgon == rs
    rc = Us.T @ rgon
    rd = Ud.T @ rc
```

(Also extend the existing `Args:` block in the docstring with one line for `S`: `S: 3x3 rotation matrix (sample-remount; default identity).`)

- [ ] **Step 4: Run all dislocations tests to verify nothing regressed**

Run: `pytest tests/test_dislocations.py tests/test_dislocations_smoke.py -v`
Expected: All previously-passing tests still pass; the 3 new tests pass too. The `Fd_find_smoke` golden test confirms bit-exact backward-compat at S=I.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations.py
git commit -m "feat(crystal): Fd_find accepts sample-remount S kwarg"
```

---

### Task 3: Thread `S` through `Fd_find_mixed`

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (function `Fd_find_mixed` at line 216)
- Modify: `tests/test_dislocations_mixed.py`

- [ ] **Step 1: Add the failing tests to `tests/test_dislocations_mixed.py`**

Append at the end of the file:

```python
# --- Sample-remount (S) for mixed-character ---

class TestFdFindMixedSampleRemount:
    """Tests for the S kwarg on Fd_find_mixed (Purdue_Paper port)."""

    def _inputs(self):
        rl = np.linspace(-1.0, 1.0, 12).reshape(1, -1)
        rl = np.vstack([rl, rl, rl])  # (3, 12)
        Us = np.eye(3)
        Theta = np.eye(3)
        # Ud_mix: arbitrary proper rotation. Pick one whose columns are unit
        # length (Eq.3 of Borgi 2025 takes (b, n, t) columns).
        Ud_mix = np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
            ]
        )
        return rl, Us, Ud_mix, Theta

    def test_S_kwarg_default_matches_omitted(self) -> None:
        from dfxm_geo.crystal.dislocations import Fd_find_mixed

        rl, Us, Ud_mix, Theta = self._inputs()
        without = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta)
        with_I = Fd_find_mixed(
            rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=np.identity(3)
        )
        np.testing.assert_array_equal(without, with_I)

    def test_S2_yields_distinct_output(self) -> None:
        from dfxm_geo.crystal.dislocations import Fd_find_mixed
        from dfxm_geo.crystal.remount import S2

        rl, Us, Ud_mix, Theta = self._inputs()
        with_I = Fd_find_mixed(
            rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=np.identity(3)
        )
        with_S2 = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=S2)
        assert not np.allclose(with_I, with_S2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dislocations_mixed.py::TestFdFindMixedSampleRemount -v`
Expected: `TypeError: Fd_find_mixed() got an unexpected keyword argument 'S'`.

- [ ] **Step 3: Modify `Fd_find_mixed` in `src/dfxm_geo/crystal/dislocations.py`**

Change the signature at line 216 from:

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
```

to:

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
    S: np.ndarray = np.identity(3),
) -> np.ndarray:
```

Then change the rotation chain at lines 280-282 from:

```python
    rs = Theta @ rl
    rc = Us.T @ rs
    rd = Ud_mix.T @ rc
```

to:

```python
    rs = Theta @ rl
    rgon = S.T @ rs  # sample-remount (Purdue 2024); S = identity → rgon == rs
    rc = Us.T @ rgon
    rd = Ud_mix.T @ rc
```

(Add a docstring `Args:` line: `S: 3x3 rotation matrix (sample-remount; default identity).`)

- [ ] **Step 4: Run all mixed tests to verify nothing regressed**

Run: `pytest tests/test_dislocations_mixed.py -v`
Expected: All previously-passing tests still pass; the 2 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations_mixed.py
git commit -m "feat(crystal): Fd_find_mixed accepts sample-remount S kwarg"
```

---

### Task 4: Thread `S` through `Fd_find_multi_dislocs_mixed`

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (function `Fd_find_multi_dislocs_mixed` at line 335)
- Modify: `tests/test_dislocations_mixed.py`

- [ ] **Step 1: Add the failing test to `tests/test_dislocations_mixed.py`**

Append after the `TestFdFindMixedSampleRemount` class:

```python
class TestFdFindMultiDislocsMixedSampleRemount:
    def test_S2_threads_through_to_inner_Fd_find_mixed(self) -> None:
        """Fd_find_multi_dislocs_mixed(S=S2) must differ from S=I."""
        from dfxm_geo.crystal.dislocations import (
            Fd_find_multi_dislocs_mixed,
            MixedDislocSpec,
        )
        from dfxm_geo.crystal.remount import S2

        rl = np.linspace(-1.0, 1.0, 12).reshape(1, -1)
        rl = np.vstack([rl, rl, rl])
        Us = np.eye(3)
        Theta = np.eye(3)
        Ud_mix = np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
            ]
        )
        specs = [
            MixedDislocSpec(Ud_mix=Ud_mix, rotation_deg=30.0),
            MixedDislocSpec(
                Ud_mix=Ud_mix, rotation_deg=60.0, position_lab_um=(1.0, 0.0, 0.0)
            ),
        ]

        with_I = Fd_find_multi_dislocs_mixed(
            rl, Us, specs, Theta, S=np.identity(3)
        )
        with_S2 = Fd_find_multi_dislocs_mixed(rl, Us, specs, Theta, S=S2)
        assert not np.allclose(with_I, with_S2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dislocations_mixed.py::TestFdFindMultiDislocsMixedSampleRemount -v`
Expected: `TypeError: Fd_find_multi_dislocs_mixed() got an unexpected keyword argument 'S'`.

- [ ] **Step 3: Modify `Fd_find_multi_dislocs_mixed`**

Change the signature at line 335 from:

```python
def Fd_find_multi_dislocs_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    crystals: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
```

to:

```python
def Fd_find_multi_dislocs_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    crystals: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    S: np.ndarray = np.identity(3),
) -> np.ndarray:
```

Then update the inner call (currently at line 369-378) to thread S through:

```python
        Fg_one = Fd_find_mixed(
            rl,
            Us,
            Ud_mix=spec.Ud_mix,
            rotation_deg=spec.rotation_deg,
            Theta=Theta,
            b=b,
            ny=ny,
            position_lab_um=spec.position_lab_um,
            S=S,
        )
```

(Add a docstring `Args:` line: `S: 3x3 rotation matrix (sample-remount; default identity).`)

- [ ] **Step 4: Run all mixed tests**

Run: `pytest tests/test_dislocations_mixed.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations_mixed.py
git commit -m "feat(crystal): Fd_find_multi_dislocs_mixed threads S to inner call"
```

---

### Task 5: Thread `S` through `load_or_generate_Hg`

**Files:**
- Modify: `src/dfxm_geo/io/strain_cache.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Add the failing test to `tests/test_io.py`**

Append at end of file (the file already imports numpy, pytest, etc — keep imports tidy if you add new ones):

```python
class TestLoadOrGenerateHgSampleRemount:
    """The S kwarg threads through to Fd_find, producing different Fg."""

    def _rl(self, n: int = 8) -> np.ndarray:
        lin = np.linspace(-1.0, 1.0, n)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
        return grid.reshape(3, -1)

    def test_S_kwarg_default_matches_omitted(self, tmp_path) -> None:
        """Calling with S=identity must equal calling without S."""
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        rl = self._rl()
        Ud = Us = Theta = np.eye(3)

        path_a = str(tmp_path / "fg_a.npy")
        path_b = str(tmp_path / "fg_b.npy")

        Hg_omitted = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=path_a
        )
        Hg_with_I = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=path_b, S=np.identity(3)
        )
        np.testing.assert_array_equal(Hg_omitted, Hg_with_I)

    def test_S2_yields_distinct_Hg(self, tmp_path) -> None:
        """S=S2 must produce a different Hg than S=identity on the same inputs."""
        from dfxm_geo.crystal.remount import S2
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        rl = self._rl()
        Ud = Us = Theta = np.eye(3)

        Hg_I = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=3, file_path=str(tmp_path / "fg_I.npy")
        )
        Hg_S2 = load_or_generate_Hg(
            rl,
            Ud,
            Us,
            Theta,
            dis=1.0,
            ndis=3,
            file_path=str(tmp_path / "fg_S2.npy"),
            S=S2,
        )
        assert not np.allclose(Hg_I, Hg_S2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::TestLoadOrGenerateHgSampleRemount -v`
Expected: `TypeError: load_or_generate_Hg() got an unexpected keyword argument 'S'`.

- [ ] **Step 3: Modify `load_or_generate_Hg` in `src/dfxm_geo/io/strain_cache.py`**

Change the function signature at line 9 from:

```python
def load_or_generate_Hg(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float,
    ndis: int,
    file_path: str | None = None,
) -> np.ndarray:
```

to:

```python
def load_or_generate_Hg(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float,
    ndis: int,
    file_path: str | None = None,
    *,
    S: np.ndarray = np.identity(3),
) -> np.ndarray:
```

Then update the inner `Fd_find` call at line 50 from:

```python
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)
```

to:

```python
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis, S=S)
```

Update the docstring with one line: `S: 3x3 rotation matrix (sample-remount; default identity).`

- [ ] **Step 4: Run all io tests**

Run: `pytest tests/test_io.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/io/strain_cache.py tests/test_io.py
git commit -m "feat(io): load_or_generate_Hg threads sample-remount S"
```

---

### Task 6: Thread `S` + `remount_name` through `Find_Hg`, encode in cache filename

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (function `Find_Hg` at line 119)
- Modify: `tests/test_forward_model_smoke.py`

- [ ] **Step 1: Inspect the existing smoke test** so the new test slots in alongside it

Run: `head -50 tests/test_forward_model_smoke.py` to see the existing pattern. The new test will follow the same monkeypatch approach.

- [ ] **Step 2: Add the failing test to `tests/test_forward_model_smoke.py`**

Append at the end:

```python
class TestFindHgSampleRemount:
    """Find_Hg passes the resolved S and remount_name through correctly."""

    def test_find_hg_passes_S_and_filename_to_load_or_generate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Fg cache filename includes _remount{name} and S kwarg arrives."""
        import dfxm_geo.direct_space.forward_model as fm
        from dfxm_geo.crystal.remount import S2

        captured: dict = {}

        def fake_load(rl, Ud, Us, Theta, dis, ndis, file_path=None, *, S=None):
            captured["file_path"] = file_path
            captured["S"] = S
            # Return a plausibly-shaped Fg-derived Hg
            return np.zeros((rl.shape[1], 3, 3))

        monkeypatch.setattr(
            "dfxm_geo.direct_space.forward_model.load_or_generate_Hg", fake_load
        )

        Hg, q_hkl = fm.Find_Hg(
            dis=4, ndis=2, psize=fm.psize, zl_rms=fm.zl_rms,
            S=S2, remount_name="S2",
        )

        assert captured["file_path"] is not None
        assert "_remountS2.npy" in captured["file_path"]
        np.testing.assert_array_equal(captured["S"], S2)

    def test_find_hg_default_uses_S1_filename(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting S / remount_name defaults to identity / 'S1'."""
        import dfxm_geo.direct_space.forward_model as fm

        captured: dict = {}

        def fake_load(rl, Ud, Us, Theta, dis, ndis, file_path=None, *, S=None):
            captured["file_path"] = file_path
            captured["S"] = S
            return np.zeros((rl.shape[1], 3, 3))

        monkeypatch.setattr(
            "dfxm_geo.direct_space.forward_model.load_or_generate_Hg", fake_load
        )

        fm.Find_Hg(dis=4, ndis=2, psize=fm.psize, zl_rms=fm.zl_rms)

        assert "_remountS1.npy" in captured["file_path"]
        np.testing.assert_array_equal(captured["S"], np.identity(3))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_forward_model_smoke.py::TestFindHgSampleRemount -v`
Expected: `TypeError: Find_Hg() got an unexpected keyword argument 'S'` (or similar).

- [ ] **Step 4: Modify `Find_Hg` in `src/dfxm_geo/direct_space/forward_model.py`**

Change the signature at line 119 from:

```python
def Find_Hg(
    dis: float,
    ndis: int,
    psize: float,
    zl_rms: float,
    h: int = -1,
    k: int = 1,
    l: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
```

to:

```python
def Find_Hg(
    dis: float,
    ndis: int,
    psize: float,
    zl_rms: float,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = np.identity(3),
    remount_name: str = "S1",
) -> tuple[np.ndarray, np.ndarray]:
```

Change the `Fg_path` construction at lines 152-163 from:

```python
    Fg_path = str(
        _REPO_ROOT
        / "direct_space"
        / "deformation_gradient_tensors"
        / "Fg_{}_{}nm_{}nm_px{}_sub{}.npy".format(
            str(dis).replace(".", ""),
            int(psize * 1e9),
            int(zl_rms * 2.35e9),
            Npixels,
            Nsub,
        )
    )
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, Fg_path)
```

to:

```python
    Fg_path = str(
        _REPO_ROOT
        / "direct_space"
        / "deformation_gradient_tensors"
        / "Fg_{}_{}nm_{}nm_px{}_sub{}_remount{}.npy".format(
            str(dis).replace(".", ""),
            int(psize * 1e9),
            int(zl_rms * 2.35e9),
            Npixels,
            Nsub,
            remount_name,
        )
    )
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, Fg_path, S=S)
```

Also extend the `vars` dict at line 167 with two entries so the sidecar `_vars.txt` records the remount:

```python
        vars = {
            "Resq_i": pkl_fn,
            "psize [nm]": psize,
            "zl_rms": zl_rms,
            "theta_0 [rad]": theta_0,
            "Npixels": Npixels,
            "Nsub": Nsub,
            "Ud": Ud.tolist(),
            "Us": Us.tolist(),
            "Theta": Theta.tolist(),
            "ndis": ndis,
            "dis [micrometer]": dis,
            "q_hkl": q_hkl.tolist(),
            "S_remount_name": remount_name,
            "S_remount_matrix": S.tolist(),
        }
```

Update the docstring with two new `Args:` lines: `S: 3x3 sample-remount rotation (default identity).` and `remount_name: Name used in the Fg cache filename (default "S1").`

- [ ] **Step 5: Run forward-model tests**

Run: `pytest tests/test_forward_model_smoke.py tests/test_forward_model_paths.py -v`
Expected: All pass (including the 2 new tests).

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_model_smoke.py
git commit -m "feat(forward-model): Find_Hg threads S; cache filename gains _remount suffix"
```

---

### Task 7: `CrystalConfig.sample_remount` field + validation

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add the failing tests to `tests/test_pipeline.py`**

Insert a new test class after the existing `TestSimulationConfig` block. The exact insertion point is after the line that ends `TestSimulationConfig` (find by searching for `class TestPostprocess` — insert above that):

```python
class TestCrystalConfigSampleRemount:
    """Tests for the sample_remount field on CrystalConfig."""

    def test_default_is_S1(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        cfg = CrystalConfig()
        assert cfg.sample_remount == "S1"

    def test_accepts_S2_S3_S4(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        for name in ("S2", "S3", "S4"):
            cfg = CrystalConfig(sample_remount=name)
            assert cfg.sample_remount == name

    def test_rejects_unknown_remount(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        with pytest.raises(ValueError, match="sample_remount must be one of"):
            CrystalConfig(sample_remount="S99")

    def test_toml_round_trip_with_sample_remount(self, tmp_path: Path) -> None:
        """TOML containing sample_remount round-trips through SimulationConfig."""
        from dfxm_geo.pipeline import SimulationConfig

        toml_text = """
[crystal]
dis = 4.0
ndis = 151
sample_remount = "S3"

[scan]
phi_range = 0.05
phi_steps = 5
chi_range = 0.05
chi_steps = 5
"""
        path = tmp_path / "cfg.toml"
        path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(path)
        assert cfg.crystal.sample_remount == "S3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::TestCrystalConfigSampleRemount -v`
Expected: `TypeError: CrystalConfig() got an unexpected keyword argument 'sample_remount'`.

- [ ] **Step 3: Modify `CrystalConfig` in `src/dfxm_geo/pipeline.py`**

Add an import near the top of the file (with the other `dfxm_geo.crystal.*` imports — after the `MixedDislocSpec` import block):

```python
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
```

Then change the `CrystalConfig` dataclass at line 46 from:

```python
@dataclass
class CrystalConfig:
    dis: float = 4.0  # inter-dislocation distance (µm)
    ndis: int = 151  # number of dislocations
```

to:

```python
@dataclass
class CrystalConfig:
    dis: float = 4.0  # inter-dislocation distance (µm)
    ndis: int = 151  # number of dislocations
    sample_remount: str = "S1"  # one of S1/S2/S3/S4; Purdue 2024 paper

    def __post_init__(self) -> None:
        if self.sample_remount not in SAMPLE_REMOUNT_OPTIONS:
            valid = ", ".join(SAMPLE_REMOUNT_OPTIONS.keys())
            raise ValueError(
                f"sample_remount must be one of: {valid} "
                f"(got {self.sample_remount!r})"
            )
```

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All pass (existing tests unaffected because they don't pass `sample_remount`; default "S1" is valid).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): CrystalConfig.sample_remount field + validation"
```

---

### Task 8: `run_simulation` resolves and threads `S`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (function `run_simulation` at line 257)
- Modify: `tests/test_pipeline.py` (update `fake_find_hg` signature)

- [ ] **Step 1: Add the failing test to `tests/test_pipeline.py`**

Insert at the end of the existing `TestRunSimulation` class (before its closing — find the last method in that class and add after it):

```python
    def test_run_simulation_passes_resolved_S_to_find_hg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sample_remount='S2' must arrive at fm.Find_Hg as the S2 matrix."""
        from dfxm_geo.crystal.remount import S2

        config = SimulationConfig(
            crystal=CrystalConfig(dis=2.0, ndis=4, sample_remount="S2"),
            scan=ScanConfig(phi_range=0.05, phi_steps=5, chi_range=0.05, chi_steps=5),
            io=IOConfig(),
        )

        captured: dict = {}
        fake_Hg = np.ones((3, 3, 4, 4, 4))
        fake_q = np.array([0.0, 0.0, 1.0])

        def fake_find_hg(dis, ndis, psize, zl_rms, **kwargs):
            captured["kwargs"] = kwargs
            return fake_Hg, fake_q

        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Find_Hg", fake_find_hg)
        monkeypatch.setattr(
            "dfxm_geo.pipeline.save_images_parallel", lambda *a, **k: True
        )
        monkeypatch.setattr("dfxm_geo.pipeline.fm.psize", 0.1)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_rms", 1.0)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.q_hkl", None)

        run_simulation(config, tmp_path / "out")

        assert "S" in captured["kwargs"]
        np.testing.assert_array_equal(captured["kwargs"]["S"], S2)
        assert captured["kwargs"]["remount_name"] == "S2"
```

Also update the existing `fake_find_hg` in `test_golden_path_writes_both_stacks` and `test_skip_perfect_crystal` (and any other place it appears in this file). Its signature currently is `def fake_find_hg(dis, ndis, psize, zl_rms):` — change to `def fake_find_hg(dis, ndis, psize, zl_rms, **kwargs):` in every occurrence so it tolerates the new kwargs once `run_simulation` starts passing them.

Run a search to confirm you got them all: `grep -n "def fake_find_hg" tests/test_pipeline.py`.

- [ ] **Step 2: Run the new test to verify it fails**

Run: `pytest tests/test_pipeline.py::TestRunSimulation::test_run_simulation_passes_resolved_S_to_find_hg -v`
Expected: KeyError or AssertionError because `run_simulation` does not yet pass `S` / `remount_name` to `Find_Hg`.

- [ ] **Step 3: Modify `run_simulation` in `src/dfxm_geo/pipeline.py`**

Change line 271 from:

```python
    Hg, q_hkl = fm.Find_Hg(config.crystal.dis, config.crystal.ndis, fm.psize, fm.zl_rms)
```

to:

```python
    S = SAMPLE_REMOUNT_OPTIONS[config.crystal.sample_remount]
    Hg, q_hkl = fm.Find_Hg(
        config.crystal.dis,
        config.crystal.ndis,
        fm.psize,
        fm.zl_rms,
        S=S,
        remount_name=config.crystal.sample_remount,
    )
```

- [ ] **Step 4: Run all pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All pass — both the new wiring test and the existing tests (the latter pass because we made their `fake_find_hg` accept `**kwargs`).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): run_simulation threads sample_remount to Find_Hg"
```

---

### Task 9: TOML config defaults and `sample_remount_S2.toml` variant

**Files:**
- Modify: `configs/default.toml`
- Create: `configs/variants/sample_remount_S2.toml`
- Modify: `tests/test_pipeline.py` (extend `test_all_shipped_variants_parse` to cover the new variant)

- [ ] **Step 1: Read the existing default config** to copy its structure

Run: `cat configs/default.toml`

(Inspect to see how the `[crystal]` block is structured.)

- [ ] **Step 2: Add `sample_remount = "S1"` to `configs/default.toml`**

Under the `[crystal]` section in `configs/default.toml`, add:

```toml
sample_remount = "S1"  # one of "S1", "S2", "S3", "S4" — Purdue 2024 paper. Default: identity (no remount).
```

- [ ] **Step 3: Create `configs/variants/sample_remount_S2.toml`**

Copy `configs/default.toml` to `configs/variants/sample_remount_S2.toml`, then change the `sample_remount` line to `sample_remount = "S2"`. The rest of the file (scan, io, postprocess) should be identical to `default.toml`. Add a comment at the top:

```toml
# Variant: sample remounted in orientation S2.
# Equivalent to default.toml except for the [crystal].sample_remount field.
# Used by the Purdue 2024 paper to compare the same defect across symmetry-
# equivalent mountings.
```

- [ ] **Step 4: Verify the variant parses**

The existing test `test_all_shipped_variants_parse` in `tests/test_pipeline.py` iterates over `configs/variants/*.toml` and asserts each parses through `SimulationConfig.from_toml`. The new file is auto-discovered — confirm it works:

Run: `pytest tests/test_pipeline.py::TestSimulationConfig::test_all_shipped_variants_parse -v`
Expected: PASS — the variant is parsed and `crystal.sample_remount == "S2"` follows naturally.

If the existing test does not assert on `sample_remount`, that is fine — Task 7's `test_toml_round_trip_with_sample_remount` already pins the round-trip.

- [ ] **Step 5: Commit**

```bash
git add configs/default.toml configs/variants/sample_remount_S2.toml
git commit -m "feat(configs): add sample_remount field; S2 variant"
```

---

### Task 10: CLI end-to-end smoke (kernel-gated)

**Files:**
- Modify: `tests/test_pipeline.py` (add a subprocess-based test) or create `tests/test_cli_sample_remount.py` if you prefer a separate file

- [ ] **Step 1: Inspect Round 16's CLI smoke for the pattern**

Run: `grep -n "subprocess\|dfxm-forward\|kernel_path" tests/test_pipeline_identification.py | head -20`

The Round 16 pattern: skip the test if the kernel pickle is not present at the canonical default location, otherwise run `dfxm-forward` via subprocess against a TOML and assert the output dir is non-empty.

- [ ] **Step 2: Append the new test**

Add to `tests/test_pipeline.py` (at the end of the file, after the existing class definitions):

```python
class TestDfxmForwardSampleRemountCLI:
    """End-to-end CLI smoke: dfxm-forward with sample_remount=S2."""

    def test_dfxm_forward_with_sample_remount_S2_runs(self, tmp_path: Path) -> None:
        import shutil
        import subprocess
        from pathlib import Path as _P

        # Skip if the Resq_i kernel pickle is not on disk — same gating pattern
        # as Round 16's CLI smoke. The path mirrors what forward_model loads.
        repo_root = _P(__file__).resolve().parents[1]
        kernel_path = (
            repo_root
            / "reciprocal_space"
            / "pkl_files"
            / "Resq_i_20230913_1308.pkl"
        )
        if not kernel_path.exists():
            pytest.skip(
                f"Kernel pickle {kernel_path} not present; skipping CLI smoke."
            )

        # Run dfxm-forward with the S2 variant config, output to tmp_path
        variant_config = repo_root / "configs" / "variants" / "sample_remount_S2.toml"
        result = subprocess.run(
            [
                shutil.which("dfxm-forward") or "dfxm-forward",
                "--config",
                str(variant_config),
                "--output",
                str(tmp_path / "out"),
                "--no-postprocess",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        # CLI should succeed
        assert result.returncode == 0, (
            f"dfxm-forward failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Output dir was populated
        dislocs_dir = tmp_path / "out" / "images10"
        assert dislocs_dir.is_dir(), "dislocs images dir missing"
        assert any(dislocs_dir.iterdir()), "dislocs images dir empty"
```

(CLI flags verified at plan-writing time: `--config`, `--output`, `--no-postprocess` are all real; the entry point is `dfxm-forward` registered via `[project.scripts]` in `pyproject.toml`.)

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_pipeline.py::TestDfxmForwardSampleRemountCLI -v`
Expected: PASS or SKIP (skip if the kernel pickle is absent on this machine).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test(cli): dfxm-forward end-to-end smoke with sample_remount=S2"
```

---

### Task 11: `legacy/init_forward_purdue.py` verbatim copy + header

**Files:**
- Create: `legacy/init_forward_purdue.py`

- [ ] **Step 1: Dump the Purdue branch's `init_forward.py` to disk**

```bash
git show origin/Purdue_Paper:init_forward.py > legacy/init_forward_purdue.py
```

- [ ] **Step 2: Prepend a header to the file**

Open `legacy/init_forward_purdue.py` in an editor (or use a sed/edit-tool insert at the top). Add this block at the top, before the existing first line:

```python
"""Frozen reference: init_forward.py from origin/Purdue_Paper (commit cb254d2).

This file is the verbatim demo / paper-figure script from the Purdue_Paper
branch (Sept 2024). It is preserved here for reference — to reproduce paper
figures or audit the original physics — but is NOT maintained:

- It is not imported by anything.
- It is not lint-clean (will fail ruff if added to the lint scope).
- It is not tested.
- It contains hardcoded `/dtu/...` filesystem paths from the original author.
- It loads `Test_S1_Hg_rg.npy`, which is not shipped with this repo.

The substantive physics — sample-remount rotation matrices S1-S4 and their
plumbing through Fd_find / Find_Hg — was ported into the main package in the
Purdue_Paper port (see docs/superpowers/specs/2026-05-14-purdue-paper-port-design.md).
Use `dfxm-forward` with `[crystal].sample_remount = "S2"` etc. for new work.
"""

# ---- ORIGINAL CONTENT BELOW (unmodified) ----

```

Save the file.

- [ ] **Step 3: Confirm it does not get imported and tests still pass**

Run: `pytest -q`
Expected: All tests pass. `legacy/` is already excluded from both `ruff` (`extend-exclude = ["legacy/", "notebooks/"]` in `pyproject.toml:89`) and `mypy` (`exclude = ["legacy/", "notebooks/", "tests/"]` in `pyproject.toml:106`), so the new file is automatically out of scope for both. No `pyproject.toml` change needed.

- [ ] **Step 4: Commit**

```bash
git add legacy/init_forward_purdue.py
git commit -m "docs(legacy): Purdue_Paper init_forward.py as frozen reference"
```

---

### Task 12: Docs — note goniometer frame and S1–S4 constants

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/physics.md`

- [ ] **Step 1: Locate the rotation-chain section in `docs/architecture.md`**

Run: `grep -n "rotation chain\|Theta\|Us.T\|frame" docs/architecture.md | head -20`

Find the part of `architecture.md` that describes the lab → sample → crystal → dislocation rotation chain. (If no such section exists, find a natural place — e.g. under "module layout" or "data flow".)

- [ ] **Step 2: Add a note about the goniometer frame**

Insert a short paragraph (4–6 sentences):

```markdown
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
```

- [ ] **Step 3: Add an entry to `docs/physics.md`**

Locate the section that describes the four reference frames (Theta, Us, Ud) and append:

```markdown
### Sample remount (Purdue 2024)

In addition to the lab/sample/crystal/dislocation frames, `Fd_find` supports a
sample-remount rotation `S` inserted between sample and crystal frames. This
models a physical operation: the sample is removed from the goniometer and
remounted in a different (symmetry-equivalent) orientation. The four named
constants `S1` (identity), `S2`, `S3`, `S4` are ported verbatim from the
Purdue paper (`dfxm_geo.crystal.remount`). They are proper rotations from the
cubic point group; their numerical traces differ (S2 and S4 are ~109.47°
rotations; S3 is ~70.53°) so they are not three rotations about a single
axis. Cleanup ports them as-is and does not re-derive their geometric
interpretation.
```

- [ ] **Step 4: Run lint and tests as a sanity check**

Run: `ruff check . && pytest -q`
Expected: clean / all green.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md docs/physics.md
git commit -m "docs: note sample-remount S matrices in architecture and physics"
```

---

## Final verification

After all 12 tasks are complete:

- [ ] Run the full test suite: `pytest -q` — expected: ~14 more tests than before (rough breakdown: 4 remount + 3 dislocations + 3 mixed + 2 io + 4 pipeline + 2 forward-model smoke + 1 CLI smoke; total ~19 if all land, less if `test_all_shipped_variants_parse` already auto-covers the new variant — that is the expected case).
- [ ] Run lint: `ruff check . && ruff format --check .` — expected: clean.
- [ ] Run mypy: `mypy src/dfxm_geo/` — expected: 0 errors.
- [ ] Confirm pre-commit hooks pass on a clean re-run: `pre-commit run --all-files`.
- [ ] Spot-check: `dfxm-forward --help` should still work, and the `[crystal].sample_remount` field should appear when running with `configs/variants/sample_remount_S2.toml`.

If the CLI smoke (Task 10) was SKIPPED rather than PASSED, the kernel pickle was not available. Re-run that test manually after the kernel is in place to fully validate the end-to-end path.
