# Multi-reflection kernel lookup implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `dfxm-forward` and `dfxm-identify` read `(hkl, keV)` from the `[reciprocal]` TOML block, look up the matching kernel npz from `pkl_files/` by globbing, verify bundled metadata matches, and run with that kernel — removing the single hardcoded `pkl_fn` constant.

**Architecture:** Add a pure `_lookup_kernel_path` helper to `forward_model.py`; extend `_load_default_kernel` with verification kwargs; bundle `hkl`/`keV` in the npz metadata in `generate_kernel`; add a `ReciprocalConfig` dataclass to `pipeline.py` and field-add it to `SimulationConfig` + `IdentificationConfig`; replace `_ensure_kernel_loaded()` with `_lookup_and_load_kernel(hkl, keV)`; delete the `pkl_fn` constant + module-import auto-load; update `io/hdf5.py` + `io/migrate.py` to read the actually-loaded basename from a new `fm._loaded_kernel_path` attribute.

**Tech Stack:** Python 3.11+, `numpy`, `pathlib.glob`, `tomllib` (stdlib), `pytest` with `monkeypatch` + `capsys`, `mypy --strict`.

**Spec:** `docs/superpowers/specs/2026-05-20-multi-reflection-lookup-design.md` (commit `bacab41`).

---

## Parallel-dispatch hints (for subagent-driven execution)

Per `[[feedback-parallelize-subagents]]`:

- **Tasks 1, 2, 3 can run in parallel** (different files, no conflict): T1 touches `kernel.py` + `test_kernel_cli.py`; T2 touches `forward_model.py` + new `test_kernel_lookup.py`; T3 touches `pipeline.py` + `test_pipeline.py`.
- **Task 4 depends on Task 3** (the configs use the new `[reciprocal]` schema validated by Task 3's parser).
- **Task 5 depends on Tasks 2 + 3 + 4** (composes lookup helper + ReciprocalConfig).
- **Task 6 depends on Task 5** (callers updated before constant removed).
- **Task 7 (verification) depends on everything.**

The plan is linear; the executor dispatches tasks per these dependencies.

---

## File Structure

**Modify:**
- `src/dfxm_geo/reciprocal_space/kernel.py` — `generate_kernel` adds `hkl`/`keV` to `kernel_meta`; `cli_main` passes them through (~5 LOC).
- `src/dfxm_geo/direct_space/forward_model.py` — add `_lookup_kernel_path` (~25 LOC); add `_loaded_kernel_path` module attr; extend `_load_default_kernel` with verification kwargs (~25 LOC); delete `pkl_fn` constant + auto-load (lines 59-64, 423-428). Net ~+30 LOC after deletions.
- `src/dfxm_geo/pipeline.py` — add `ReciprocalConfig` dataclass (~30 LOC); field-add to `SimulationConfig` + `IdentificationConfig`; extend `from_toml` parsers; replace `_ensure_kernel_loaded` with `_lookup_and_load_kernel`; rewire callers + effective-config print (~+50 LOC).
- `src/dfxm_geo/io/hdf5.py` — line 414: read `fm._loaded_kernel_path` instead of synthesizing from `fm.pkl_fn` (~3 LOC swap).
- `src/dfxm_geo/io/migrate.py` — line 154: same swap (~3 LOC).
- `configs/identification_single.toml`, `configs/identification_multi.toml`, `configs/identification_zscan.toml` — add minimal `[reciprocal]` block (~3 lines each).

**Create:**
- `tests/test_kernel_lookup.py` — pure-function tests for `_lookup_kernel_path` + load verification (~250 LOC).
- `tests/test_pipeline_multi_reflection.py` — integration tests via `pipeline.run_simulation` and `run_identification` (~200 LOC).

**Update:**
- `tests/test_kernel_cli.py` — extend `TestGenerateKernelOutputPath` with the new-metadata assertion; remove obsolete `TestPklFnRegression` class (Task 6).
- `tests/test_pipeline.py` — rename all `_ensure_kernel_loaded` → `_lookup_and_load_kernel`; update existing kernel-load tests to use the new signature; add `[reciprocal]` to the sample_remount TOML fixture in `TestDfxmForwardSampleRemountCLI` (Task 6).
- `tests/test_hdf5_provenance.py:51` — verify HDF5 dataset name `pkl_fn` still exists (the dataset name doesn't change, only the Python attribute is gone) (Task 6).
- `tests/test_migrate_output.py:36, 72` — replace `monkeypatch.setattr(_fm, "pkl_fn", ...)` with `monkeypatch.setattr(_fm, "_loaded_kernel_path", Path(...))` (Task 6).
- `tests/test_cluster_templates.py:200` — docstring reference; no functional change (Task 6).

**Total**: ~600-700 LOC (~250 production, ~450 tests).

**Working directory:** `C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master/`
**Branch:** `chore/spec-multi-reflection-lookup` (current). Code commits land on this branch; spec commit `bacab41` already on it.
**Python interpreter:** `C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe` (bash `python` is 2.7 — do NOT use it).

---

## Task 1: `generate_kernel` bundles `hkl` + `keV` in npz metadata

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (`generate_kernel` signature + `kernel_meta` dict; `cli_main` pass-through)
- Test: `tests/test_kernel_cli.py` (append one method to `TestGenerateKernelOutputPath`)

- [ ] **Step 1: Write the failing test**

Append to `TestGenerateKernelOutputPath` in `tests/test_kernel_cli.py`:

```python
    def test_npz_bundles_hkl_and_keV_metadata(self, tmp_path: Path) -> None:
        """Sub-project D: the npz must include `hkl` and `keV` as bundled scalars
        so downstream load verification can confirm filename ↔ contents agree."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "Resq_i_h2_k0_l0_17keV_test.npz"
        generate_kernel(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            output_path=out,
            hkl=(2, 0, 0),
            keV=17.0,
        )
        loaded = np.load(out)
        assert "hkl" in loaded.files
        assert "keV" in loaded.files
        assert tuple(int(x) for x in loaded["hkl"]) == (2, 0, 0)
        assert float(loaded["keV"]) == 17.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestGenerateKernelOutputPath::test_npz_bundles_hkl_and_keV_metadata -v
```
Expected: FAIL — `generate_kernel` doesn't accept `hkl`/`keV` kwargs yet (or accepts them but doesn't bundle).

- [ ] **Step 3: Add `hkl` and `keV` kwargs to `generate_kernel`**

In `src/dfxm_geo/reciprocal_space/kernel.py`, modify `generate_kernel` to accept `hkl` and `keV` as keyword-only arguments. Find the function signature (currently around line 32) and add to the kwargs list:

```python
def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    # ... existing kwargs ...
    output_path: Path | None = None,
    hkl: tuple[int, int, int] | None = None,
    keV: float | None = None,
) -> Path:
```

In the docstring, add lines explaining the new args:
```
        hkl: Miller indices of the reflection (optional). Bundled into the
            npz scalar metadata for downstream load verification (sub-project D).
        keV: beam energy in keV (optional). Bundled into the npz scalar
            metadata for downstream load verification (sub-project D).
```

In the `kernel_meta` dict (currently around line 87), add the two new entries AFTER `dphi_range`:

```python
    kernel_meta = {
        # ... existing 20 keys ...
        "dphi_range": np.float64(dphi_range),
        # Sub-project D: reflection identity. Verified by
        # _load_default_kernel when a lookup expects a specific (hkl, keV).
        "hkl": np.array(hkl if hkl is not None else (0, 0, 0), dtype=np.int64),
        "keV": np.float64(keV if keV is not None else 0.0),
    }
```

(Sentinel `(0,0,0)` / `0.0` is a placeholder for the rare case `generate_kernel` is called without hkl/keV — downstream `_load_default_kernel` only verifies if `expected_hkl`/`expected_keV` are passed, so a sentinel never produces a silent wrong answer.)

- [ ] **Step 4: Update `cli_main` to pass hkl/keV through**

Still in `kernel.py`, find the `cli_main` block where `generate_kernel(output_path=output_path, **reciprocal_kwargs)` is called (currently around line 254). Inject the validated `hkl_tuple` and `keV_for_filename` (which exist in `cli_main`'s scope from sub-project A's wiring) into the kwargs:

```python
    reciprocal_kwargs["hkl"] = hkl_tuple
    reciprocal_kwargs["keV"] = keV_for_filename
    written = generate_kernel(output_path=output_path, **reciprocal_kwargs)
```

- [ ] **Step 5: Run tests to verify pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py -q
```
Expected: all kernel-cli tests pass (the new one + the existing 42, total 43).

Also run mypy:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli.py
git commit -m "feat(kernel): bundle hkl + keV in generate_kernel npz metadata (sub-project D)"
```

---

## Task 2: `_lookup_kernel_path` helper + `_load_default_kernel` verification

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (add helper + extend loader + new module attr)
- Create: `tests/test_kernel_lookup.py`

- [ ] **Step 1: Write the failing tests (entire new test file)**

Create `tests/test_kernel_lookup.py`:

```python
"""Sub-project D: kernel lookup + load-verification tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _make_kernel_npz(
    path: Path,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
    include_metadata: bool = True,
) -> Path:
    """Write a tiny synthetic kernel npz at `path` matching the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "Resq_i": np.zeros((4, 4, 4), dtype=np.float64),
        "Nrays": np.int64(1000),
        "npoints1": np.int64(4),
        "npoints2": np.int64(4),
        "npoints3": np.int64(4),
        "qi1_range": np.float64(1e-3),
        "qi2_range": np.float64(1e-3),
        "qi3_range": np.float64(1e-3),
        "zeta_v_fwhm": np.float64(5.3e-4),
        "zeta_h_fwhm": np.float64(0.0),
        "NA_rms": np.float64(3.1e-4),
        "eps_rms": np.float64(6e-5),
        "theta": np.float64(0.165),
        "D": np.float64(5.6e-4),
        "d1": np.float64(0.274),
        "phys_aper": np.float64(2e-3),
        "beamstop": np.bool_(True),
        "bs_height": np.float64(25e-3),
        "aperture": np.bool_(True),
        "knife_edge": np.bool_(False),
        "dphi_range": np.float64(0.0),
    }
    if include_metadata:
        data["hkl"] = np.array(hkl, dtype=np.int64)
        data["keV"] = np.float64(keV)
    np.savez(path, **data)
    return path


class TestLookupKernelPath:
    def test_single_match_returns_path(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        result = _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path)
        assert result == p

    def test_multi_match_returns_newest_and_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        older = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260518_1142.npz")
        middle = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        newest = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260521_0930.npz")
        # Force mtimes for deterministic ordering on Windows.
        os.utime(older, (1000, 1000))
        os.utime(middle, (2000, 2000))
        os.utime(newest, (3000, 3000))

        result = _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path)
        assert result == newest

        err = capsys.readouterr().err
        assert "found 3 kernels" in err
        assert "(newest, will use)" in err
        assert "20260521_0930" in err

    def test_zero_match_raises_file_not_found(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        with pytest.raises(FileNotFoundError, match=r"no kernel found for hkl=\(2, 0, 0\)"):
            _lookup_kernel_path((2, 0, 0), 17.0, tmp_path)

    def test_glob_isolates_by_hkl(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        # Looking up a different reflection should not return the (-1,1,-1) file.
        with pytest.raises(FileNotFoundError):
            _lookup_kernel_path((2, 0, 0), 17.0, tmp_path)

    def test_keV_int_and_float_both_match_same_file(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        assert _lookup_kernel_path((-1, 1, -1), 17, tmp_path) == p
        assert _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path) == p


class TestLoadDefaultKernelVerification:
    def test_metadata_match_loads_cleanly(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz")
        fm._load_default_kernel(
            str(p),
            expected_hkl=(-1, 1, -1),
            expected_keV=17.0,
            compute_Hg=False,
        )
        assert fm._loaded_kernel_path == p

    def test_hkl_mismatch_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        with pytest.raises(ValueError, match=r"has hkl=\(-1, 1, -1\) but lookup requested hkl=\(2, 0, 0\)"):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(2, 0, 0),
                expected_keV=17.0,
                compute_Hg=False,
            )

    def test_keV_mismatch_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        with pytest.raises(ValueError, match=r"has keV=17.0 but lookup requested keV=15.0"):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(-1, 1, -1),
                expected_keV=15.0,
                compute_Hg=False,
            )

    def test_legacy_npz_missing_metadata_raises_keyerror(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_legacy.npz",
            include_metadata=False,
        )
        with pytest.raises(KeyError, match=r"lacks `hkl` metadata"):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(-1, 1, -1),
                expected_keV=17.0,
                compute_Hg=False,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_lookup.py -v
```
Expected: all tests FAIL — `_lookup_kernel_path` doesn't exist; `_load_default_kernel` doesn't accept `expected_hkl`/`expected_keV` kwargs; `_loaded_kernel_path` attribute doesn't exist.

- [ ] **Step 3: Add `_lookup_kernel_path` to `forward_model.py`**

In `src/dfxm_geo/direct_space/forward_model.py`, insert after the existing `_load_default_kernel` function (which is currently around lines 230-280; you may need to scroll to find the end). The exact insertion point is "after `_load_default_kernel`, before the module-import auto-load at line ~423":

```python
def _lookup_kernel_path(
    hkl: tuple[int, int, int],
    keV: float,
    pkl_fpath: str | Path,
) -> Path:
    """Find the newest kernel npz on disk matching the requested (hkl, keV).

    Globs ``<pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz``, sorts by
    mtime descending, returns the newest. Emits a stderr WARN listing all
    matches when more than one exists. Raises FileNotFoundError with a
    ``dfxm-bootstrap`` instruction on zero matches.

    Sub-project D: replaces the previous ``pkl_fn``-constant lookup.
    """
    import sys

    h, k, l = hkl
    pkl_fpath = Path(pkl_fpath)
    pattern = f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz"
    matches = sorted(
        pkl_fpath.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"no kernel found for hkl={hkl} at {keV} keV in {pkl_fpath}/.\n"
            f"Run: dfxm-bootstrap --config <yourconfig.toml>\n"
            f"(produces Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_<date>.npz, "
            f"~50 s wall-clock at default Nrays=1e8)"
        )
    if len(matches) > 1:
        lines = [f"warning: found {len(matches)} kernels matching hkl={hkl} keV={keV:g} in {pkl_fpath}:"]
        for i, m in enumerate(matches):
            tag = "  (newest, will use)" if i == 0 else ""
            lines.append(f"  {m.name}{tag}")
        print("\n".join(lines), file=sys.stderr)
    return matches[0]
```

- [ ] **Step 4: Add `_loaded_kernel_path` module attribute**

Still in `forward_model.py`, find the existing comment block at line 54-58 (the one above `pkl_fpath`). Add a new module attribute below `pkl_fpath` (and above the `pkl_fn` constant, which will be deleted in Task 6):

```python
# Sub-project D: set by `_load_default_kernel` on successful load; read by
# `io/hdf5.py` and `io/migrate.py` for provenance recording. `None` until a
# kernel is loaded.
_loaded_kernel_path: Path | None = None
```

- [ ] **Step 5: Extend `_load_default_kernel` with verification kwargs**

Find `_load_default_kernel` (around line 230). Extend its signature and body:

```python
def _load_default_kernel(
    pkl_path: str | Path | None = None,
    *,
    expected_hkl: tuple[int, int, int] | None = None,
    expected_keV: float | None = None,
    compute_Hg: bool = True,
) -> None:
    """Load kernel npz into module-level state.

    Args:
        pkl_path: Explicit path to a kernel npz. If None, defaults to
            ``<pkl_fpath>/<pkl_fn>`` (legacy behavior — removed in Task 6).
        expected_hkl: If given, verifies the npz's bundled ``hkl`` metadata
            matches. Raises ValueError on mismatch, KeyError if the npz
            lacks metadata (pre-sub-project-D kernel).
        expected_keV: As above for ``keV``.
        compute_Hg: If True, also compute Hg via Find_Hg (existing behavior).
            Tests typically pass False to skip the heavy compute.
    """
    global _loaded_kernel_path

    if pkl_path is None:
        # Legacy default — kept temporarily; Task 6 removes pkl_fn entirely.
        pkl_path = os.path.join(pkl_fpath, pkl_fn)

    # ... existing body up to and including the np.load(...) call ...

    # NEW: verify bundled metadata against the lookup request.
    if expected_hkl is not None:
        if "hkl" not in data.files:
            raise KeyError(
                f"kernel at {pkl_path} lacks `hkl` metadata — "
                f"pre-sub-project-D bootstrap.\n"
                f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
            )
        meta_hkl = tuple(int(x) for x in data["hkl"])
        if meta_hkl != tuple(expected_hkl):
            raise ValueError(
                f"kernel at {pkl_path} has hkl={meta_hkl} but lookup requested "
                f"hkl={tuple(expected_hkl)} — file may have been manually "
                f"renamed or copied wrong."
            )
    if expected_keV is not None:
        if "keV" not in data.files:
            raise KeyError(
                f"kernel at {pkl_path} lacks `keV` metadata — "
                f"pre-sub-project-D bootstrap.\n"
                f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
            )
        meta_keV = float(data["keV"])
        if meta_keV != expected_keV:
            raise ValueError(
                f"kernel at {pkl_path} has keV={meta_keV} but lookup requested "
                f"keV={expected_keV} — file may have been manually renamed or "
                f"copied wrong."
            )

    # ... existing body that populates globals (Resq_i, qi*_range, etc.) ...

    # NEW: track the actually-loaded path for provenance.
    _loaded_kernel_path = Path(pkl_path)
```

Note: the existing body is ~50 lines; don't rewrite it. Use the `Edit` tool to insert the verification block BEFORE the global-population code, and the `_loaded_kernel_path = Path(pkl_path)` assignment AT THE END. The existing `data = np.load(pkl_path)` call stays.

- [ ] **Step 6: Run tests to verify pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_lookup.py -v
```
Expected: all 9 tests PASS.

Also run mypy:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_kernel_lookup.py
git commit -m "feat(forward): add _lookup_kernel_path + load verification (sub-project D)"
```

---

## Task 3: `ReciprocalConfig` dataclass + `from_toml` plumbing

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (add dataclass + field-add to `SimulationConfig` + `IdentificationConfig` + extend both `from_toml` parsers)
- Test: `tests/test_pipeline.py` (append a new `TestReciprocalConfigParsing` class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestReciprocalConfigParsing:
    """Sub-project D: SimulationConfig + IdentificationConfig parse [reciprocal]."""

    def _write_minimal_sim_toml(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "config.toml"
        cfg.write_text(body)
        return cfg

    def test_simulation_config_parses_reciprocal_block(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = true\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        config = SimulationConfig.from_toml(cfg)
        assert config.reciprocal is not None
        assert config.reciprocal.hkl == (-1, 1, -1)
        assert config.reciprocal.keV == 17.0

    def test_simulation_config_missing_reciprocal_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = true\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n",
        )
        with pytest.raises(ValueError, match=r"missing \[reciprocal\] block"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_missing_hkl_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = true\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match=r"missing `hkl` in \[reciprocal\]"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_missing_keV_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = true\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\n",
        )
        with pytest.raises(ValueError, match=r"missing `keV` in \[reciprocal\]"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_invalid_hkl_propagates_validate_error(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = true\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [0, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match=r"hkl=\(0,0,0\) is not a valid reflection"):
            SimulationConfig.from_toml(cfg)
```

(`IdentificationConfig.from_toml` parsing tests are added separately in Task 4 Step 4, after the identification configs have been migrated to include `[reciprocal]`.)

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py::TestReciprocalConfigParsing -v
```
Expected: all 5 tests FAIL — `ReciprocalConfig` doesn't exist; `SimulationConfig.reciprocal` field doesn't exist; `from_toml` doesn't parse `[reciprocal]`.

- [ ] **Step 3: Add `ReciprocalConfig` dataclass to `pipeline.py`**

In `src/dfxm_geo/pipeline.py`, insert before `class SimulationConfig` (currently around line 99):

```python
@dataclass
class ReciprocalConfig:
    """Sub-project D: reflection identity for kernel lookup.

    The TOML ``[reciprocal]`` block carries both this (small, consumed by
    forward + identify) and bootstrap's MC params (large, consumed only by
    `dfxm-bootstrap`). This dataclass holds only the lookup-relevant keys.
    """

    hkl: tuple[int, int, int]
    keV: float

    @classmethod
    def from_dict(cls, data: dict | None) -> ReciprocalConfig:
        if data is None:
            raise ValueError(
                "missing [reciprocal] block — forward/identify require explicit "
                "hkl + keV; see configs/default.toml."
            )
        if "hkl" not in data:
            raise ValueError(
                "missing `hkl` in [reciprocal] — required for kernel lookup."
            )
        if "keV" not in data:
            raise ValueError(
                "missing `keV` in [reciprocal] — required for kernel lookup."
            )
        hkl = tuple(data["hkl"])
        keV = float(data["keV"])
        # Early validation per spec — catches typos / Bragg-unsatisfiable
        # before the kernel lookup. Propagates A's ValueErrors verbatim.
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection
        _validate_reflection(hkl, keV, 4.0495e-10)
        return cls(hkl=hkl, keV=keV)
```

- [ ] **Step 4: Field-add to `SimulationConfig` + extend `from_toml`**

Add the `reciprocal` field to `SimulationConfig` (around line 99):

```python
@dataclass
class SimulationConfig:
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    scan: ScanConfig = field(
        default_factory=lambda: ScanConfig(
            phi_range=0.0006 * 180 / np.pi,
            phi_steps=61,
            chi_range=0.002 * 180 / np.pi,
            chi_steps=61,
        )
    )
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Sub-project D: optional in Python construction (defaults to None for
    # back-compat with test fixtures). `from_toml` requires it; `run_simulation`
    # raises if None at runtime.
    reciprocal: ReciprocalConfig | None = None
```

Find `SimulationConfig.from_toml` (around line 113) and extend it. The existing body parses `data["crystal"]`, `data["scan"]`, etc. Add at the end (before the return statement):

```python
        reciprocal = ReciprocalConfig.from_dict(data.get("reciprocal"))

        return cls(
            crystal=crystal,
            scan=scan,
            io=io,
            postprocess=postprocess,
            reciprocal=reciprocal,
        )
```

- [ ] **Step 5: Field-add to `IdentificationConfig` + extend `load_identification_config`**

Find `IdentificationConfig` (around line 177) and `load_identification_config` (around line 230). Add the `reciprocal` field:

```python
@dataclass
class IdentificationConfig:
    mode: str
    crystal: IdentificationCrystalConfig
    scan: IdentificationScanConfig
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None
    # Sub-project D: optional in Python construction; load_identification_config
    # requires it.
    reciprocal: ReciprocalConfig | None = None
```

In `load_identification_config`, before the `return IdentificationConfig(...)`:

```python
    reciprocal = ReciprocalConfig.from_dict(data.get("reciprocal"))

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        io=io,
        multi=multi,
        zscan=zscan,
        reciprocal=reciprocal,
    )
```

- [ ] **Step 6: Run tests to verify pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py::TestReciprocalConfigParsing -v
```
Expected: all 5 tests PASS.

Also run the existing test_pipeline.py suite to confirm no regression:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -q
```
Expected: pre-existing tests still pass EXCEPT the ones that construct `SimulationConfig.from_toml` with TOMLs lacking `[reciprocal]` — those will fail and need fixture updates in Task 4 or Task 6. List the failures; they're expected at this stage.

mypy:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add ReciprocalConfig + plumb into SimulationConfig + IdentificationConfig (sub-project D)"
```

---

## Task 4: Migrate 3 identification_*.toml configs

**Files:**
- Modify: `configs/identification_single.toml`
- Modify: `configs/identification_multi.toml`
- Modify: `configs/identification_zscan.toml`
- Test: (no new tests; Task 3's `from_toml` parsing tests cover the schema)

- [ ] **Step 1: Add `[reciprocal]` block to each identify config**

For each of the three files, add at the top (after the leading comment block, before `mode = "..."`):

```toml
[reciprocal]
hkl = [-1, 1, -1]   # Al 111 reflection — preserves pre-D implicit default
keV = 17.0
```

- [ ] **Step 2: Verify each config still parses**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "from dfxm_geo.pipeline import load_identification_config; from pathlib import Path; print(load_identification_config(Path('configs/identification_single.toml')).reciprocal)"
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "from dfxm_geo.pipeline import load_identification_config; from pathlib import Path; print(load_identification_config(Path('configs/identification_multi.toml')).reciprocal)"
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "from dfxm_geo.pipeline import load_identification_config; from pathlib import Path; print(load_identification_config(Path('configs/identification_zscan.toml')).reciprocal)"
```
Expected each: `ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0)`.

- [ ] **Step 3: Add IdentificationConfig parsing tests**

Append to `TestReciprocalConfigParsing` in `tests/test_pipeline.py`:

```python
    def test_identification_config_parses_reciprocal_block(self) -> None:
        from dfxm_geo.pipeline import load_identification_config

        config = load_identification_config(Path("configs/identification_single.toml"))
        assert config.reciprocal is not None
        assert config.reciprocal.hkl == (-1, 1, -1)
        assert config.reciprocal.keV == 17.0

    def test_identification_config_missing_reciprocal_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import load_identification_config

        cfg = tmp_path / "identify.toml"
        cfg.write_text(
            'mode = "single"\n'
            "[crystal]\nslip_plane_normal = [1, 1, 1]\n"
            "angle_start_deg = 0.0\nangle_stop_deg = 10.0\nangle_step_deg = 1.0\n"
            "sweep_all_slip_planes = false\nexclude_invisibility = false\n"
            "invisibility_threshold_deg = 10.0\n"
            "[scan]\nphi_rad = 1.5e-4\npoisson_noise = false\n"
            "rng_seed = 0\nintensity_scale = 7.0\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = false\n"
        )
        with pytest.raises(ValueError, match=r"missing \[reciprocal\] block"):
            load_identification_config(cfg)
```

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py::TestReciprocalConfigParsing -v
```
Expected: all 7 tests (5 from Task 3 + 2 new) pass.

- [ ] **Step 4: Commit**

```bash
git add configs/identification_single.toml configs/identification_multi.toml configs/identification_zscan.toml tests/test_pipeline.py
git commit -m "config(identify): add [reciprocal] block to 3 identification configs (sub-project D)"
```

---

## Task 5: `_lookup_and_load_kernel` + pipeline rewire + integration tests

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (replace `_ensure_kernel_loaded` with `_lookup_and_load_kernel`; rewire `run_simulation` + `run_identification`; update effective-config print)
- Create: `tests/test_pipeline_multi_reflection.py`

- [ ] **Step 1: Write the failing integration tests (new file)**

Create `tests/test_pipeline_multi_reflection.py`:

```python
"""Sub-project D integration tests: lookup-driven forward + identify."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _make_kernel_npz(
    path: Path,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
    include_metadata: bool = True,
) -> Path:
    """Inline copy of the helper from test_kernel_lookup.py to keep this file self-contained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "Resq_i": np.zeros((4, 4, 4), dtype=np.float64),
        "Nrays": np.int64(1000),
        "npoints1": np.int64(4),
        "npoints2": np.int64(4),
        "npoints3": np.int64(4),
        "qi1_range": np.float64(1e-3),
        "qi2_range": np.float64(1e-3),
        "qi3_range": np.float64(1e-3),
        "zeta_v_fwhm": np.float64(5.3e-4),
        "zeta_h_fwhm": np.float64(0.0),
        "NA_rms": np.float64(3.1e-4),
        "eps_rms": np.float64(6e-5),
        "theta": np.float64(0.165),
        "D": np.float64(5.6e-4),
        "d1": np.float64(0.274),
        "phys_aper": np.float64(2e-3),
        "beamstop": np.bool_(True),
        "bs_height": np.float64(25e-3),
        "aperture": np.bool_(True),
        "knife_edge": np.bool_(False),
        "dphi_range": np.float64(0.0),
    }
    if include_metadata:
        data["hkl"] = np.array(hkl, dtype=np.int64)
        data["keV"] = np.float64(keV)
    np.savez(path, **data)
    return path


class TestForwardMultiReflection:
    def test_happy_path_with_explicit_hkl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_simulation looks up the kernel from [reciprocal].hkl/keV, loads it,
        runs the (monkey-patched) forward, records actual loaded path in provenance."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Stage a kernel matching (2, 0, 0) @ 17 keV
        kernel_path = _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(2, 0, 0), keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        # Monkey-patch the heavy pieces.
        calls: list[str] = []
        monkeypatch.setattr(fm, "Find_Hg", lambda *a, **k: (np.zeros((4, 4, 4)), np.zeros(3)))
        monkeypatch.setattr(p, "save_images_parallel", lambda *a, **k: calls.append("save") or [])
        # Skip postprocess for this test.
        # (run_simulation calls it only if config.postprocess.enabled.)

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = false\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        p.run_simulation(config, tmp_path / "out")

        # Verify the actually-loaded path is the staged kernel.
        assert fm._loaded_kernel_path == kernel_path

    def test_lookup_miss_errors_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files_empty") + os.sep)
        (tmp_path / "pkl_files_empty").mkdir()

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = false\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(FileNotFoundError, match=r"no kernel found for hkl=\(2, 0, 0\)"):
            p.run_simulation(config, tmp_path / "out")

    def test_metadata_mismatch_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File named for (2,0,0) but contains bundled hkl=(-1,1,-1) → ValueError."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Filename says 200 but metadata says -1,1,-1.
        _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(-1, 1, -1), keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[crystal]\ndis = 4\nndis = 151\nsample_remount = \"S1\"\n"
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            "[io]\nfn_prefix = \"/x\"\nftype = \".npy\"\n"
            "dislocs_dirname = \"d\"\nperfect_dirname = \"p\"\ninclude_perfect_crystal = false\n"
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            "figures_dirname = \"f\"\ndata_dirname = \"a\"\n"
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(ValueError, match=r"has hkl=\(-1, 1, -1\) but lookup requested hkl=\(2, 0, 0\)"):
            p.run_simulation(config, tmp_path / "out")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_multi_reflection.py -v
```
Expected: all 3 tests FAIL — `_lookup_and_load_kernel` doesn't exist yet.

- [ ] **Step 3: Add `_lookup_and_load_kernel` to `pipeline.py`**

In `src/dfxm_geo/pipeline.py`, find `_ensure_kernel_loaded()` (currently around line 253). Replace it entirely with:

```python
def _lookup_and_load_kernel(
    hkl: tuple[int, int, int],
    keV: float,
) -> None:
    """Pre-flight: look up the kernel npz matching (hkl, keV) and load it.

    Replaces the previous `_ensure_kernel_loaded()`. Composes:
    1. `fm._lookup_kernel_path(hkl, keV, fm.pkl_fpath)` — glob + newest pick.
    2. `fm._load_default_kernel(path, expected_hkl=hkl, expected_keV=keV)` —
       load + bundled-metadata verification.

    Idempotent for the same (hkl, keV): if `fm._loaded_kernel_path` already
    matches what we'd look up, skip the reload. (Helpful for test loops and
    interactive REPL.)

    Raises FileNotFoundError on lookup miss, ValueError on metadata mismatch,
    KeyError on pre-sub-project-D legacy npz lacking metadata.
    """
    target = fm._lookup_kernel_path(hkl, keV, fm.pkl_fpath)
    if fm._loaded_kernel_path == target:
        return
    fm._load_default_kernel(
        str(target),
        expected_hkl=hkl,
        expected_keV=keV,
    )
```

- [ ] **Step 4: Rewire `run_simulation` to call the new function**

Find `run_simulation` (around line 276). Replace the existing `_ensure_kernel_loaded()` call (line 305) with:

```python
    if config.reciprocal is None:
        raise ValueError(
            "SimulationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_simulation."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)
```

Also update the effective-config print (lines 295-303). Change `kernel={fm.pkl_fpath}{fm.pkl_fn}` to `kernel={fm._loaded_kernel_path}`:

```python
    print(
        f"[dfxm-forward] effective config:\n"
        f"  Nsub={fm.Nsub}  Npixels={fm.Npixels}  NN1={fm.NN1}  NN2={fm.NN2}\n"
        f"  kernel={fm._loaded_kernel_path}\n"
        f"  Fg cache={_expected_fg}\n"
        f"  dis={config.crystal.dis}  ndis={config.crystal.ndis}  "
        f"remount={config.crystal.sample_remount}",
        flush=True,
    )
```

Note: this print needs to come AFTER `_lookup_and_load_kernel` so `fm._loaded_kernel_path` is set. Move the print to AFTER the kernel-load call.

- [ ] **Step 5: Rewire `run_identification` similarly**

Find `run_identification` (somewhere after `run_simulation`; grep for it). Add at the top of its body:

```python
    if config.reciprocal is None:
        raise ValueError(
            "IdentificationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_identification."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)
```

- [ ] **Step 6: Run tests**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_multi_reflection.py -v
```
Expected: all 3 PASS.

Also run mypy:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_multi_reflection.py
git commit -m "feat(pipeline): _lookup_and_load_kernel; rewire run_simulation + run_identification (sub-project D)"
```

---

## Task 6: Delete `pkl_fn` + module-import auto-load; update io callers + test cleanups

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (delete `pkl_fn` constant + module-import auto-load block)
- Modify: `src/dfxm_geo/io/hdf5.py:414` (use `fm._loaded_kernel_path`)
- Modify: `src/dfxm_geo/io/migrate.py:154` (use `fm._loaded_kernel_path`)
- Modify: `tests/test_kernel_cli.py` — remove `TestPklFnRegression` class (obsolete after pkl_fn deletion)
- Modify: `tests/test_pipeline.py` — rename all `_ensure_kernel_loaded` → `_lookup_and_load_kernel`; update kernel-load tests to use the new signature
- Modify: `tests/test_migrate_output.py:36, 72` — replace `monkeypatch.setattr(_fm, "pkl_fn", ...)` with `monkeypatch.setattr(_fm, "_loaded_kernel_path", Path(...))`
- Modify: `tests/test_hdf5_provenance.py:51` — confirm HDF5 dataset name `pkl_fn` still works (no change expected; dataset name stays)
- Modify: `tests/test_cluster_templates.py:200` — update docstring reference (cosmetic)

- [ ] **Step 1: Delete `pkl_fn` constant + module-import auto-load in `forward_model.py`**

In `src/dfxm_geo/direct_space/forward_model.py`:

(a) Delete lines 59-64 (the `pkl_fn` constant + its comment):

```python
# Constant name preserved (`pkl_fn`) so import-time monkeypatches in tests
# and the dfxm-bootstrap CLI don't break. Value follows the
# Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz pattern introduced in
# sub-project A; sync to the actual filename produced by
# `dfxm-bootstrap` on each host (laptop ran 2026-05-20 20:14).
pkl_fn = "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz"  # Update after `dfxm-bootstrap` regen (per-reflection pattern)
```

(b) Delete lines 423-428 (the module-import auto-load block):

```python
if os.path.exists(os.path.join(pkl_fpath, pkl_fn)):
    _load_default_kernel()
else:
    print(
        f"NOTE: default kernel npz not found at {os.path.join(pkl_fpath, pkl_fn)!r}; "
        f"call _load_default_kernel(pkl_path) before forward(), or run `dfxm-bootstrap`."
    )
```

(c) Update `_load_default_kernel`'s default-pkl_path handling. The current Task 2 step kept `pkl_path = os.path.join(pkl_fpath, pkl_fn)` as a legacy fallback. Now that `pkl_fn` is gone, replace it with a hard error:

```python
    if pkl_path is None:
        raise ValueError(
            "_load_default_kernel requires an explicit `pkl_path` after sub-project D. "
            "Use `_lookup_kernel_path(hkl, keV, pkl_fpath)` to find the matching kernel."
        )
```

(d) Update `forward_model.py:202` where `pkl_fn` was referenced inside `print_kernel_meta` or similar (grep for it). Replace the reference with `_loaded_kernel_path.name if _loaded_kernel_path else "<none>"`.

(e) Update `forward_model.py:250` — inside `_load_default_kernel`, the `Resq_i_filename` it uses for printing. Replace `pkl_fn` with `Path(pkl_path).name`.

- [ ] **Step 2: Update `io/hdf5.py:414`**

In `src/dfxm_geo/io/hdf5.py`, find line 414:

```python
        kernel_npz = Path(_fm.pkl_fpath) / _fm.pkl_fn
```

Replace with:

```python
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing HDF5 provenance."
            )
```

- [ ] **Step 3: Update `io/migrate.py:154`**

In `src/dfxm_geo/io/migrate.py`, find line 154:

```python
    kernel_npz = Path(_fm.pkl_fpath) / _fm.pkl_fn
```

Replace with:

```python
    kernel_npz = _fm._loaded_kernel_path
    if kernel_npz is None:
        raise RuntimeError(
            "no kernel loaded — migration requires a loaded kernel for provenance recording."
        )
```

- [ ] **Step 4: Remove `TestPklFnRegression` from `tests/test_kernel_cli.py`**

Find the `TestPklFnRegression` class (around line 686). Delete the entire class — `pkl_fn` no longer exists, so the regression guard is obsolete.

- [ ] **Step 5: Update `tests/test_pipeline.py` — rename + update fixtures**

In `tests/test_pipeline.py`:

(a) Line 24: change `from dfxm_geo.pipeline import _ensure_kernel_loaded` → `from dfxm_geo.pipeline import _lookup_and_load_kernel`. (If `_ensure_kernel_loaded` is referenced from a class import list, update the import name.)

(b) All call-sites `_ensure_kernel_loaded()` → `_lookup_and_load_kernel((-1, 1, -1), 17.0)`. Grep for them; ~5 sites around lines 189-242, 281, 327, 366.

(c) Lines 189-242: tests that `monkeypatch.setattr(fm, "pkl_fn", ...)` — these test the legacy fallback. After Task 6 they should test the lookup path. Update each to:
- Stage a synthetic kernel via `_make_kernel_npz` (copy the helper inline or import from test_kernel_lookup.py)
- `monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + os.sep)`
- Call `_lookup_and_load_kernel((-1, 1, -1), 17.0)`
- Assert behavior

(d) Existing `test_dfxm_forward_with_sample_remount_S2_runs` (the `@pytest.mark.slow` test): add a `[reciprocal]` block to its inline TOML fixture if it constructs one.

- [ ] **Step 6: Update `tests/test_migrate_output.py:36, 72`**

Line 36 — change `monkeypatch.setattr(_fm, "pkl_fn", "fake_kernel.npz")` to:

```python
    monkeypatch.setattr(_fm, "_loaded_kernel_path", Path("fake_kernel.npz"))
```

Line 72 — the HDF5 dataset assertion `f["/dfxm_geo/kernel/pkl_fn"][()].decode() == "fake_kernel.npz"` stays as-is (the HDF5 dataset name doesn't change, only the Python attribute does).

- [ ] **Step 7: Update `tests/test_cluster_templates.py:200`**

Line 200 — the docstring reference to `dfxm_geo.direct_space.forward_model.pkl_fn`. Update to reference `_loaded_kernel_path` or just remove that part of the docstring; this is cosmetic.

- [ ] **Step 8: Run the full kernel + pipeline test suite**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_lookup.py tests/test_kernel_cli.py tests/test_pipeline.py tests/test_pipeline_multi_reflection.py tests/test_migrate_output.py tests/test_hdf5_provenance.py -v
```
Expected: all pass.

Also mypy:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors.

If anything fails: investigate inline. Common issues:
- A test still references `fm.pkl_fn` → update to use `fm._loaded_kernel_path` or remove
- A test calls `_load_default_kernel()` with no args → update to pass a `pkl_path`

- [ ] **Step 9: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py src/dfxm_geo/io/hdf5.py src/dfxm_geo/io/migrate.py tests/test_kernel_cli.py tests/test_pipeline.py tests/test_migrate_output.py tests/test_cluster_templates.py
git commit -m "refactor(forward): delete pkl_fn constant + module-import auto-load (sub-project D)"
```

---

## Task 7: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pytest suite**

```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q
```

Expected:
- ~30 new tests added by sub-project D (`TestLookupKernelPath` 5 + `TestLoadDefaultKernelVerification` 4 + `TestReciprocalConfigParsing` 5+2-for-identify = 7 + `TestForwardMultiReflection` 3 + 1 in `TestGenerateKernelOutputPath` = ~20-30 new).
- 2 pre-existing handled-failures: `test_hdf5_writer_bit_equivalent_to_legacy_npy_golden` (xfailed) + `test_dfxm_forward_with_sample_remount_S2_runs` (deselected via `@pytest.mark.slow`).
- 1 pre-existing xfail: `TestForwardOutputBitEquivalence::test_forward_output_matches_pickle_era_snapshot`.
- 9 deselected: bench.
- 1 more deselected: the new `slow` mark.
- Everything else PASS.

If any other test FAILS, that's a sub-project D regression — STOP and report.

- [ ] **Step 2: Bootstrap regen to produce a new-metadata-bundled kernel**

The current `Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz` (from sub-project A's bootstrap) lacks `hkl`/`keV` metadata. Re-run bootstrap to produce a kernel with the new metadata. This is necessary to test the end-to-end CLI path against a real kernel:

```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m dfxm_geo.reciprocal_space.kernel --config configs/default.toml --force
```

Expected: ~50 s wall-clock. Stdout echoes `reflection: hkl=(-1, 1, -1), keV=17 -> theta = 8.9732 deg`. Produces a new npz at `reciprocal_space/pkl_files/Resq_i_h-1_k1_l-1_17keV_<new-date>.npz` with `hkl` + `keV` bundled.

Verify the new metadata:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "import numpy as np, glob; p = sorted(glob.glob('reciprocal_space/pkl_files/Resq_i_h-1_k1_l-1_17keV_*.npz'))[-1]; d = np.load(p); print('files:', sorted(d.files)); print('hkl:', d['hkl']); print('keV:', d['keV'])"
```
Expected output includes `hkl: [-1  1 -1]` and `keV: 17.0` in addition to the existing 20 scalars.

- [ ] **Step 3: CLI smoke test end-to-end against the new kernel**

Use a tiny config to exercise `dfxm-forward` with the new lookup machinery and the freshly-bootstrapped kernel:

```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "
from pathlib import Path
import tempfile
from dfxm_geo.pipeline import SimulationConfig, run_simulation
import tomllib

# Use the default.toml but override scan size for speed.
with open('configs/default.toml', 'rb') as f:
    data = tomllib.load(f)
data['scan']['phi_steps'] = 2
data['scan']['chi_steps'] = 2

import tempfile
with tempfile.TemporaryDirectory() as td:
    cfg_path = Path(td) / 'tmp.toml'
    import tomli_w
    cfg_path.write_text(open('configs/default.toml').read().replace('phi_steps = 61', 'phi_steps = 2').replace('chi_steps = 61', 'chi_steps = 2'))
    config = SimulationConfig.from_toml(cfg_path)
    run_simulation(config, Path(td) / 'out')
print('smoke OK')
"
```

Expected: prints `smoke OK` after ~10-30s. Confirms the entire stack (TOML → ReciprocalConfig → _lookup_and_load_kernel → run_simulation → HDF5 write) works end-to-end against the freshly-bootstrapped kernel.

If the smoke fails: inspect the error, fix inline, commit with `fix(...): post-Task-7 smoke correction for [...]`.

- [ ] **Step 4: Final commit (if any fixups in Step 3)**

If Step 3 required fixes:
```bash
git add -p
git commit -m "fix: post-Task-7 verification fixes"
```

If no fixes needed, skip.

- [ ] **Step 5: Confirm with Sina before pushing (per CLAUDE.md)**

Do NOT push without explicit approval. Send a Discord summary with:
- All 7 task commits SHAs
- Final test count + mypy status
- Bootstrap regen confirmation (new file basename)
- Smoke result

Wait for "push" / "merge locally" / "keep" decision (the `superpowers:finishing-a-development-branch` skill flow).

---

## Operational follow-ups (after PR merge)

- **Cluster bootstrap regen**: run `dfxm-bootstrap --config configs/default.toml --force` on the DTU cluster to produce a cluster-local kernel with the new `hkl`/`keV` metadata bundled. Without this, any forward/identify run on the cluster will hit the `KeyError("kernel at <path> lacks 'hkl' metadata — pre-sub-project-D bootstrap")` from sub-project D's verification. The error message tells the user exactly what to do.
- **Update CLAUDE.md** with the "PR #6 / sub-project A operational follow-up" entry to note that the per-host bootstrap step is now self-documenting (the error message instructs the user).
- **Pipeline-features arc next**: sub-project B+C (scan modes + crystal layout modes) per the v1.2.0 roadmap.
