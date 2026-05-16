# Kernel Pickle → npz Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `pickle`-based reciprocal-space resolution kernel format with `.npz`. Kill both `pickle.load` in the loader and `eval(_vars.txt)` in the sidecar parser. Ships as v1.0.3 patch.

**Architecture:** Single `.npz` per kernel with the resolution array AND all generation params bundled as named scalar entries (no sidecar). Loader detects file extension; `.pkl` paths raise a loud `RuntimeError` directing the user to `dfxm-bootstrap`. Migration shape is B2 — pickle support removed entirely; users (i.e., Sina) regenerate fresh via `dfxm-bootstrap` (~50 s one-time).

**Tech Stack:** numpy (`np.savez` / `np.load`), pytest, pytest-benchmark. No new runtime deps.

**Branch:** `worktree-kernel-npz`, branched off `worktree-phase8-closeout` HEAD (`8280c63`). Worktree at `.claude/worktrees/kernel-npz`. Rebases cleanly onto `main` once PR #5 (Phase 8 close-out) merges as v1.0.2.

**Source design:** `~/.claude/projects/.../memory/followups_kernel_pickle_alternatives.md` (the /grill-me resolution).

**Plan revisions (2026-05-16, post-Opus-subagent-review):** Fixed 6 blocking issues + 5 should-fix nits caught by an independent cold-context review. Substantive changes:
- File-structure header line ranges corrected (auto-load block was cited at 388-394; actual is 412-418, off by 24 because the plan drafted against pre-PR-#5 line numbers).
- Task 3 Step 3/4 reordered: signature kwarg added BEFORE the body change (the body references `kernel_meta`; the signature must define it first).
- Task 4 expanded from 2 cascade-update steps to 5 explicit steps: loader body, module-level constants, module docstring (lines 7-10), `forward()` init-state RuntimeError (lines 290-296), and the import-time auto-load block (lines 409-418) — the old plan missed the latter three.
- Task 4 Step 3 loader rewrite uses `Resq_i = np.array(arch["Resq_i"])` to materialize the array before the `with np.load(...) as arch:` block closes (subagent flagged: lazy view may break in some numpy versions after close).
- Task 5 Step 3 quotes exact `.pkl` strings + correct line numbers for `tests/test_pipeline.py:622` (was 608) and adds `tests/test_kernel_cli.py` / `tests/test_forward_model_paths.py` / `tests/test_forward_model_smoke.py` to the audit list. Also widens `test_cluster_templates.py` regex to catch hypothetical `.npz` hardcoding.
- Task 1 Step 1 snapshot generator gained an `fm.Hg is None` check (previously could silently call `forward()` with no Hg).
- Task 3 Step 7 adds round-trip assertions for `theta` and `D` (write-only audit fields) alongside the load-back fields.
- Task 8 Step 1 dropped per-test latency assertion in favour of the project's saved-baseline pattern (matches existing benches; unused `result` variable also removed).
- Task 9 docstring clarifies the audit is scoped to src/ only (via `mod.__file__`), so test-side pickle imports in Task 6 are safe.
- Task 0 PowerShell-friendly: dropped `cat | head` and `/tmp/` Unix idioms.

---

## File Structure

**Modify:**
- `src/dfxm_geo/reciprocal_space/resolution.py:152-178,319-335` — add `kernel_meta` kwarg to `reciprocal_res_func` signature (lines 152-178); switch `pickle.dump` → `np.savez` with bundled params at the save block (lines 319-335); drop `import pickle`
- `src/dfxm_geo/reciprocal_space/kernel.py:84-93,120-144` — drop `vars_path` derivation (lines 84-93); drop `_vars.txt` sidecar write (lines 120-144); pass `kernel_meta` dict to `reciprocal_res_func`; flip default-return filename `.pkl` → `.npz` (line 146)
- `src/dfxm_geo/direct_space/forward_model.py:7-10,15-16,55-58,224-267,290-296,409-418` — update module docstring (lines 7-10); drop `import pickle` (line 16); drop `vars_fn` constant (line 58); change `pkl_fn` value to `.npz` filename (line 57); rewrite `_load_default_kernel` body — `np.load` instead of `pickle.load`+`eval(_vars.txt)`, raise on `.pkl` (lines 224-267); update `forward()`'s init-state RuntimeError text (lines 290-296); update import-time auto-load block (lines 409-418)
- `docs/architecture.md`, `docs/reproducibility.md` — update format references

**Create:**
- `tests/test_kernel_format.py` — Layers 1+2+3 of the test plan
- `tests/data/golden/forward_snapshot_pickle_era.npy` — Layer 2 bit-equivalence reference (captured from current pickle-loaded `forward()`)
- `tests/data/golden/synthetic_kernel.npz` — Layer 3 cross-platform reference (tiny, ~5 KB)
- `tests/_gen_forward_snapshot_pickle_era.py` — one-shot generator for the above (committed for reproducibility)
- `tests/_gen_synthetic_kernel.py` — one-shot generator for the synthetic golden

**Not committed (artifacts only):**
- The regenerated `Resq_i_<new-timestamp>.npz` (gitignored under `*.npy` — wait, that pattern doesn't cover `.npz`. Need to add `*.npz` to .gitignore as part of Task 0).

---

## Task 0: Update .gitignore for `.npz`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current .gitignore to confirm scope**

Read `.gitignore` directly via the Read tool (this is a Windows / PowerShell environment — avoid `cat | head` Unix idioms).

Expected: see `*.pkl` and `*.npy` under "Project — generated outputs"; `*.npz` not present.

- [ ] **Step 2: Add `*.npz` to .gitignore alongside `*.pkl` / `*.npy`**

Edit `.gitignore`, find the "Project — generated outputs" block:
```
# Project — generated outputs
*.npy
*.pkl
*.edf
*.mp4
output/
results/
rockingcurve*/
mixed_ims*/
final_figures/
```

Add `*.npz` on a new line after `*.pkl`. Also confirm the existing `!tests/data/golden/*.npy` exception still applies; add `!tests/data/golden/*.npz` directly under it so our golden synthetic kernel stays tracked:

```
# Project — generated outputs
*.npy
*.pkl
*.npz
*.edf
*.mp4
output/
results/
rockingcurve*/
mixed_ims*/
final_figures/

# Project — keep small reference data
!tests/data/golden/*.npy
!tests/data/golden/*.npz
!docs/figures/*.png
```

- [ ] **Step 3: Verify the gitignore behaves**

Run from worktree root (PowerShell-friendly — the project is on Windows):
```powershell
New-Item -ItemType File -Path "reciprocal_space\pkl_files\foo.npz" -Force | Out-Null
git check-ignore "reciprocal_space\pkl_files\foo.npz"
Remove-Item "reciprocal_space\pkl_files\foo.npz"
```

Expected: `git check-ignore` prints the path (meaning it IS ignored). If it errors with no output, the ignore didn't take.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore *.npz alongside *.pkl / *.npy; preserve tests/data/golden exception"
```

---

## Task 1: Capture pickle-era `forward()` snapshot for the Layer 2 golden

**Files:**
- Create: `tests/_gen_forward_snapshot_pickle_era.py`
- Create: `tests/data/golden/forward_snapshot_pickle_era.npy`

**Why:** Once Task 3 lands, the pickle reader is gone — we lose the ability to verify "loading from npz produces the same `forward()` output as loading from the original pickle." So this snapshot must be captured first, against the CURRENT pickle-based loader.

- [ ] **Step 1: Write the snapshot generator**

Create `tests/_gen_forward_snapshot_pickle_era.py`:

```python
"""One-shot generator for the pickle-era forward() snapshot golden.

Run once before the kernel pickle → npz migration to capture a deterministic
forward() output using the CURRENT pickle-based loader. The resulting .npy
becomes the bit-equivalence reference for test_kernel_format.py:
test_forward_output_matches_pickle_era_snapshot.

Run from the repo root:

    python -m tests._gen_forward_snapshot_pickle_era

Output: tests/data/golden/forward_snapshot_pickle_era.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dfxm_geo.direct_space.forward_model as fm


def main() -> None:
    if fm.Resq_i is None:
        raise RuntimeError(
            "Resq_i not loaded; this generator requires the existing pickle "
            f"to be present at {fm.pkl_fpath}{fm.pkl_fn}."
        )
    if fm.Hg is None:
        raise RuntimeError(
            "Hg not populated; the auto-load block at forward_model.py:412-418 "
            "must have run with compute_Hg=True. Re-import the module fresh "
            "and re-run."
        )

    # Fixed (phi, chi) chosen to exercise the typical Bragg condition path.
    # Values match the snap_old.npy snapshot taken during Phase 8 Round 25
    # to keep this reference comparable to pre-existing dev snapshots.
    phi = 0.0
    chi = 0.0
    out = fm.forward(fm.Hg, phi=phi, chi=chi)

    if isinstance(out, tuple):
        out = out[0]

    dst = Path("tests/data/golden/forward_snapshot_pickle_era.npy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    print(f"wrote {dst} shape={out.shape} dtype={out.dtype} sum={out.sum():.6e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run from worktree root:
```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m tests._gen_forward_snapshot_pickle_era
```

Expected: prints something like `wrote tests/data/golden/forward_snapshot_pickle_era.npy shape=(510, 170) dtype=float64 sum=3.397580e+06` (sum value should match snap_old.npy's `3397579.6706151115` from Round 25 within FP noise).

- [ ] **Step 3: Sanity-check the snapshot**

Run:
```bash
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "import numpy as np; a = np.load(r'tests/data/golden/forward_snapshot_pickle_era.npy'); print('shape:', a.shape, 'dtype:', a.dtype, 'min:', a.min(), 'max:', a.max(), 'sum:', a.sum())"
```

Expected: non-zero `max`, `sum` matches the generator's printed sum, dtype is `float64`.

- [ ] **Step 4: Commit**

```bash
git add tests/_gen_forward_snapshot_pickle_era.py tests/data/golden/forward_snapshot_pickle_era.npy
git commit -m "test(kernel): capture pickle-era forward() snapshot as bit-equivalence golden

Generated via tests/_gen_forward_snapshot_pickle_era.py against the
current Resq_i_*.pkl loader. Becomes the reference for test_kernel_format
once the pickle reader is removed in the upcoming npz migration."
```

---

## Task 2: Generate the synthetic cross-platform golden (Layer 3)

**Files:**
- Create: `tests/_gen_synthetic_kernel.py`
- Create: `tests/data/golden/synthetic_kernel.npz`

- [ ] **Step 1: Write the synthetic-kernel generator**

Create `tests/_gen_synthetic_kernel.py`:

```python
"""One-shot generator for the synthetic cross-platform kernel golden.

A 4x4x4 deterministic synthetic Resq_i plus bundled scalar params, used as
the cross-platform reference in tests/test_kernel_format.py.

Run from the repo root:

    python -m tests._gen_synthetic_kernel

Output: tests/data/golden/synthetic_kernel.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(20260516)
    Resq_i = rng.uniform(0.0, 1.0, size=(4, 4, 4)).astype(np.float64)

    dst = Path("tests/data/golden/synthetic_kernel.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        Resq_i=Resq_i,
        qi1_range=np.float64(5e-4),
        qi2_range=np.float64(0.75e-2),
        qi3_range=np.float64(0.75e-2),
        npoints1=np.int64(4),
        npoints2=np.int64(4),
        npoints3=np.int64(4),
        Nrays=np.int64(1_000),
    )
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run from worktree root:
```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m tests._gen_synthetic_kernel
```

Expected: `wrote tests/data/golden/synthetic_kernel.npz (~5000 bytes)`. Size should be a few KB, not megabytes.

- [ ] **Step 3: Verify the file is loadable and has the expected keys**

Run:
```bash
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "import numpy as np; d = np.load(r'tests/data/golden/synthetic_kernel.npz'); print('keys:', sorted(d.files)); print('Resq_i shape:', d['Resq_i'].shape); print('qi1_range:', float(d['qi1_range']))"
```

Expected:
```
keys: ['Nrays', 'Resq_i', 'npoints1', 'npoints2', 'npoints3', 'qi1_range', 'qi2_range', 'qi3_range']
Resq_i shape: (4, 4, 4)
qi1_range: 0.0005
```

- [ ] **Step 4: Commit**

```bash
git add tests/_gen_synthetic_kernel.py tests/data/golden/synthetic_kernel.npz
git commit -m "test(kernel): add 4x4x4 synthetic kernel golden for cross-platform npz check

Deterministic RNG seed (20260516); 8 bundled entries matching the
canonical kernel layout. Used in test_kernel_format.test_load_synthetic_kernel
to verify npz loading works regardless of the writing platform."
```

---

## Task 3: Switch the writer (`reciprocal_res_func`) from `pickle.dump` to `np.savez`

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py:319-335`
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py:95-146`
- Test: `tests/test_kernel_format.py` (new file)

- [ ] **Step 1: Write the failing round-trip test**

Create `tests/test_kernel_format.py`:

```python
"""Tests for the .npz kernel format that replaces the legacy pickle format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"


class TestNpzRoundTrip:
    """Layer 1: round-trip and param-bundling correctness."""

    def test_savez_round_trip_preserves_resq_i(self, tmp_path: Path) -> None:
        """np.savez → np.load preserves the resolution array bit-for-bit."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0, 1, size=(8, 6, 4)).astype(np.float64)

        dst = tmp_path / "round_trip.npz"
        np.savez(dst, Resq_i=arr, qi1_range=np.float64(5e-4))

        loaded = np.load(dst)
        assert np.array_equal(loaded["Resq_i"], arr)

    def test_bundled_scalars_extract_to_correct_types(self, tmp_path: Path) -> None:
        """Scalar params bundled into npz extract back to the right Python types."""
        dst = tmp_path / "bundled.npz"
        np.savez(
            dst,
            Resq_i=np.zeros((2, 2, 2)),
            qi1_range=np.float64(5e-4),
            qi2_range=np.float64(7.5e-3),
            qi3_range=np.float64(7.5e-3),
            npoints1=np.int64(400),
            npoints2=np.int64(200),
            npoints3=np.int64(200),
            Nrays=np.int64(100_000_000),
        )

        loaded = np.load(dst)
        # 0-d numpy arrays — extract with .item() or float() / int()
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        assert int(loaded["npoints1"]) == 400
        assert int(loaded["Nrays"]) == 100_000_000
```

- [ ] **Step 2: Run the test, verify it passes**

Run:
```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestNpzRoundTrip -v
```

Expected: 2 passed. (Pure numpy behaviour; no project code touched yet.)

- [ ] **Step 3: Add `kernel_meta` kwarg to `reciprocal_res_func` signature**

This step MUST come before the body change (Step 4). The body references `kernel_meta`; if the signature doesn't have it yet, the function won't parse.

In `src/dfxm_geo/reciprocal_space/resolution.py`, locate the `def reciprocal_res_func(...)` signature at lines 152-178. Append `kernel_meta: dict | None = None` after `output_path: Path | None = None` (line 177):

```python
def reciprocal_res_func(
    Nrays: int,
    npoints1: int,
    npoints2: int,
    npoints3: int,
    qi1_range: float,
    qi2_range: float,
    qi3_range: float,
    plot_figs: bool,
    save_resqi: bool,
    zeta_v_fwhm: float,
    zeta_h_fwhm: float,
    NA_rms: float,
    eps_rms: float,
    theta: float,
    phys_aper: float,
    date: str,
    mem_save: bool = True,
    rng: np.random.Generator | None = None,
    return_qs: bool = False,
    dphi_range: float = 0.0,
    beamstop: bool = False,
    bs_height: float | None = None,
    aperture: bool = False,
    knife_edge: bool = False,
    output_path: Path | None = None,
    kernel_meta: dict | None = None,
) -> tuple[np.ndarray, ...] | None:
```

Also remove the now-unused `import pickle` at the top of `resolution.py`.

- [ ] **Step 4: Modify the writer body at `resolution.py:319-335` from `pickle.dump` to `np.savez`**

Current block (lines 319-335):

```python
    if save_resqi == 1:
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as output:
                pickle.dump(normResq_i, output)
            print(f"Resq_i saved to {output_path}")
        else:
            # Legacy default: write to pkl_files/Resq_i_<date>.pkl in CWD.
            check_folder("", "pkl_files")
            with open(f"pkl_files/Resq_i_{date}.pkl", "wb") as output:
                pickle.dump(normResq_i, output)
            print(f"Resq_i saved as Resq_i_{date}.pkl")
```

Replace with (Step 3 has already added the `kernel_meta` kwarg):

```python
    if save_resqi == 1:
        meta_arrays = {k: np.asarray(v) for k, v in (kernel_meta or {}).items()}
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, Resq_i=normResq_i, **meta_arrays)
            print(f"Resq_i saved to {output_path}")
        else:
            # Legacy default: write to pkl_files/Resq_i_<date>.npz in CWD.
            check_folder("", "pkl_files")
            default_path = Path("pkl_files") / f"Resq_i_{date}.npz"
            np.savez(default_path, Resq_i=normResq_i, **meta_arrays)
            print(f"Resq_i saved as Resq_i_{date}.npz")
```

- [ ] **Step 5: Modify `kernel.py:generate_kernel` to build & pass `kernel_meta` and drop the sidecar write**

Edit `src/dfxm_geo/reciprocal_space/kernel.py`. Current sidecar write at lines 120-144:

```python
    vars_used = {
        "Nrays": Nrays,
        ...
    }

    vars_path.parent.mkdir(parents=True, exist_ok=True)
    vars_path.write_text(str(vars_used))
```

Replace with: keep `vars_used` (rename to `kernel_meta` for clarity), drop the `vars_path` plumbing entirely. New body of `generate_kernel`:

```python
def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    npoints1: int = 400,
    # ... unchanged ...
    output_path: Path | None = None,
) -> Path:
    """Run the kernel-generation Monte Carlo and write the npz to ``pkl_files/``."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    if output_path is not None:
        output_path = Path(output_path)

    kernel_meta = {
        "Nrays": np.int64(Nrays),
        "npoints1": np.int64(npoints1),
        "npoints2": np.int64(npoints2),
        "npoints3": np.int64(npoints3),
        "qi1_range": np.float64(qi1_range),
        "qi2_range": np.float64(qi2_range),
        "qi3_range": np.float64(qi3_range),
        "zeta_v_fwhm": np.float64(zeta_v_fwhm),
        "zeta_h_fwhm": np.float64(zeta_h_fwhm),
        "NA_rms": np.float64(NA_rms),
        "eps_rms": np.float64(eps_rms),
        "theta": np.float64(theta),
        "D": np.float64(D),
        "d1": np.float64(d1),
        "phys_aper": np.float64(phys_aper),
        "beamstop": np.bool_(beamstop),
        "bs_height": np.float64(bs_height),
        "aperture": np.bool_(aperture),
        "knife_edge": np.bool_(knife_edge),
        "dphi_range": np.float64(dphi_range),
    }

    reciprocal_res_func(
        Nrays,
        npoints1,
        # ... (unchanged positional + kwargs through dphi_range) ...
        dphi_range=dphi_range,
        output_path=output_path,
        kernel_meta=kernel_meta,
    )

    return output_path if output_path is not None else Path("pkl_files") / f"Resq_i_{date}.npz"
```

Drop `vars_used`, `vars_path`, `vars_path.write_text(str(vars_used), encoding="utf-8")` entirely (kernel.py lines 120-144). Also drop the `vars_path` derivation block at `kernel.py:84-93` (the `# Resolve the destination...` comment plus the `if output_path is not None: ... else: vars_path = ...` if/else). The new code only needs `if output_path is not None: output_path = Path(output_path)`. Flip the final-return default filename `Resq_i_{date}.pkl` → `Resq_i_{date}.npz` at `kernel.py:146`.

- [ ] **Step 6: Run existing tests to confirm the writer change doesn't break upstream tests**

Run from worktree root:
```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_reciprocal_resolution.py tests/test_kernel_cli.py -q
```

Expected: most tests still pass. `test_kernel_cli.py` may fail on tests that assume `.pkl` / sidecar `.txt` filenames or content — note the failure list; we'll fix in Task 5.

- [ ] **Step 7: Add a round-trip test against the writer**

In `tests/test_kernel_format.py`, append a new class:

```python
class TestGenerateKernelWritesNpz:
    """The generate_kernel function writes a .npz with bundled params."""

    def test_generate_kernel_writes_npz_with_bundled_meta(self, tmp_path: Path) -> None:
        """generate_kernel(output_path=tmp_path/foo.npz) produces a valid npz."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        dst = tmp_path / "Resq_i_test.npz"
        generate_kernel(
            output_path=dst,
            Nrays=1000,         # tiny; just exercise the write path
            npoints1=4,
            npoints2=4,
            npoints3=4,
        )

        assert dst.is_file()
        loaded = np.load(dst)
        assert "Resq_i" in loaded.files
        assert loaded["Resq_i"].shape == (4, 4, 4)
        assert int(loaded["Nrays"]) == 1000
        assert int(loaded["npoints1"]) == 4
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        # Round-trip the write-only audit fields too — these aren't read by
        # _load_default_kernel but must travel intact for reproducibility.
        assert float(loaded["theta"]) > 0  # Bragg angle, set by _default_theta_al_111
        assert float(loaded["D"]) == pytest.approx(2 * np.sqrt(50e-6 * 1.6e-3))
        assert bool(loaded["beamstop"]) is True
        assert bool(loaded["aperture"]) is True

        # Sidecar must NOT exist
        sidecar = dst.with_name(dst.stem + "_vars.txt")
        assert not sidecar.exists(), f"unexpected sidecar at {sidecar}"
```

- [ ] **Step 8: Run the new test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestGenerateKernelWritesNpz -v
```

Expected: PASS.

- [ ] **Step 8b: Debug if Step 8 fails** (skip if Step 8 passed)

Common failure modes and fixes:
- `NameError: name 'kernel_meta' is not defined` in resolution.py → Step 3 wasn't applied or wasn't applied first; re-check the signature.
- `TypeError: reciprocal_res_func() got unexpected keyword argument 'kernel_meta'` → same; Step 3 missing.
- `ImportError: pickle ...` or stray `pickle.dump` reference → Step 3's `import pickle` removal incomplete or Step 4's body change incomplete.
- Sidecar `_vars.txt` written anyway → Step 5's `vars_path.write_text` removal incomplete.

Re-edit per the failure, then re-run Step 8.

- [ ] **Step 9: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/resolution.py src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_format.py
git commit -m "feat(kernel): write .npz with bundled scalar params (replaces pickle.dump)

reciprocal_res_func now takes a kernel_meta dict and writes the resolution
array + all generation params into one .npz via np.savez. generate_kernel
builds the meta dict and drops the _vars.txt sidecar.

Layer 1 round-trip + param-bundling tests added; existing forward-model
loader still expects pickle and will break — fixed in the next commit."
```

---

## Task 4: Switch the loader (`_load_default_kernel`) from `pickle.load` to `np.load`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py:55-58, 244-252, 388-394`
- Test: `tests/test_kernel_format.py` (extend)

- [ ] **Step 1: Write a failing test that asserts the loader handles .npz**

Append to `tests/test_kernel_format.py`:

```python
class TestLoadDefaultKernel:
    """Layer 1 + 2: the loader resolves npz paths and populates module state."""

    def test_load_default_kernel_loads_npz(self, tmp_path: Path) -> None:
        """_load_default_kernel reads an .npz and sets module globals correctly."""
        import dfxm_geo.direct_space.forward_model as fm

        # Build a tiny synthetic npz
        dst = tmp_path / "Resq_i_test.npz"
        np.savez(
            dst,
            Resq_i=np.ones((4, 4, 4), dtype=np.float64),
            qi1_range=np.float64(1e-3),
            qi2_range=np.float64(2e-3),
            qi3_range=np.float64(3e-3),
            npoints1=np.int64(4),
            npoints2=np.int64(4),
            npoints3=np.int64(4),
        )

        saved_resq = fm.Resq_i
        saved_qi1 = fm.qi1_range
        try:
            fm._load_default_kernel(pkl_path=str(dst), compute_Hg=False)
            assert fm.Resq_i is not None
            assert fm.Resq_i.shape == (4, 4, 4)
            assert np.array_equal(fm.Resq_i, np.ones((4, 4, 4)))
            assert fm.qi1_range == pytest.approx(1e-3)
            assert fm.npoints1 == 4
        finally:
            fm.Resq_i = saved_resq
            fm.qi1_range = saved_qi1
```

- [ ] **Step 2: Run the test — it should fail with a pickle UnpicklingError or similar**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestLoadDefaultKernel -v
```

Expected: FAIL (e.g. `_pickle.UnpicklingError: ... not a pickle`).

- [ ] **Step 3: Modify `_load_default_kernel` body at lines 244-257 from pickle to npz**

Current block (`forward_model.py:244-257`):

```python
    if pkl_path is None:
        pkl_path = os.path.join(pkl_fpath, pkl_fn)
    if vars_path is None:
        vars_path = os.path.join(pkl_fpath, vars_fn)

    print("Loading Resq_i.")
    with open(pkl_path, "rb") as f:
        Resq_i = pickle.load(f)
    with open(vars_path, encoding="utf-8") as f:
        var_d = eval(f.read())
    qi1_range, npoints1 = var_d["qi1_range"], var_d["npoints1"]
    qi2_range, npoints2 = var_d["qi2_range"], var_d["npoints2"]
    qi3_range, npoints3 = var_d["qi3_range"], var_d["npoints3"]
    print("Resq_i loaded.")
```

Replace with (drop the `vars_path` defaulting; the npz reader handles materialization explicitly via `np.array(...)` so the loaded array survives the `with` block exit):

```python
    if pkl_path is None:
        pkl_path = os.path.join(pkl_fpath, pkl_fn)

    if str(pkl_path).endswith(".pkl"):
        raise RuntimeError(
            f"Detected legacy pickle at {pkl_path!r}; pickle support was "
            "removed in v1.0.3. Run `dfxm-bootstrap --config configs/default.toml` "
            "to regenerate the kernel as .npz."
        )

    print(f"Loading kernel from {pkl_path}.")
    with np.load(pkl_path) as arch:
        Resq_i = np.array(arch["Resq_i"])
        qi1_range = float(arch["qi1_range"])
        qi2_range = float(arch["qi2_range"])
        qi3_range = float(arch["qi3_range"])
        npoints1 = int(arch["npoints1"])
        npoints2 = int(arch["npoints2"])
        npoints3 = int(arch["npoints3"])
    print("Kernel loaded.")
```

Also remove `vars_path: str | None = None,` from the `_load_default_kernel` signature (currently at line 226) — its only use was the now-gone `eval` path. Remove `import pickle` from the imports (line 16).

- [ ] **Step 4: Update the module-level constants at `forward_model.py:55-58`**

Current:

```python
pkl_fpath = str(_REPO_ROOT / "reciprocal_space" / "pkl_files") + os.sep
pkl_fn = "Resq_i_20230913_1308.pkl"  # Change accordingly
vars_fn = os.path.splitext(pkl_fn)[0] + "_vars.txt"
```

Replace with:

```python
pkl_fpath = str(_REPO_ROOT / "reciprocal_space" / "pkl_files") + os.sep
# Constant name preserved (`pkl_fn`) so import-time monkeypatches in tests
# and the dfxm-bootstrap CLI don't break. Value is now the npz canonical.
pkl_fn = "Resq_i_20230913_1308.npz"  # Update after `dfxm-bootstrap` regen
```

(The `vars_fn` line at line 58 disappears.)

- [ ] **Step 5: Update the module docstring at `forward_model.py:7-10`**

Current:

```python
The reciprocal-space resolution kernel `Resq_i` is loaded lazily by
`_load_default_kernel()` — at module import iff the default pickle is on
disk, otherwise on explicit call. This lets the module be imported on a
clean clone or in CI without the precomputed pickle present.
```

Replace with:

```python
The reciprocal-space resolution kernel `Resq_i` is loaded lazily by
`_load_default_kernel()` — at module import iff the default .npz is on
disk, otherwise on explicit call. This lets the module be imported on a
clean clone or in CI without the precomputed kernel present.
```

- [ ] **Step 6: Update `forward()`'s init-state RuntimeError at `forward_model.py:290-296`**

Current:

```python
    if Resq_i is None:
        raise RuntimeError(
            "forward_model state is not initialized. Call "
            "_load_default_kernel(pkl_path, vars_path) before calling forward()."
        )
```

Replace with:

```python
    if Resq_i is None:
        raise RuntimeError(
            "forward_model state is not initialized. Call "
            "_load_default_kernel(pkl_path) before calling forward()."
        )
```

(Drop the `vars_path` argument from the error message — the parameter no longer exists.)

- [ ] **Step 7: Update the import-time auto-load block at `forward_model.py:409-418`**

Current:

```python
# Auto-load the default kernel iff it exists on disk. Preserves the
# pre-cleanup behavior for callers (e.g. init_forward.py) that expect
# `Resq_i`, `Hg`, `q_hkl`, etc. to be ready at import time.
if os.path.exists(os.path.join(pkl_fpath, pkl_fn)):
    _load_default_kernel()
else:
    print(
        f"NOTE: default kernel pickle not found at {os.path.join(pkl_fpath, pkl_fn)!r}; "
        f"call _load_default_kernel(pkl_path, vars_path) before forward()."
    )
```

Replace with:

```python
# Auto-load the default kernel iff it exists on disk. Preserves the
# pre-cleanup behavior for callers (e.g. init_forward.py) that expect
# `Resq_i`, `Hg`, `q_hkl`, etc. to be ready at import time.
if os.path.exists(os.path.join(pkl_fpath, pkl_fn)):
    _load_default_kernel()
else:
    print(
        f"NOTE: default kernel npz not found at {os.path.join(pkl_fpath, pkl_fn)!r}; "
        f"call _load_default_kernel(pkl_path) before forward(), or run `dfxm-bootstrap`."
    )
```

(Dropped `vars_path` from the print message; "pickle" → "npz".)

- [ ] **Step 8: Run the new loader test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestLoadDefaultKernel::test_load_default_kernel_loads_npz -v
```

Expected: PASS.

- [ ] **Step 9: Run the broader forward-model smoke test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_forward_model_smoke.py tests/test_forward_model_paths.py -v
```

Expected: most pass; any failures relate to the pickle->npz rename and need follow-up fixes in Task 5. Note them and proceed.

- [ ] **Step 10: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_kernel_format.py
git commit -m "feat(kernel): load .npz instead of pickle + sidecar eval

_load_default_kernel reads Resq_i and qi-grid params from an .npz archive.
The legacy pickle code path is replaced by a loud RuntimeError pointing
users at dfxm-bootstrap. Drops 'import pickle' and the eval(_vars.txt)
footgun in one shot."
```

---

## Task 5: Loud-error path for legacy pickle + fix-up tests broken by Task 4

**Files:**
- Test: `tests/test_kernel_format.py` (extend)
- Possibly modify: tests in `tests/test_kernel_cli.py`, `tests/test_pipeline.py`, `tests/test_pipeline_identification.py` that hardcode `.pkl` paths

- [ ] **Step 1: Write the failing test for the loud-error path**

Append to `tests/test_kernel_format.py`:

```python
class TestLegacyPickleRejection:
    """The loader refuses .pkl paths with a clear migration message."""

    def test_load_raises_on_pkl_path(self, tmp_path: Path) -> None:
        """A .pkl path raises RuntimeError mentioning dfxm-bootstrap."""
        import dfxm_geo.direct_space.forward_model as fm

        # Create a non-empty file with the wrong extension
        fake_pkl = tmp_path / "Resq_i_legacy.pkl"
        fake_pkl.write_bytes(b"\x80\x04\x95")  # pickle magic; never read

        with pytest.raises(RuntimeError, match="pickle support was removed"):
            fm._load_default_kernel(pkl_path=str(fake_pkl), compute_Hg=False)
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestLegacyPickleRejection -v
```

Expected: PASS (the loader already raises on .pkl per Task 4 Step 3).

- [ ] **Step 3: Audit + fix broken downstream tests**

Run the full suite once and list failures:

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -40
```

Expected failures are tests that hardcode `Resq_i_20230913_1308.pkl` or expect a sidecar. Exact edits (verified against current source):

**`tests/test_pipeline.py:622`** — Change:
```python
kernel_path = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.pkl"
```
to:
```python
kernel_path = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.npz"
```

Also the comment at line 619 (`"# Skip if the Resq_i kernel pickle is not on disk"`) and the skip message around line 624 (`"Kernel pickle {kernel_path} not present"`) — change "pickle" → "npz".

**`tests/test_pipeline_identification.py:410`** — Change:
```python
pkl = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.pkl"
```
to:
```python
pkl = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.npz"
```

Also update the docstring at line 402 (`"Skipped if the default Resq_i kernel pickle is missing"` → `"Skipped if the default Resq_i kernel npz is missing"`) and the skip message at line 412 (`f"kernel pickle missing: {pkl}"` → `f"kernel npz missing: {pkl}"`). Local variable name `pkl` may stay (renaming is scope creep).

**`tests/test_cluster_templates.py:217`** — Widen the regex pattern from:
```python
pattern = re.compile(r"Resq_i_\d+_\d+\.pkl|Resq_i_\*\.pkl")
```
to also catch hypothetical hardcoded `.npz` filenames:
```python
pattern = re.compile(r"Resq_i_\d+_\d+\.(pkl|npz)|Resq_i_\*\.(pkl|npz)")
```
And update the comment at lines 214-215 and the failure message at line 222 to say "kernel" instead of "kernel pickle".

**`tests/test_kernel_cli.py`** — multiple tests reference `Resq_i_*.pkl` paths or assert sidecar existence; expected breakages:
- `Resq_i_test.pkl` → `Resq_i_test.npz` (lines 19, 38)
- `Resq_i_explicit.pkl` → `Resq_i_explicit.npz` (lines 38, 40)
- `Resq_i_explicit_vars.txt` sidecar assertion at line 40 + 49 — replace with "sidecar must NOT exist" assertions (since params now bundled into npz). For each test that asserts `_vars.txt` existence: remove the assertion entirely. For tests that assert the sidecar is at a specific location: invert to assert it does NOT exist.
- `Resq_i_canonical.pkl` → `Resq_i_canonical.npz` (line 174)

Make the minimal edit per file (path string + the human-readable references to "pickle"). Do NOT change test logic.

**`tests/test_forward_model_paths.py:61`** — Update comment: `"forward_model prints a 'Loading Resq_i' or 'default kernel pickle not'"` → `"forward_model prints a 'Loading kernel from' or 'default kernel npz not'"`.

**`tests/test_forward_model_smoke.py:4`** — Update docstring `"no reciprocal-space kernel pickle present"` → `"no reciprocal-space kernel npz present"`.

- [ ] **Step 4: Re-run the suite**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -10
```

Expected: failures down to zero (or the kernel-pickle-missing skips). If anything else fails, debug per the message.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(kernel): loud-error on .pkl path; update tests off legacy pickle naming

Adds TestLegacyPickleRejection covering the RuntimeError path; updates
test_pipeline / test_pipeline_identification / test_kernel_cli to point
at .npz instead of .pkl. No production-code changes."
```

---

## Task 6: Forward-output bit-equivalence test (Layer 2)

**Files:**
- Test: `tests/test_kernel_format.py` (extend)

- [ ] **Step 1: Write the failing forward-equivalence test**

Append to `tests/test_kernel_format.py`:

```python
class TestForwardOutputBitEquivalence:
    """Layer 2: forward() output using npz-loaded kernel must match the
    pickle-era snapshot bit-for-bit.

    Skipped if either the existing pickle (to regenerate the npz from) or the
    golden snapshot is missing.
    """

    def test_forward_output_matches_pickle_era_snapshot(self, tmp_path: Path) -> None:
        import dfxm_geo.direct_space.forward_model as fm

        snapshot_path = GOLDEN_DIR / "forward_snapshot_pickle_era.npy"
        if not snapshot_path.exists():
            pytest.skip(f"snapshot not present at {snapshot_path}")

        # Build an npz from the existing pickle on disk by reading it once via
        # a deliberately-allowed pickle import in the *test*, then writing npz.
        # This isolates pickle.load to the test harness, not production code.
        # Task 9's import-audit is scoped to src/ files (via fm.__file__), so
        # this test-side pickle import does NOT trip the defensive guard.
        import pickle as _pkl

        legacy_pkl = Path(fm.pkl_fpath) / "Resq_i_20230913_1308.pkl"
        if not legacy_pkl.exists():
            pytest.skip(f"legacy pickle not present at {legacy_pkl}; cannot build comparison npz")

        with open(legacy_pkl, "rb") as f:
            Resq_i_from_pickle = _pkl.load(f)

        # Read the existing _vars.txt one last time (also via eval — test-only)
        vars_txt = legacy_pkl.with_name(legacy_pkl.stem + "_vars.txt")
        var_d = eval(vars_txt.read_text())

        dst = tmp_path / "Resq_i_from_pickle.npz"
        np.savez(
            dst,
            Resq_i=Resq_i_from_pickle,
            qi1_range=np.float64(var_d["qi1_range"]),
            qi2_range=np.float64(var_d["qi2_range"]),
            qi3_range=np.float64(var_d["qi3_range"]),
            npoints1=np.int64(var_d["npoints1"]),
            npoints2=np.int64(var_d["npoints2"]),
            npoints3=np.int64(var_d["npoints3"]),
        )

        saved_resq = fm.Resq_i
        saved_qi = (fm.qi1_range, fm.qi2_range, fm.qi3_range)
        saved_np = (fm.npoints1, fm.npoints2, fm.npoints3)
        try:
            fm._load_default_kernel(pkl_path=str(dst), compute_Hg=False)
            out = fm.forward(fm.Hg, phi=0.0, chi=0.0)
            if isinstance(out, tuple):
                out = out[0]
            golden = np.load(snapshot_path)
            assert np.array_equal(out, golden), "forward() output differs from pickle-era snapshot"
        finally:
            fm.Resq_i = saved_resq
            fm.qi1_range, fm.qi2_range, fm.qi3_range = saved_qi
            fm.npoints1, fm.npoints2, fm.npoints3 = saved_np
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestForwardOutputBitEquivalence -v
```

Expected: PASS if both the legacy pickle and the snapshot are present. SKIP otherwise. PASS proves the format change preserves forward() bit-exactly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_format.py
git commit -m "test(kernel): forward() output bit-equivalence vs pickle-era snapshot

Reads the legacy pickle once (via pickle.load isolated to the test, not
production code), bundles into an npz, runs forward() with the npz-loaded
kernel, asserts np.array_equal vs the pickle-era golden snapshot."
```

---

## Task 7: Cross-platform synthetic-kernel test (Layer 3)

**Files:**
- Test: `tests/test_kernel_format.py` (extend)

- [ ] **Step 1: Write the cross-platform test**

Append to `tests/test_kernel_format.py`:

```python
class TestCrossPlatformSyntheticKernel:
    """Layer 3: the committed synthetic golden loads regardless of platform."""

    def test_load_synthetic_kernel_committed_golden(self) -> None:
        """The 4x4x4 synthetic golden npz loads with all expected keys."""
        golden = GOLDEN_DIR / "synthetic_kernel.npz"
        assert golden.exists(), f"missing committed golden at {golden}"

        loaded = np.load(golden)
        expected_keys = {
            "Resq_i", "qi1_range", "qi2_range", "qi3_range",
            "npoints1", "npoints2", "npoints3", "Nrays",
        }
        assert set(loaded.files) == expected_keys

        assert loaded["Resq_i"].shape == (4, 4, 4)
        assert loaded["Resq_i"].dtype == np.float64
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        assert int(loaded["npoints1"]) == 4
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestCrossPlatformSyntheticKernel -v
```

Expected: PASS (the golden was committed in Task 2; numpy's npz reader handles cross-platform natively).

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_format.py
git commit -m "test(kernel): cross-platform synthetic-kernel load check (Layer 3)"
```

---

## Task 8: Performance regression bench (Layer 3, second half)

**Files:**
- Test: `tests/test_benchmarks.py` (extend)

- [ ] **Step 1: Write the load-time benchmark**

Match the existing pattern in `tests/test_benchmarks.py` — other benches use `@pytest.mark.bench` and let `pytest-benchmark --compare=baseline_v1_0_1` catch regressions via the saved baseline JSON. No per-test latency assertion needed; the 500 ms target is captured implicitly by the recorded baseline plus any future regression run.

Append to `tests/test_benchmarks.py`:

```python
@pytest.mark.bench
def test_load_default_kernel_bench(benchmark, tmp_path: Path) -> None:
    """Regression baseline for _load_default_kernel on the canonical npz size.

    The canonical kernel is (400, 200, 200) float64 = 128 MB; cold np.load
    typically lands at 100-300 ms on Sina's laptop. `pytest -m bench
    --benchmark-compare=baseline_v1_0_1` will flag regressions beyond the
    saved baseline's std-dev threshold.
    """
    import dfxm_geo.direct_space.forward_model as fm

    rng = np.random.default_rng(0)
    dst = tmp_path / "Resq_i_bench.npz"
    np.savez(
        dst,
        Resq_i=rng.random((400, 200, 200), dtype=np.float64),
        qi1_range=np.float64(5e-4),
        qi2_range=np.float64(0.75e-2),
        qi3_range=np.float64(0.75e-2),
        npoints1=np.int64(400),
        npoints2=np.int64(200),
        npoints3=np.int64(200),
    )

    saved = fm.Resq_i
    try:
        benchmark(fm._load_default_kernel, pkl_path=str(dst), compute_Hg=False)
    finally:
        fm.Resq_i = saved
```

- [ ] **Step 2: Run the bench**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_benchmarks.py::test_load_default_kernel_bench -v -m bench --benchmark-only
```

Expected: PASS, with the recorded mean in the 100-300 ms range for a fresh np.load on a 128 MB npz. The test records timing into `.benchmarks/`; future regressions surface via `--benchmark-compare=baseline_v1_0_1`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_benchmarks.py
git commit -m "test(kernel): pytest-benchmark regression for _load_default_kernel <500ms"
```

---

## Task 9: Defensive — assert `import pickle` removed from runtime modules

**Files:**
- Test: `tests/test_kernel_format.py` (extend)

- [ ] **Step 1: Write the import-audit test**

Append to `tests/test_kernel_format.py`:

```python
class TestNoPickleImportsInRuntime:
    """Defensive guard: runtime modules must not import pickle.

    Audit scope: src/ files only — accessed via `mod.__file__`, which resolves
    to the installed/editable src path, never tests/. Test files MAY use
    pickle (e.g. TestForwardOutputBitEquivalence) without tripping this guard.
    """

    def test_forward_model_does_not_import_pickle(self) -> None:
        """forward_model.py (src) must not have `import pickle` or `from pickle`."""
        import dfxm_geo.direct_space.forward_model as fm

        src = Path(fm.__file__).read_text(encoding="utf-8")
        assert "import pickle" not in src, "pickle import re-introduced in forward_model.py"
        assert "from pickle" not in src

    def test_kernel_module_does_not_import_pickle(self) -> None:
        """kernel.py and resolution.py (src) must not import pickle."""
        from dfxm_geo.reciprocal_space import kernel, resolution

        for mod in (kernel, resolution):
            src = Path(mod.__file__).read_text(encoding="utf-8")
            assert "import pickle" not in src, f"pickle import re-introduced in {mod.__file__}"
            assert "from pickle" not in src, f"pickle import re-introduced in {mod.__file__}"
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_format.py::TestNoPickleImportsInRuntime -v
```

Expected: PASS (Task 3 dropped the `import pickle` in resolution.py; Task 4 dropped it in forward_model.py).

If FAIL: find the lingering `import pickle` line and remove it. Re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_format.py
git commit -m "test(kernel): defensive guard against pickle import regression"
```

---

## Task 10: Docs update

**Files:**
- Modify: `docs/architecture.md` (Resq_i mentions)
- Modify: `docs/reproducibility.md` (pickle + sidecar mentions)

- [ ] **Step 1: Update `docs/architecture.md` — sites at lines 19-20, 37, 52, 59-62, 116-117**

Verified mentions (Grep against current source):
- **Line 19-20**: tree comment `kernel.py — Driver script: generates Resq_i pickle` → change "pickle" → "npz".
- **Line 37**: `generate_Resq_i, exposure_time}.py are kept as **deprecation shims**` — no change (this refers to the legacy module names, not the pickle format).
- **Line 52**: `1. **Generate the reciprocal-space resolution kernel `Resq_i`** (one-off,` — no change to the bullet itself; the paragraph below it needs updating.
- **Lines 59-62**: `This writes pkl_files/Resq_i_<timestamp>.pkl plus a sidecar _vars.txt describing the parameters used. The pickle is picked up at ... matches pkl_fn there.` → replace with: "This writes `pkl_files/Resq_i_<timestamp>.npz` with the array and all generation parameters bundled into the single archive (no separate sidecar). The npz is picked up at import time by `_load_default_kernel()` whenever its filename matches `pkl_fn` in `forward_model.py`."
- **Lines 116-117**: `Resq_i pickle (if present on disk) via _load_default_kernel(), or defers loading until the user calls _load_default_kernel(pkl_path)` → change "pickle" → "npz"; keep `_load_default_kernel(pkl_path)` signature (already correct after Task 4's signature change).

- [ ] **Step 2: Update `docs/reproducibility.md` — sites at lines 12-13, 17, 20, 25, 97, 191**

Verified mentions:
- **Lines 12-13**: `A precomputed reciprocal-space resolution kernel pickle (Resq_i_*.pkl) with its sidecar _vars.txt. This is **not** in the repo (it is large` → `A precomputed reciprocal-space resolution kernel npz (Resq_i_*.npz) with all generation parameters bundled into the same archive. This is **not** in the repo (it is large`.
- **Line 17**: `python reciprocal_space/generate_Resq_i.py` — leave as-is (legacy command path; the deprecation shim still routes correctly).
- **Line 20**: `Place the resulting Resq_i_<timestamp>.pkl + <timestamp>_vars.txt` → `Place the resulting Resq_i_<timestamp>.npz`.
- **Line 25**: `_load_default_kernel(pkl_path, vars_path) explicitly before invoking` → `_load_default_kernel(pkl_path) explicitly before invoking` (drop `vars_path` arg — no longer exists after Task 4).
- **Line 97**: `Reciprocal-space kernel parameters | hard-coded in generate_Resq_i.py | Independent generation step.` — no change (refers to hard-coded defaults in the generator, not the output format).
- **Line 191**: `- The Resq_i_*.pkl filename and its _vars.txt sidecar.` → `- The Resq_i_*.npz filename (params bundled, no separate sidecar).`

After the substantive edits, add a new paragraph near the top of the "Reciprocal-space kernel" section:

```markdown
> **v1.0.3 migration note:** Legacy `.pkl` files from before v1.0.3 are no
> longer supported by `_load_default_kernel`. Regenerate via
> `dfxm-bootstrap --config configs/default.toml` (~50 s); the resulting
> `.npz` is a drop-in replacement at the same canonical path.
```

- [ ] **Step 3: Verify doc-validation tests still pass**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_readme_sections.py tests/test_cluster_templates.py -v
```

Expected: PASS. If any test validates docs content and trips on the new wording, update the assertion.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md docs/reproducibility.md
git commit -m "docs(kernel): refresh architecture + reproducibility for npz format

Stage-1 kernel artifact is now a single .npz with bundled params (no
sidecar). Drop pkl_path + vars_path argument pair from the reproducibility
guide; note v1.0.3 legacy-pickle removal."
```

---

## Task 11: Full smoke + mypy + final verification

**Files:**
- None (verification only)

- [ ] **Step 1: Full smoke suite**

```bash
PYTHONPATH=src /c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q
```

Expected: all tests pass (Phase-8-era baseline was 281 passed; this PR adds ~9 new tests and changes a handful of existing skip messages). The 2 kernel-pickle-dependent skips persist as kernel-npz-dependent skips (same files, new extension). Bench-marked tests deselect by default.

If anything red, fix per the message before committing.

- [ ] **Step 2: mypy**

```bash
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```

Expected: 0 errors.

- [ ] **Step 3: ruff**

```bash
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

Expected: no issues.

- [ ] **Step 4: Final summary commit (if any leftover docstring / lint fixes)**

If any small cleanups were needed in Steps 1-3, commit them:

```bash
git add <files>
git commit -m "chore: lint + type cleanup after kernel npz migration"
```

---

## Deployment note (post-merge action, not part of this PR)

After this PR merges as v1.0.3, **Sina runs**:

```bash
dfxm-bootstrap --config configs/default.toml
```

This regenerates the canonical kernel as `reciprocal_space/pkl_files/Resq_i_<today-timestamp>.npz`. The `pkl_fn` constant in `forward_model.py` needs updating to the new filename. This is a one-time op (~50 s wall time on Sina's laptop).

Alternatively the `pkl_fn` value can stay at `Resq_i_20230913_1308.npz` if Sina passes `--output` to `dfxm-bootstrap` to land the new generation at that path.

---

## Self-Review

**Spec coverage** (against the resolved design in `followups_kernel_pickle_alternatives.md`):
- npz format → Tasks 3 + 4 ✓
- params bundled into npz, no sidecar → Tasks 3 (writer drops sidecar) + 4 (loader skips sidecar) ✓
- Loud error on legacy .pkl → Task 5 ✓
- Round-trip + param-bundling tests (Layer 1) → Task 3 ✓
- Forward-output bit-equivalence (Layer 2) → Task 6 ✓
- Defensive `import pickle` removal (Layer 2) → Task 9 ✓
- Cross-platform synthetic golden (Layer 3) → Tasks 2 + 7 ✓
- Performance regression bench (Layer 3) → Task 8 ✓
- Docs update → Task 10 ✓
- v1.0.3 ship target → addressed in deployment note ✓
- No --migrate helper (B2) → not in scope; explicitly omitted ✓

**Placeholder scan:** No "TBD", "implement later", or "add appropriate error handling" found. Code blocks shown for all code steps. Exact paths used throughout.

**Type consistency:**
- `kernel_meta` dict (Task 3) is the same key used in `reciprocal_res_func` and `generate_kernel`. ✓
- `pkl_fn` constant retained (not renamed) per loader-API stability. ✓
- `Resq_i`, `qi1_range`, `qi2_range`, `qi3_range`, `npoints1`, `npoints2`, `npoints3` are the consistent npz-key names across writer, loader, and all tests. ✓
- `forward_snapshot_pickle_era.npy` filename consistent in Task 1 (create), Task 6 (read). ✓
- `synthetic_kernel.npz` filename consistent in Task 2 (create), Task 7 (read). ✓
