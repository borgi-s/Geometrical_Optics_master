# Sub-project E: Identification HDF5 + master/per-scan layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship sub-project E for v1.2.0: identification mode writes HDF5 (BLISS schema), forward mode retrofitted to a master + per-scan-dirs layout matching ESRF beamline disk shape, all three identify sub-modes consume `[scan.<axis>]` from B+C, plus migration tools.

**Architecture:** A `MasterWriter` context manager owns the master HDF5 (one per run); per-scan LIMA-style detector files live in `scan0001/`, `scan0002/`, … subdirs and are linked into `/N.1/instrument/dfxm_sim_detector` via `ExternalLink`. Runners (`_run_identification_*`) become generators yielding `ScanSpec` tuples; a shared `write_identification_h5` orchestrator drives detector-file writes and master `add_scan` calls. Forward (`write_simulation_h5`) uses the same machinery. The `render_per_dislocation` opt-in produces three detector files per multi-mode scan dir.

**Tech Stack:** Python 3.11+, h5py, numpy, tqdm, pytest. Existing `dfxm_geo.io.hdf5` is refactored in place (no new module). `io/migrate.py` extends with `cli_main_h5_to_h5`.

**Locked design decisions from spec open questions:**
- **Poisson noise + `render_per_dislocation`:** The combined detector receives Poisson noise (existing noise stream). Per-dis detectors are **noiseless** (deterministic ML labels). One Poisson draw per multi sample; per-dis files contain `image_arr * intensity_scale` only.
- **Test file location:** flat under `tests/` (not `tests/io/`), matching the repo's existing convention. Spec's `tests/io/test_*.py` paths are translated to `tests/test_*.py`.
- **Symlink caveat:** documented in `docs/output-format.md` only; no code change.
- **Many small files for `dfxm-identify single`:** accepted as-is for v1.2.0. `[io].batch_scans_per_file` knob deferred to v1.3.0+.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/dfxm_geo/io/hdf5.py` | Helpers, `_write_detector_file`, `_compute_and_write_detector_file_parallel`, `MasterWriter`, `write_simulation_h5`, `write_identification_h5` | Major refactor (in place) |
| `src/dfxm_geo/io/migrate.py` | `migrate_npy_dir_to_h5` (updated for new layout), new `migrate_h5_master_to_master`, `cli_main_h5_to_h5` | Extend |
| `src/dfxm_geo/pipeline.py` | `run_simulation` rewired to new layout; `_run_identification_*` become generators yielding `ScanSpec`; `run_identification` consumes generator via `write_identification_h5`; eager guards | Heavy edits |
| `pyproject.toml` | Add `dfxm-migrate-h5` entry point | Add 1 line |
| `configs/identification_zscan.toml` | Move `[zscan].phi_*/chi_*` → `[scan.phi]/[scan.chi]` | Edit |
| `docs/output-format.md` | Rewrite for master + per-scan layout | Rewrite |
| `docs/release-notes-1.2.0.md` | NEW; v1.2.0 changelog (A + D + B + C + E) | Create |
| `tests/test_master_writer.py` | `MasterWriter` unit tests | Create |
| `tests/test_detector_file.py` | `_write_detector_file` + parallel writer unit tests | Create |
| `tests/test_pipeline_identification_hdf5.py` | End-to-end identify (single/multi/z-scan) → HDF5 | Create |
| `tests/test_identification_multi_per_dis.py` | `render_per_dislocation` opt-in | Create |
| `tests/test_identification_scan_modes.py` | Identify with `[scan.phi]/[scan.chi]` grids | Create |
| `tests/test_migrate_h5.py` | `dfxm-migrate-h5` round-trip | Create |
| `tests/test_pipeline.py`, `test_hdf5_*`, `test_pipeline_identification.py`, `test_pipeline_scan_modes.py`, `test_pipeline_crystal_modes.py`, `test_pipeline_multi_reflection.py`, `test_migrate_output.py` | Fixture updates for new layout; rewrite identify tests off `.npy`/manifest | Edit |
| `CLAUDE.md` (one level up from repo) | Update pipeline-features arc + tag chain | Edit |

---

## Quality gates (run after every phase)

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected baseline before phase 1: 430 passed / 4 skipped / 10 deselected / 1 xfailed; mypy 0 errors.

---

## Phase 1: Naming constants + LIMA detector file writer

**Goal:** Land the foundation that every later phase calls: naming constants, an in-memory `_write_detector_file`, and a parallel `_compute_and_write_detector_file_parallel`. These are pure additions — no existing call sites change yet.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_detector_file.py`

### Task 1.1: Add naming constants

- [ ] **Step 1: Open `src/dfxm_geo/io/hdf5.py`. After the existing import block (after line ~28, before `def _set_nx_class`), insert:**

```python
# Output layout constants (v1.2.0).
MASTER_FORWARD = "dfxm_geo.h5"
MASTER_IDENTIFY = "dfxm_identify.h5"
SCAN_DIR_FMT = "scan{:04d}"
DETECTOR_FILE_FMT = "{name}_0000.h5"
DETECTOR_INTERNAL_PATH = "/entry_0000/dfxm_sim_detector/image"
```

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py
git commit -m "io: add v1.2.0 layout naming constants"
```

### Task 1.2: Write the failing test for `_write_detector_file`

- [ ] **Step 1: Create `tests/test_detector_file.py`:**

```python
"""Unit tests for the LIMA-style per-scan detector file writer."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    _compute_and_write_detector_file_parallel,
    _compute_frame,
    _write_detector_file,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel


def test_write_detector_file_structure(tmp_path: Path) -> None:
    stack = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    out = tmp_path / "dfxm_sim_detector_0000.h5"

    _write_detector_file(out, stack)

    assert out.is_file()
    with h5py.File(out, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["creator"] == "dfxm-geo"
        assert f.attrs["default"] == "entry_0000"
        e = f["entry_0000"]
        assert e.attrs["NX_class"] == "NXentry"
        det = e["dfxm_sim_detector"]
        assert det.attrs["NX_class"] == "NXdetector"
        img = det["image"]
        assert img.shape == (2, 3, 4)
        assert img.attrs["interpretation"] == "image"
        assert img.chunks == (1, 3, 4)
        assert img.compression == "gzip"
        assert img.compression_opts == 4
        np.testing.assert_array_equal(img[...], stack)
        # NXdata + measurement soft links
        plot = e["plot"]
        assert plot.attrs["NX_class"] == "NXdata"
        assert plot.attrs["signal"] == "image"
        np.testing.assert_array_equal(plot["image"][...], stack)
        np.testing.assert_array_equal(
            e["measurement"][...], stack
        )  # h5py auto-follows SoftLink to a dataset


def test_write_detector_file_internal_path_matches_constant(tmp_path: Path) -> None:
    out = tmp_path / "det.h5"
    _write_detector_file(out, np.zeros((1, 2, 2)))
    with h5py.File(out, "r") as f:
        # The dataset at DETECTOR_INTERNAL_PATH is what ExternalLink targets will use.
        assert DETECTOR_INTERNAL_PATH in f
        assert f[DETECTOR_INTERNAL_PATH].shape == (1, 2, 2)
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_file.py -v
```

Expected: ImportError on `_write_detector_file`, both tests fail.

### Task 1.3: Implement `_create_detector_skeleton` helper + `_write_detector_file`

The two LIMA-style writers (`_write_detector_file` and the parallel
variant added in Task 1.5) share the same NX-root / NXentry / NXdetector
/ NXdata + softlink scaffolding. Extract the scaffolding into a private
helper that both call so the writers stay short and the file shape is
defined exactly once. The helper either inlines a full image stack
(``data=...``, used by the serial writer) or pre-allocates a streaming
dataset (``data=None``, used by the parallel writer in Task 1.5).

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, after `_compute_frame` (around line ~103), add the `_FrameArgs` alias and the skeleton helper:**

```python
# (frame_idx, Hg, phi, chi) — one frame's worth of args for `_compute_frame`.
_FrameArgs = tuple[int, np.ndarray, float, float]


def _create_detector_skeleton(
    f: h5py.File,
    *,
    n_frames: int,
    height: int,
    width: int,
    data: np.ndarray | None = None,
) -> h5py.Dataset:
    """Create the LIMA-style NX skeleton in ``f`` and return the image dataset.

    Builds /entry_0000 (NXentry) → dfxm_sim_detector (NXdetector) → image,
    plus the /entry_0000/plot (NXdata) and /entry_0000/measurement SoftLinks
    that point at `DETECTOR_INTERNAL_PATH`. When ``data`` is given, the
    dataset is created with the array inline (full-stack write). When
    ``data is None`` the dataset is pre-allocated at (n_frames, height,
    width) for streaming writes from worker threads.
    """
    f.attrs["NX_class"] = "NXroot"
    f.attrs["creator"] = "dfxm-geo"
    f.attrs["default"] = "entry_0000"
    entry = f.create_group("entry_0000")
    _set_nx_class(entry, "NXentry")
    det = entry.create_group("dfxm_sim_detector")
    _set_nx_class(det, "NXdetector")
    if data is not None:
        img = det.create_dataset(
            "image",
            data=data,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    else:
        img = det.create_dataset(
            "image",
            shape=(n_frames, height, width),
            dtype=np.float64,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    img.attrs["interpretation"] = "image"
    plot = entry.create_group("plot")
    _set_nx_class(plot, "NXdata")
    plot.attrs["signal"] = "image"
    plot["image"] = h5py.SoftLink(DETECTOR_INTERNAL_PATH)
    entry["measurement"] = h5py.SoftLink(DETECTOR_INTERNAL_PATH)
    return img


def _write_detector_file(path: Path, image_stack: np.ndarray) -> None:
    """Write a pre-computed (N, H, W) image stack as a LIMA-style detector file.

    Produces /entry_0000/dfxm_sim_detector/image with chunks=(1, H, W),
    gzip-4 + shuffle, @interpretation="image", plus NXdata/measurement
    soft-links to that dataset. Used by identification single/multi paths
    when frames are computed serially in RAM.
    """
    _, h, w = image_stack.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _create_detector_skeleton(
            f, n_frames=image_stack.shape[0], height=h, width=w, data=image_stack
        )
```

- [ ] **Step 2: Run test to confirm pass**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_file.py -v
```

Expected: 2 passed.

### Task 1.4: Write the failing test for the parallel detector writer

- [ ] **Step 1: Append to `tests/test_detector_file.py`:**

```python
def test_compute_and_write_detector_file_parallel_roundtrip(tmp_path: Path) -> None:
    """Workers run forward() and stream into one detector file; pixels match a
    serial reference (probed-frame-0 plus the workers' results)."""
    # `forward()` requires Hg of shape (NN1*NN2*NN3, 3, 3) — ~1.5M rows at the
    # default kernel. We can't construct that from scratch; reuse the kernel-
    # loaded `fm.Hg` (the canonical pattern; see also test_hdf5_writer.py:
    # test_save_scan_parallel_to_h5_uses_w2_pattern). Imports for fm,
    # _lookup_and_load_kernel, and the parallel writer live at module scope
    # alongside `_write_detector_file`.
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    if fm.Hg is None:
        pytest.skip("forward_model.Hg not populated; run dfxm-bootstrap.")
    try:
        Hg = fm.Hg
        args = [
            (0, Hg, 0.0, 0.0),
            (1, Hg, 1e-5, 0.0),
            (2, Hg, 0.0, 1e-5),
            (3, Hg, 1e-5, 1e-5),
        ]
        out = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
        _compute_and_write_detector_file_parallel(out, args, max_workers=2)

        # Reference: run forward() serially using _compute_frame
        ref = np.empty((4,) + _compute_frame(args[0])[1].shape, dtype=np.float64)
        for a in args:
            idx, im = _compute_frame(a)
            ref[idx] = im
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f[DETECTOR_INTERNAL_PATH][...], ref)
    finally:
        # Restore the kernel-state sentinels so downstream tests that branch on
        # `fm.Hg is None` / `fm._loaded_kernel_path is None` retain their baseline
        # skip behavior. (There's a latent ordering bug in
        # test_hdf5_provenance.TestHdf5NewAttrs._reset_kernel_state — it resets
        # _loaded_kernel_path but not Resq_i/Hg/qi*_start — that this teardown
        # masks. Out of Phase-1 scope; flag for later.)
        fm.Hg = None
        fm._loaded_kernel_path = None
```

- [ ] **Step 2: Run test to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_file.py::test_compute_and_write_detector_file_parallel_roundtrip -v
```

Expected: ImportError on `_compute_and_write_detector_file_parallel`.

### Task 1.5: Implement `_compute_and_write_detector_file_parallel`

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, after `_write_detector_file`, add:**

```python
def _compute_and_write_detector_file_parallel(
    path: Path,
    args_list: list[_FrameArgs],
    *,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    """Producer-consumer parallel writer for a LIMA-style detector file.

    Workers run `forward()` via `_compute_frame` and stream results into a
    pre-allocated (N_frames, H, W) dataset. The dataset is at the canonical
    `DETECTOR_INTERNAL_PATH` so masters can ExternalLink to it.

    Args:
        path: Output detector file. Parent dirs are created if missing.
        args_list: Sequence of (frame_idx, Hg, phi, chi) tuples, one per frame.
            Frame indices must be a contiguous 0..N-1 set.
        max_workers: Override for `_auto_max_workers()`.
        detector_shape: (H, W). If None, probes args_list[0] to discover shape.
    """
    if not args_list:
        raise ValueError("args_list must contain at least one frame")
    n_frames = len(args_list)

    if detector_shape is None:
        # probe doubles as frame-0 result when shape unknown
        probe_idx, probe_im = _compute_frame(args_list[0])
        h, w = probe_im.shape
    else:
        h, w = detector_shape
        probe_idx, probe_im = None, None

    path.parent.mkdir(parents=True, exist_ok=True)
    workers = max_workers if max_workers is not None else _auto_max_workers()

    with h5py.File(path, "w") as f:
        img = _create_detector_skeleton(f, n_frames=n_frames, height=h, width=w)

        if probe_im is not None:
            img[probe_idx] = probe_im
            workers_args = args_list[1:]
        else:
            workers_args = args_list

        if workers_args:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for k, im in tqdm(
                    ex.map(_compute_frame, workers_args),
                    total=len(workers_args),
                ):
                    img[k] = im
```

- [ ] **Step 2: Run test to confirm pass**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_file.py -v
```

Expected: 3 passed.

### Task 1.6: Verify full suite still green; commit

- [ ] **Step 1: Run the full suite**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
```

Expected: 433 passed / 4 skipped / 10 deselected / 1 xfailed (3 new tests added on top of baseline 430).

- [ ] **Step 2: Run mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_detector_file.py
git commit -m "io: add LIMA-style per-scan detector file writer (E foundation)"
```

---

## Phase 2: `MasterWriter` context manager

**Goal:** A single class that owns the master HDF5 handle for one run, exposes `add_scan(...)` for each `/N.1`, and writes `/dfxm_geo/` provenance on `__exit__`.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_master_writer.py`

### Task 2.1: Failing test — open/close + provenance once

- [ ] **Step 1: Create `tests/test_master_writer.py`:**

```python
"""Unit tests for the MasterWriter context manager."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
)


@pytest.fixture(autouse=True)
def _reset_kernel_state():
    """Restore module-level forward_model state after each test.

    `_kernel_for_tests()` loads a kernel via `_lookup_and_load_kernel`,
    which sets `fm.Hg` and `fm._loaded_kernel_path` as side effects.
    Downstream tests (e.g. the `TestHdf5NewAttrs` baseline-skip pattern in
    `tests/test_detector_file.py`) rely on `_loaded_kernel_path is None`,
    so we reset both after every test in this file to avoid cross-test
    bleed.
    """
    yield
    import dfxm_geo.direct_space.forward_model as fm

    fm.Hg = None
    fm._loaded_kernel_path = None


def _kernel_for_tests() -> Path:
    """Pick the bundled kernel for provenance + lookup tests."""
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    import dfxm_geo.direct_space.forward_model as fm

    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    assert fm._loaded_kernel_path is not None
    return Path(fm._loaded_kernel_path)


def test_master_writer_open_close_writes_provenance(tmp_path: Path) -> None:
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    with MasterWriter(
        master_path,
        cli="pytest test_master_writer",
        config_toml="[scan]\nphi_range = 0.01\n",
        kernel_npz=kernel,
    ):
        pass  # no add_scan calls

    assert master_path.is_file()
    with h5py.File(master_path, "r") as f:
        assert "/dfxm_geo" in f
        g = f["/dfxm_geo"]
        assert g["cli"][()].decode() == "pytest test_master_writer"
        assert g["config_toml"][()].decode().startswith("[scan]")
        assert "kernel" in g
        assert g["kernel"]["pkl_fn"][()].decode() == kernel.name


def test_master_writer_provenance_written_exactly_once(tmp_path: Path) -> None:
    """Re-entering the context manager on the same file must not double-write."""
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    with MasterWriter(master_path, cli="first", config_toml="", kernel_npz=kernel):
        pass
    # Second open should overwrite cleanly (no duplicate-key errors).
    with MasterWriter(master_path, cli="second", config_toml="", kernel_npz=kernel):
        pass
    with h5py.File(master_path, "r") as f:
        assert f["/dfxm_geo/cli"][()].decode() == "second"
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_master_writer.py::test_master_writer_open_close_writes_provenance -v
```

Expected: ImportError on `MasterWriter`.

### Task 2.2: Implement `MasterWriter` open/close

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, after `_write_provenance` (around line ~228), add:**

```python
class MasterWriter:
    """Context manager that owns the master HDF5 handle for one pipeline run.

    Use:
        with MasterWriter(path, cli=..., config_toml=..., kernel_npz=...) as m:
            m.add_scan(scan_id="1.1", ...)
            m.add_scan(scan_id="2.1", ...)
        # provenance written on close

    The master is opened in mode 'w' (fresh file every run) so the writer
    is idempotent across re-runs to the same path.
    """

    def __init__(
        self,
        path: Path,
        *,
        cli: str,
        config_toml: str,
        kernel_npz: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.cli = cli
        self.config_toml = config_toml
        self.kernel_npz = kernel_npz
        self._fh: h5py.File | None = None

    def __enter__(self) -> "MasterWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = h5py.File(self.path, "w")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._fh is not None and exc_type is None:
                _write_provenance(
                    self._fh,
                    cli=self.cli,
                    kernel_npz=self.kernel_npz,
                    config_toml=self.config_toml,
                )
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
```

- [ ] **Step 2: Run tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_master_writer.py -v
```

Expected: 2 passed.

### Task 2.3: Failing test — `add_scan` writes a single `/N.1` correctly

- [ ] **Step 1: Append to `tests/test_master_writer.py`:**

```python
def test_add_scan_writes_external_link_and_metadata(tmp_path: Path) -> None:
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    # Pre-create a detector file the master will link to.
    scan_dir = tmp_path / "scan0001"
    det_path = scan_dir / "dfxm_sim_detector_0000.h5"
    stack = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    _write_detector_file(det_path, stack)

    with MasterWriter(
        master_path, cli="t", config_toml="", kernel_npz=kernel
    ) as m:
        m.add_scan(
            scan_id="1.1",
            title="fscan2d phi -0.01 0.01 2 chi 0 0 1 1.0",
            start_time="2026-05-21T10:00:00",
            end_time="2026-05-21T10:00:05",
            sample={
                "name": "simulated, dislocation identification (single)",
                "slip_plane_normal": np.asarray([1, 1, 1], dtype=np.int32),
                "burgers": np.asarray([1, 0, 1], dtype=np.int32),
                "rotation_deg": 45.0,
            },
            positioners={"phi": np.array([-0.01, 0.01]), "chi": 0.0},
            detector_links={
                "dfxm_sim_detector": (
                    det_path.relative_to(tmp_path),
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={
                "Hg": np.eye(3).reshape(1, 3, 3),
                "q_hkl": np.array([0.0, 0.0, 1.0]),
                "theta": 11.5,
                "psize": 7.5e-7,
                "zl_rms": 1.0,
            },
            attrs={
                "scan_mode": "rocking",
                "scanned_axes": ["phi"],
                "identify_mode": "single",
            },
        )

    with h5py.File(master_path, "r") as f:
        assert "1.1" in f
        scan = f["1.1"]
        assert scan.attrs["NX_class"] == "NXentry"
        assert scan.attrs["scan_mode"] == "rocking"
        assert list(scan.attrs["scanned_axes"]) == ["phi"]
        assert scan.attrs["identify_mode"] == "single"
        # Sample metadata
        samp = scan["sample"]
        assert samp.attrs["NX_class"] == "NXsample"
        assert samp["name"][()].decode().startswith("simulated, dislocation")
        np.testing.assert_array_equal(samp["slip_plane_normal"][...], [1, 1, 1])
        np.testing.assert_array_equal(samp["burgers"][...], [1, 0, 1])
        assert samp["rotation_deg"][()] == 45.0
        # Positioners
        pos = scan["instrument"]["positioners"]
        assert pos.attrs["NX_class"] == "NXcollection"
        np.testing.assert_allclose(pos["phi"][...], np.degrees([-0.01, 0.01]))
        assert pos["chi"][()] == 0.0  # scalar fixed axis
        # ExternalLink to detector file resolves transparently
        det_ds = scan["instrument"]["dfxm_sim_detector"]["data"]
        np.testing.assert_array_equal(det_ds[...], stack)
        # And the link target itself is recorded as a relative POSIX path
        # (forward-slash regardless of host OS — required for HDF5 portability).
        link = scan["instrument"]["dfxm_sim_detector"].get(
            "data", getlink=True
        )
        assert isinstance(link, h5py.ExternalLink)
        assert link.filename == "scan0001/dfxm_sim_detector_0000.h5"


def test_add_scan_multi_detector_links(tmp_path: Path) -> None:
    """render_per_dislocation case: three NXdetector groups per /N.1."""
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_identify.h5"
    scan_dir = tmp_path / "scan0001"
    detectors = {}
    for name in ("dfxm_sim_detector", "dfxm_sim_detector_dis0", "dfxm_sim_detector_dis1"):
        p = scan_dir / f"{name}_0000.h5"
        _write_detector_file(p, np.zeros((1, 2, 2)))
        detectors[name] = (p.relative_to(tmp_path), DETECTOR_INTERNAL_PATH)

    with MasterWriter(
        master_path, cli="t", config_toml="", kernel_npz=kernel
    ) as m:
        m.add_scan(
            scan_id="1.1",
            title="t",
            start_time="t",
            end_time="t",
            sample={"name": "x"},
            positioners={"phi": 0.0, "chi": 0.0},
            detector_links=detectors,
            dfxm_geo={},
            attrs={},
        )

    with h5py.File(master_path, "r") as f:
        instr = f["/1.1/instrument"]
        for name in detectors:
            assert name in instr
            assert instr[name].attrs["NX_class"] == "NXdetector"
            # measurement soft-links exist per detector
            assert name in f["/1.1/measurement"]
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_master_writer.py -v
```

Expected: the two new tests fail with AttributeError on `add_scan`.

### Task 2.4: Implement `MasterWriter.add_scan`

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, inside the `MasterWriter` class, add the `add_scan` method. Append after `__exit__`:**

```python
    def add_scan(
        self,
        *,
        scan_id: str,
        title: str,
        start_time: str,
        end_time: str,
        sample: dict[str, object],
        positioners: dict[str, np.ndarray | float],
        detector_links: dict[str, tuple[Path, str]],
        dfxm_geo: dict[str, object],
        attrs: dict[str, str | list[str]],
    ) -> None:
        """Append one BLISS scan entry `/<scan_id>` to the master.

        Args:
            scan_id: BLISS scan identifier, e.g. "1.1", "2.1", "3.1".
            title: Scan title string (e.g. fscan2d command).
            start_time, end_time: ISO-8601 timestamps recorded on `/N.1`.
            sample: Dict of NXsample contents. Numpy arrays become datasets;
                scalars become 0-D datasets. Special key "name" expected.
                Nested dicts (e.g. "dislocations") become NXcollection groups.
            positioners: Dict of motor axis name → 1-D array (scanned) or
                scalar (fixed). Arrays are stored in degrees with units attr.
                ASSUMES input is in radians (matches existing convention).
            detector_links: Dict of detector name → (rel_file_path, internal_h5_path).
                Each becomes an NXdetector group with a `data` ExternalLink.
                A `/N.1/measurement/<name>` SoftLink is also created.
            dfxm_geo: Sim-specific per-scan metadata: Hg, q_hkl, theta, psize, zl_rms.
                Any subset may be supplied; missing keys are skipped.
            attrs: Per-`/N.1` attributes (scan_mode, scanned_axes,
                crystal_mode | identify_mode).
        """
        if self._fh is None:
            raise RuntimeError("MasterWriter is not open; use as a context manager")
        f = self._fh

        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        if "title" in scan:
            del scan["title"]
        scan.create_dataset("title", data=title)
        if "start_time" in scan:
            del scan["start_time"]
        scan.create_dataset("start_time", data=start_time)
        if "end_time" in scan:
            del scan["end_time"]
        scan.create_dataset("end_time", data=end_time)

        # Attrs (scan_mode / scanned_axes / crystal_mode | identify_mode)
        for k, v in attrs.items():
            if isinstance(v, list):
                scan.attrs[k] = list(v)
            else:
                scan.attrs[k] = v

        # /N.1/sample/
        samp = scan.require_group("sample")
        _set_nx_class(samp, "NXsample")
        _write_sample_dict(samp, sample)

        # /N.1/instrument/<detector_name>/data → ExternalLink
        instr = scan.require_group("instrument")
        _set_nx_class(instr, "NXinstrument")
        meas = scan.require_group("measurement")
        _set_nx_class(meas, "NXcollection")
        for det_name, (rel_path, internal_path) in detector_links.items():
            det = instr.require_group(det_name)
            _set_nx_class(det, "NXdetector")
            if "data" in det:
                del det["data"]
            det["data"] = h5py.ExternalLink(str(rel_path).replace("\\", "/"), internal_path)
            if det_name in meas:
                del meas[det_name]
            meas[det_name] = h5py.SoftLink(f"/{scan_id}/instrument/{det_name}/data")

        # /N.1/instrument/positioners/
        pos = instr.require_group("positioners")
        _set_nx_class(pos, "NXcollection")
        for axis_name, val in positioners.items():
            if axis_name in pos:
                del pos[axis_name]
            if isinstance(val, np.ndarray):
                ds = pos.create_dataset(axis_name, data=np.degrees(val))
            else:
                ds = pos.create_dataset(axis_name, data=float(np.degrees(val)))
            ds.attrs["units"] = "degree"
            if axis_name in meas:
                del meas[axis_name]
            meas[axis_name] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/{axis_name}")

        # /N.1/dfxm_geo/
        if dfxm_geo:
            d = scan.require_group("dfxm_geo")
            # Use a distinct loop var (g_val) so mypy does not unify with
            # the earlier `val: np.ndarray | float` in the positioners loop.
            for key, g_val in dfxm_geo.items():
                if g_val is None:
                    continue
                if key in d:
                    del d[key]
                if isinstance(g_val, (int, float)):
                    d.create_dataset(key, data=float(g_val))
                else:
                    d.create_dataset(key, data=g_val)


def _write_sample_dict(group: h5py.Group, sample: dict[str, object]) -> None:
    """Write a sample dict into an NXsample group.

    Scalars and arrays become datasets; nested dicts become sub-groups.
    Only the key ``"dislocations"`` is special-cased: it becomes an
    NXcollection whose children are NXsample sub-groups (one per
    dislocation index). Every other nested dict becomes a plain NXsample
    sub-group, with its contents recursed into.
    """
    for key, val in sample.items():
        if key in group:
            del group[key]
        if isinstance(val, dict):
            sub = group.create_group(key)
            if key == "dislocations":
                _set_nx_class(sub, "NXcollection")
                for idx_key, sub_sample in val.items():
                    item = sub.create_group(str(idx_key))
                    _set_nx_class(item, "NXsample")
                    _write_sample_dict(item, sub_sample)
            else:
                _set_nx_class(sub, "NXsample")
                _write_sample_dict(sub, val)
        elif isinstance(val, (int, float, str)):
            # h5py handles int/float/str natively; no explicit cast needed.
            group.create_dataset(key, data=val)
        else:
            group.create_dataset(key, data=np.asarray(val))
```

- [ ] **Step 2: Run tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_master_writer.py -v
```

Expected: 4 passed.

### Task 2.5: Verify suite + commit

- [ ] **Step 1: Full suite + mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: 437 passed / 4 skipped / 10 deselected / 1 xfailed (4 new MasterWriter tests); mypy 0 errors.

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_master_writer.py
git commit -m "io: add MasterWriter context manager for v1.2.0 master layout"
```

---

## Phase 3: Forward retrofit — `write_simulation_h5` rewrites to new layout

**Goal:** `pipeline.run_simulation` now produces `out_dir/dfxm_geo.h5` (master) + `out_dir/scan0001/dfxm_sim_detector_0000.h5` + (optionally) `out_dir/scan0002/...`. External-API signature unchanged; only the on-disk shape changes.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (rewrite `write_simulation_h5`, keep but deprecate `_save_scan_parallel_to_h5` and `write_h5_scan`)
- Modify: `src/dfxm_geo/pipeline.py` (no signature change to `run_simulation`)

### Task 3.1: Rewrite `write_simulation_h5` to use `MasterWriter`

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, replace the body of `write_simulation_h5` (lines ~387-518) with the MasterWriter-driven implementation:**

```python
def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    include_perfect_crystal: bool = True,
    sample_dis: float,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    crystal_mode: str | None = None,
    scan_mode: str | None = None,
    scanned_axes: list[str] | None = None,
) -> None:
    """One-call entry point for forward mode, v1.2.0 layout.

    `path` is the master file (`<out_dir>/dfxm_geo.h5`); detector pixels live
    in sibling `scan0001/` / `scan0002/` directories alongside it, linked
    via ExternalLink. External signature unchanged from v1.1.0.
    """
    import datetime as _dt2

    if kernel_npz is None:
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing HDF5 provenance."
            )

    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)
    n = phi_steps * chi_steps
    phi_per_frame = np.empty(n, dtype=np.float64)
    chi_per_frame = np.empty(n, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            k = chi_idx * phi_steps + phi_idx
            phi_per_frame[k] = Phi[phi_idx]
            chi_per_frame[k] = Chi[chi_idx]

    title = _scan_title(phi_range, phi_steps, chi_range, chi_steps)

    def _now() -> str:
        return _dt2.datetime.now(_dt2.UTC).isoformat(timespec="seconds")

    def _build_args(Hg_in: np.ndarray) -> list[tuple]:
        out = []
        for chi_idx in range(chi_steps):
            for phi_idx in range(phi_steps):
                k = chi_idx * phi_steps + phi_idx
                out.append((k, Hg_in, float(Phi[phi_idx]), float(Chi[chi_idx])))
        return out

    attrs_1_1: dict[str, str | list[str]] = {}
    if scan_mode is not None:
        attrs_1_1["scan_mode"] = scan_mode
    if scanned_axes is not None:
        attrs_1_1["scanned_axes"] = list(scanned_axes)
    if crystal_mode is not None:
        attrs_1_1["crystal_mode"] = crystal_mode

    with MasterWriter(
        path, cli=cli, config_toml=config_toml, kernel_npz=kernel_npz
    ) as master:
        # /1.1 dislocations
        scan1_dir = out_dir / SCAN_DIR_FMT.format(1)
        det1_path = scan1_dir / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
        start1 = _now()
        _compute_and_write_detector_file_parallel(
            det1_path, _build_args(Hg), max_workers=max_workers
        )
        end1 = _now()
        master.add_scan(
            scan_id="1.1",
            title=title,
            start_time=start1,
            end_time=end1,
            sample={
                "name": "simulated, dislocations",
                "dis": float(sample_dis),
                "ndis": int(sample_ndis),
                "sample_remount": sample_remount,
            },
            positioners={"phi": phi_per_frame, "chi": chi_per_frame},
            detector_links={
                "dfxm_sim_detector": (
                    Path(SCAN_DIR_FMT.format(1))
                    / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={
                "Hg": Hg,
                "q_hkl": q_hkl,
                "theta": float(_fm.theta),
                "psize": float(_fm.psize),
                "zl_rms": float(_fm.zl_rms),
            },
            attrs=attrs_1_1,
        )

        if include_perfect_crystal:
            scan2_dir = out_dir / SCAN_DIR_FMT.format(2)
            det2_path = scan2_dir / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
            Hg_zero = np.zeros_like(Hg)
            start2 = _now()
            _compute_and_write_detector_file_parallel(
                det2_path, _build_args(Hg_zero), max_workers=max_workers
            )
            end2 = _now()
            master.add_scan(
                scan_id="2.1",
                title=title,
                start_time=start2,
                end_time=end2,
                sample={
                    "name": "simulated, perfect crystal",
                    "dis": float(sample_dis),
                    "ndis": int(sample_ndis),
                    "sample_remount": sample_remount,
                },
                positioners={"phi": phi_per_frame, "chi": chi_per_frame},
                detector_links={
                    "dfxm_sim_detector": (
                        Path(SCAN_DIR_FMT.format(2))
                        / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                        DETECTOR_INTERNAL_PATH,
                    )
                },
                dfxm_geo={
                    "Hg": Hg_zero,
                    "q_hkl": q_hkl,
                    "theta": float(_fm.theta),
                    "psize": float(_fm.psize),
                    "zl_rms": float(_fm.zl_rms),
                },
                attrs=attrs_1_1,  # B+C followup: /2.1 also carries mode attrs
            )
```

- [ ] **Step 2: Remove the now-dead helpers** `_save_scan_parallel_to_h5` and `write_h5_scan` from `src/dfxm_geo/io/hdf5.py` (lines ~105-176 and ~230-332). Confirm no other module imports them:

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import ast, pathlib; [print(p) for p in pathlib.Path('src').rglob('*.py') if any(s in p.read_text() for s in ('_save_scan_parallel_to_h5', 'write_h5_scan'))]"
```

Expected after removal: prints only `src/dfxm_geo/io/migrate.py` (which still imports `write_h5_scan`). **Note:** migrate.py is rewritten in Phase 11 — for Phase 3, keep `write_h5_scan` in place so migrate.py keeps working. Delete `_save_scan_parallel_to_h5` only.

Concrete edit: delete only `_save_scan_parallel_to_h5` (lines ~105-176). Keep `write_h5_scan` until Phase 11.

### Task 3.2: Confirm `load_h5_scan` still works through ExternalLink (no code change needed)

- [ ] **Step 1: Write a test asserting load_h5_scan transparently follows the external link.** Create `tests/test_load_h5_scan_external_link.py`:

```python
"""Verify load_h5_scan reads through ExternalLink correctly (v1.2.0 layout)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
    load_h5_scan,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel


def test_load_h5_scan_follows_external_link(tmp_path: Path) -> None:
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    scan_dir = tmp_path / "scan0001"
    det_path = scan_dir / "dfxm_sim_detector_0000.h5"
    # 2 phi x 3 chi = 6 frames; need phi inner, chi outer ordering preserved
    h, w = 5, 5
    stack = np.arange(2 * 3 * h * w, dtype=np.float64).reshape(2 * 3, h, w)
    _write_detector_file(det_path, stack)

    master_path = tmp_path / "dfxm_geo.h5"
    cfg = "[scan.phi]\nrange = 0.01\nsteps = 2\n[scan.chi]\nrange = 0.01\nsteps = 3\n"
    with MasterWriter(master_path, cli="t", config_toml=cfg, kernel_npz=Path("placeholder.npz")) as m:
        # Use a temp kernel placeholder path; provenance recording will fail
        # if the path doesn't exist, so override _write_provenance kernel_npz to None:
        pass
    # The above 'with' fails on kernel_npz validation. Re-do without kernel for this test.
    master_path.unlink()
    with MasterWriter(master_path, cli="t", config_toml=cfg, kernel_npz=None) as m:
        m.add_scan(
            scan_id="1.1",
            title="t",
            start_time="t",
            end_time="t",
            sample={"name": "x"},
            positioners={"phi": np.zeros(6), "chi": np.zeros(6)},
            detector_links={
                "dfxm_sim_detector": (
                    Path("scan0001") / "dfxm_sim_detector_0000.h5",
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={},
            attrs={},
        )

    flat, reshape, dim_h, dim_w = load_h5_scan(
        master_path, scan_id="1.1", phi_steps=2, chi_steps=3
    )
    np.testing.assert_array_equal(flat, stack)
    assert dim_h == h and dim_w == w
```

- [ ] **Step 2: Run test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_load_h5_scan_external_link.py -v
```

Expected: PASS. (h5py follows ExternalLink transparently; the test just asserts that.)

### Task 3.3: Update `pipeline.run_simulation` — no signature change

- [ ] **Step 1: Verify `pipeline.run_simulation` already calls `write_simulation_h5(path, ...)` where `path = output_dir / "dfxm_geo.h5"`.** From the existing code at `src/dfxm_geo/pipeline.py:678-697`, the call site is correct. No edits needed — the new `write_simulation_h5` body uses `path.parent` as `out_dir` and creates `scan0001/` / `scan0002/` siblings automatically.

- [ ] **Step 2: Run the existing forward end-to-end smoke test to confirm:**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_hdf5_run_simulation_end_to_end.py -v
```

Expected: failures — existing tests assert the OLD single-file layout (image data directly under `/1.1/instrument/dfxm_sim_detector/data`). The reads STILL work (h5py follows the ExternalLink), but tests that check the on-disk shape (e.g. "scan dataset is in this one file") will fail. Note these failures — Phase 4 fixes them.

### Task 3.4: Commit Phase 3

- [ ] **Step 1: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_load_h5_scan_external_link.py
git commit -m "io: rewrite write_simulation_h5 for master + per-scan layout"
```

---

## Phase 4: Update existing forward tests to new layout

**Goal:** Fix the test suite so existing forward tests acknowledge the new on-disk shape. Read-path tests (those calling `load_h5_scan`) keep working unchanged; tests that probe the file layout directly need updates.

**Files:**
- Modify: `tests/test_hdf5_run_simulation_end_to_end.py`
- Modify: `tests/test_hdf5_pipeline.py`
- Modify: `tests/test_hdf5_provenance.py`
- Modify: `tests/test_hdf5_writer.py`
- Modify: `tests/test_hdf5_defensive.py`
- Modify: `tests/test_hdf5_reader.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_pipeline_scan_modes.py`
- Modify: `tests/test_pipeline_crystal_modes.py`
- Modify: `tests/test_pipeline_multi_reflection.py`

### Task 4.1: Survey which tests need updates

- [ ] **Step 1: Inventory failing tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q --no-header --tb=no | Select-String -Pattern "FAIL"
```

- [ ] **Step 2: For each failing test, classify:**
  - **Class A** — asserts pixel data via `load_h5_scan` or h5py read of `/N.1/.../data` (still works through link). Should NOT be failing. If it is, investigate.
  - **Class B** — asserts the master file contains the pixel dataset directly (e.g. `f["/1.1/.../data"].shape`). h5py auto-follows the link, so most such reads continue to work. Failing reads only happen if a test opens the master file and checks file size / lists all internal datasets without following links.
  - **Class C** — asserts a specific directory structure on disk (e.g. "only `dfxm_geo.h5` exists in `out_dir`"). These need updating to expect `scan0001/`, `scan0002/`.

### Task 4.2: Update Class C tests (directory-shape assertions)

For each Class C failure, find the `out_dir.iterdir()` or `os.listdir(out_dir)` style assertion and update to expect:

```python
expected = {"dfxm_geo.h5", "scan0001", "scan0002"}  # plus any sidecars
if not include_perfect_crystal:
    expected = {"dfxm_geo.h5", "scan0001"}
assert set(p.name for p in out_dir.iterdir()) >= expected
```

- [ ] **Step 1: For `tests/test_hdf5_run_simulation_end_to_end.py`,** locate any iterdir / file-existence checks and update them to include `scan0001/` (and `scan0002/` when applicable). Add a check that the detector file exists:

```python
assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
if config.io.include_perfect_crystal:
    assert (out_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
```

- [ ] **Step 2: Repeat for `test_hdf5_pipeline.py`, `test_pipeline.py`, `test_pipeline_scan_modes.py`, `test_pipeline_crystal_modes.py`, `test_pipeline_multi_reflection.py`.** Only update the directory-shape assertions; leave content reads via `load_h5_scan` untouched.

- [ ] **Step 3: Update `test_hdf5_provenance.py`** — provenance lives at `/dfxm_geo/` in the master, unchanged. Test should still pass; if it fails, it's because it opens the file and walks `f["/1.1/instrument/dfxm_sim_detector"]` listing members directly. Update such assertions to recognise that `data` is now an ExternalLink:

```python
# Was: assert "data" in det
# Now: data is an external link, but still resolves
assert "data" in det  # still True — h5py reports linked members
link = det.get("data", getlink=True)
assert isinstance(link, h5py.ExternalLink)
```

- [ ] **Step 4: Update `test_hdf5_writer.py` and `test_hdf5_defensive.py`** — these likely call `write_h5_scan` or `_save_scan_parallel_to_h5` directly. The latter was deleted; the former is preserved for migrate.py. If tests for `_save_scan_parallel_to_h5` exist, either:
  - Delete them (the function is gone), OR
  - Rewrite them as tests for `_compute_and_write_detector_file_parallel` (already covered in `test_detector_file.py`; safe to delete).

- [ ] **Step 5: Update `test_hdf5_reader.py`** — tests for `load_h5_scan` should still pass since reads transparent. If a test specifically constructs a fake `dfxm_geo.h5` with pixel data inline, update the fixture to either:
  - Build a master + scan dir using `MasterWriter` + `_write_detector_file`, OR
  - Use a single-file legacy fixture but call it explicitly v1.1 layout (only valid for the migrate test in Phase 11; otherwise drop).

### Task 4.3: Verify the suite is green

- [ ] **Step 1: Run full suite**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
```

Expected: green at the pre-Phase-3 baseline + new tests from Phases 1-2 + the new external-link test (438 passed / 4 skipped / 10 deselected / 1 xfailed).

- [ ] **Step 2: mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors.

### Task 4.4: Commit Phase 4

- [ ] **Step 1: Commit**

```powershell
git add tests/
git commit -m "tests: update forward fixtures for v1.2.0 master/per-scan layout"
```

---

## Phase 5: Config dataclass changes

**Goal:** Simplify `IdentificationZScanConfig` (drop duplicates of `[scan.<axis>]`), extend `IdentificationMonteCarloConfig` with `render_per_dislocation`. Update `load_identification_config` accordingly.

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Create: `tests/test_identification_config_changes.py`

### Task 5.1: Failing test — new config shape

- [ ] **Step 1: Create `tests/test_identification_config_changes.py`:**

```python
"""Verify v1.2.0 IdentificationZScanConfig / IdentificationMonteCarloConfig shape."""
from __future__ import annotations

import pytest

from dfxm_geo.pipeline import (
    IdentificationMonteCarloConfig,
    IdentificationZScanConfig,
)


def test_zscan_config_drops_phi_chi_duplicates() -> None:
    cfg = IdentificationZScanConfig(
        z_offsets_um=[-1.0, 0.0, 1.0],
        include_secondary=True,
        secondary_rng_offset=1,
    )
    # These fields must no longer exist on the dataclass:
    for stale in ("phi_range_deg", "phi_steps", "chi_range_deg", "chi_steps"):
        assert not hasattr(cfg, stale), f"{stale} should be removed in v1.2.0"


def test_zscan_config_rejects_old_kwargs() -> None:
    with pytest.raises(TypeError):
        IdentificationZScanConfig(  # type: ignore[call-arg]
            z_offsets_um=[0.0],
            phi_range_deg=0.034,  # removed
            phi_steps=21,
            chi_range_deg=0.114,
            chi_steps=21,
        )


def test_multi_config_adds_render_per_dislocation_default_false() -> None:
    cfg = IdentificationMonteCarloConfig(n_samples=10, pos_std_um=5.0)
    assert cfg.render_per_dislocation is False


def test_multi_config_render_per_dislocation_opt_in() -> None:
    cfg = IdentificationMonteCarloConfig(
        n_samples=10, pos_std_um=5.0, render_per_dislocation=True
    )
    assert cfg.render_per_dislocation is True


def test_multi_config_drops_n_png_previews() -> None:
    cfg = IdentificationMonteCarloConfig(n_samples=10, pos_std_um=5.0)
    assert not hasattr(cfg, "n_png_previews")
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_config_changes.py -v
```

Expected: 5 failures (config fields still present from v1.1).

### Task 5.2: Update `IdentificationZScanConfig`

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, replace `IdentificationZScanConfig` (lines ~447-463) with:**

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    """z-scan mode parameters (mode='z-scan' only).

    Each (z_layer, b, α) configuration produces a (phi_steps × chi_steps)
    rocking-curve stack on disk (driven by `config.scan.phi` / `config.scan.chi`
    from the shared B+C ScanConfig), with a randomly-drawn secondary
    dislocation if `include_secondary` is True. The secondary is drawn
    once per (z, b, α) and shared across the rocking grid.

    v1.2.0: the duplicate `phi_range_deg / phi_steps / chi_range_deg /
    chi_steps` fields have been removed; the scan grid is now read from
    `[scan.phi]` / `[scan.chi]` via the shared ScanConfig.
    """

    z_offsets_um: list[float]
    include_secondary: bool = True
    secondary_rng_offset: int = 1
```

### Task 5.3: Update `IdentificationMonteCarloConfig`

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, replace `IdentificationMonteCarloConfig` (lines ~438-445) with:**

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    """Multi-disloc Monte Carlo parameters (mode='multi' only).

    v1.2.0: `n_png_previews` removed (PNG sidecars dropped). New opt-in
    `render_per_dislocation`: when True, each scan dir also writes
    per-dislocation detector files for unambiguous instance labels.
    """

    n_samples: int = 1000
    pos_std_um: float = 5.0
    render_per_dislocation: bool = False
```

### Task 5.4: Update `load_identification_config` for the new shape

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py:541`, the line `zscan = IdentificationZScanConfig(**data["zscan"]) if data.get("zscan") is not None else None` will start raising `TypeError` for old configs (extra kwargs). Make the loader strict: pass through to `IdentificationZScanConfig` directly and let the `TypeError` propagate with a clear message:**

```python
    multi = (
        IdentificationMonteCarloConfig(**data["multi"])
        if data.get("multi") is not None
        else None
    )
    zscan = (
        IdentificationZScanConfig(**data["zscan"])
        if data.get("zscan") is not None
        else None
    )
```

(No structural change — only the dataclass shapes changed, so old-config users get a clear "unexpected keyword argument" TypeError pointing them at the migration.)

### Task 5.5: Run all tests

- [ ] **Step 1: Run config tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_config_changes.py -v
```

Expected: 5 passed.

- [ ] **Step 2: Run full suite — note expected failures**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
```

Expected: NEW failures in:
- `tests/test_pipeline_identification.py` — z-scan tests instantiate the old shape; will be rewritten in Phase 9.
- `tests/test_configs_load_under_new_schema.py` — `identification_zscan.toml` still has the old `[zscan]` fields; migrated in Phase 10.

Leave these failing for now; they're addressed by later phases.

### Task 5.6: Commit

- [ ] **Step 1: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/test_identification_config_changes.py
git commit -m "config: drop duplicate scan fields from zscan; add render_per_dislocation"
```

---

## Phase 6: `ScanSpec` + `write_identification_h5` orchestrator + eager guards

**Goal:** Define the shared `ScanSpec` data carrier and a `write_identification_h5(config, output_dir, scan_iter)` orchestrator that consumes a generator of `ScanSpec`s and writes the new layout. Add an eager `ValueError` guard in `run_identification` for unwired `[scan.two_dtheta]` / `[scan.z]`.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (add `ScanSpec`, `write_identification_h5`)
- Modify: `src/dfxm_geo/pipeline.py` (eager guards, dispatch via `write_identification_h5`)
- Create: `tests/test_write_identification_orchestrator.py`

### Task 6.1: Define `ScanSpec`

- [ ] **Step 1: At the top of `src/dfxm_geo/io/hdf5.py`, after the imports, add:**

```python
from dataclasses import dataclass, field as _dc_field


@dataclass(frozen=True)
class ScanSpec:
    """One BLISS scan worth of work, yielded by an identification runner.

    Consumed by `write_identification_h5`: the orchestrator turns each
    `ScanSpec` into one master `/N.1` entry plus one or more per-scan
    detector files (one per key in `detectors`).

    Attributes:
        title: Scan title string written to `/N.1/title`.
        sample: NXsample contents for this scan (see per-mode layouts in
            the design spec).
        positioners: motor-axis name → 1-D array (scanned) or scalar (fixed).
            ASSUMES input is in radians.
        dfxm_geo: sim-specific per-scan metadata (Hg, q_hkl, theta, psize,
            zl_rms). Any subset may be supplied.
        detectors: detector_name → list of frame args tuples
            `(frame_idx, Hg, phi, chi)` for the parallel writer. Each
            detector becomes its own LIMA-style file inside the scan dir.
        attrs: per-`/N.1` attrs — at minimum `scan_mode`, `scanned_axes`,
            `identify_mode`.
    """

    title: str
    sample: dict
    positioners: dict[str, np.ndarray | float]
    dfxm_geo: dict
    detectors: dict[str, list[tuple]]
    attrs: dict[str, str | list[str]]
```

### Task 6.2: Failing test for `write_identification_h5`

- [ ] **Step 1: Create `tests/test_write_identification_orchestrator.py`:**

```python
"""Smoke test for the write_identification_h5 orchestrator with a fake scan_iter."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MASTER_IDENTIFY,
    ScanSpec,
    write_identification_h5,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel
import dfxm_geo.direct_space.forward_model as fm


def _fake_scan_iter() -> Iterator[ScanSpec]:
    Hg = np.zeros((1, 3, 3))
    Hg[0] = np.eye(3)
    # Two scan entries, each with one detector and one frame.
    for k in range(2):
        yield ScanSpec(
            title=f"single scan {k}",
            sample={
                "name": "simulated, dislocation identification (single)",
                "slip_plane_normal": np.asarray([1, 1, 1], dtype=np.int32),
                "burgers": np.asarray([1, 0, 1], dtype=np.int32),
                "rotation_deg": float(k * 10),
            },
            positioners={"phi": 1.5e-4, "chi": 0.0},
            dfxm_geo={"Hg": Hg, "q_hkl": np.array([0.0, 0.0, 1.0])},
            detectors={"dfxm_sim_detector": [(0, Hg, 1.5e-4, 0.0)]},
            attrs={
                "scan_mode": "single",
                "scanned_axes": [],
                "identify_mode": "single",
            },
        )


def test_write_identification_h5_basic(tmp_path: Path) -> None:
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_identification_h5(
        out_dir,
        scan_iter=_fake_scan_iter(),
        cli="pytest test_write_identification_h5",
        config_toml="mode = \"single\"\n",
    )
    master = out_dir / MASTER_IDENTIFY
    assert master.is_file()
    assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (out_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
    with h5py.File(master, "r") as f:
        assert "/1.1" in f and "/2.1" in f
        assert f["/1.1"].attrs["identify_mode"] == "single"
        # External link resolves
        det = f["/1.1/instrument/dfxm_sim_detector/data"]
        assert det.shape[1:] == (fm.Npixels, fm.Npixels)
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_write_identification_orchestrator.py -v
```

Expected: ImportError on `write_identification_h5`.

### Task 6.3: Implement `write_identification_h5`

- [ ] **Step 1: In `src/dfxm_geo/io/hdf5.py`, after the `MasterWriter` class, add:**

```python
def write_identification_h5(
    output_dir: Path,
    *,
    scan_iter,  # Iterable[ScanSpec]
    cli: str,
    config_toml: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
) -> int:
    """Drive an identification run: consume ScanSpecs, write master + per-scan dirs.

    For each ScanSpec yielded:
      1. Create `output_dir/scanNNNN/`.
      2. For each `(detector_name, args_list)` in `spec.detectors`, write
         `output_dir/scanNNNN/<name>_0000.h5` via the parallel writer.
      3. Call `master.add_scan(scan_id=f"{N}.1", detector_links=..., ...)`.

    Returns the count of scans written.
    """
    import datetime as _dt2

    if kernel_npz is None:
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing identification HDF5 provenance."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / MASTER_IDENTIFY
    n_scans = 0

    def _now() -> str:
        return _dt2.datetime.now(_dt2.UTC).isoformat(timespec="seconds")

    with MasterWriter(
        master_path, cli=cli, config_toml=config_toml, kernel_npz=kernel_npz
    ) as master:
        for idx, spec in enumerate(scan_iter):
            scan_id = f"{idx + 1}.1"
            scan_dir_rel = Path(SCAN_DIR_FMT.format(idx + 1))
            scan_dir = output_dir / scan_dir_rel
            scan_dir.mkdir(parents=True, exist_ok=True)
            detector_links: dict[str, tuple[Path, str]] = {}
            start_time = _now()
            for det_name, args_list in spec.detectors.items():
                det_file = scan_dir / DETECTOR_FILE_FMT.format(name=det_name)
                _compute_and_write_detector_file_parallel(
                    det_file, args_list, max_workers=max_workers
                )
                detector_links[det_name] = (
                    scan_dir_rel / DETECTOR_FILE_FMT.format(name=det_name),
                    DETECTOR_INTERNAL_PATH,
                )
            end_time = _now()
            master.add_scan(
                scan_id=scan_id,
                title=spec.title,
                start_time=start_time,
                end_time=end_time,
                sample=spec.sample,
                positioners=spec.positioners,
                detector_links=detector_links,
                dfxm_geo=spec.dfxm_geo,
                attrs=spec.attrs,
            )
            n_scans += 1
    return n_scans
```

- [ ] **Step 2: Run test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_write_identification_orchestrator.py -v
```

Expected: 1 passed.

### Task 6.4: Add eager guard in `run_identification`

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py:run_identification` (line ~1382), add the two_dtheta/z guard right after the reciprocal None-check:**

```python
def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single / multi / z-scan runner based on config.mode."""
    if config.reciprocal is None:
        raise ValueError(
            "IdentificationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_identification."
        )
    # v1.2.0 scope: identify kernels only consume phi + chi. ScanGrid for
    # two_dtheta / z is implemented but not wired into the identify forward
    # path. Raise eagerly so users don't get silently-wrong output.
    if config.scan.two_dtheta.is_scanned or config.scan.z.is_scanned:
        unwired = [
            axis for axis in ("two_dtheta", "z") if config.scan.is_scanned(axis)
        ]
        raise ValueError(
            f"scan axes {unwired} are configured but not yet wired into "
            f"identification (v1.2.0 scope). For now, set range+steps only on "
            f"[scan.phi] and/or [scan.chi]."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)

    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    if config.mode == "multi":
        return _run_identification_multi(config, output_dir)
    return _run_identification_zscan(config, output_dir)
```

### Task 6.5: Add guard test + commit

- [ ] **Step 1: Append to `tests/test_identification_config_changes.py`:**

```python
def test_run_identification_eager_guards_unwired_axes(tmp_path) -> None:
    from dataclasses import replace
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        IdentificationConfig,
        IdentificationCrystalConfig,
        IdentificationNoiseConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
        run_identification,
    )

    scan = ScanConfig(
        phi=AxisScanConfig(value=1e-4),
        two_dtheta=AxisScanConfig(value=0.0, range=1e-3, steps=3),
    )
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        noise=IdentificationNoiseConfig(),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    with pytest.raises(ValueError, match=r"two_dtheta"):
        run_identification(cfg, tmp_path)
```

- [ ] **Step 2: Run new test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_config_changes.py::test_run_identification_eager_guards_unwired_axes -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py src/dfxm_geo/pipeline.py tests/test_write_identification_orchestrator.py tests/test_identification_config_changes.py
git commit -m "io: add ScanSpec + write_identification_h5 orchestrator; guard unwired axes"
```

---

## Phase 7: `_run_identification_single` as a generator (sidecar drop)

**Goal:** Convert the single-mode runner from a write-to-disk loop into a `ScanSpec` generator, drop `manifest.csv` / `images/*.png` / `im_data/*.npy` sidecars, and support `[scan.phi]/[scan.chi]` grids.

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (rewrite single tests for HDF5)
- Create: `tests/test_pipeline_identification_hdf5.py`

### Task 7.1: Failing test — single mode writes new layout

- [ ] **Step 1: Create `tests/test_pipeline_identification_hdf5.py`:**

```python
"""End-to-end identify → HDF5 tests (single, multi, z-scan)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import (
    AxisScanConfig,
    IOConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationNoiseConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def _minimal_single_cfg() -> IdentificationConfig:
    return IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=10.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )


def test_single_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    cfg = _minimal_single_cfg()
    run_identification(cfg, tmp_path)

    master = tmp_path / "dfxm_identify.h5"
    assert master.is_file()
    # 1 plane × 1 b × 2 angles = 2 scans
    assert (tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (tmp_path / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()

    with h5py.File(master, "r") as f:
        assert sorted(k for k in f if k != "dfxm_geo") == ["1.1", "2.1"]
        for sid in ("1.1", "2.1"):
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "single"
            assert scan.attrs["scan_mode"] == "single"
            assert list(scan.attrs["scanned_axes"]) == []
            samp = scan["sample"]
            assert samp["name"][()].decode().startswith(
                "simulated, dislocation identification (single)"
            )
            assert "slip_plane_normal" in samp
            assert "burgers" in samp
            assert "rotation_deg" in samp
        # Each /N.1 has 1 frame per detector
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 1


def test_single_mode_drops_legacy_sidecars(tmp_path: Path) -> None:
    cfg = _minimal_single_cfg()
    run_identification(cfg, tmp_path)
    assert not (tmp_path / "manifest.csv").exists()
    # No per-plane subdirs left over from the old .npy layout
    for plane_dirname in ("n_1_1_1", "n_1_m1_1", "n_1_1_m1", "n_m1_1_1"):
        assert not (tmp_path / plane_dirname).exists()
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification_hdf5.py -v
```

Expected: fails because `_run_identification_single` still writes the old `.npy + manifest` layout.

### Task 7.2: Rewrite `_run_identification_single` as a generator

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, replace `_run_identification_single` (lines ~980-1087) with the generator form and a thin dispatch wrapper:**

```python
def _iter_identification_single(
    config: IdentificationConfig,
) -> Iterator["ScanSpec"]:
    """Yield one ScanSpec per (plane, b_idx, alpha) configuration.

    Supports `[scan.phi]` / `[scan.chi]` from the shared ScanConfig: when
    either axis is scanned, each scan dir contains a (N_frames, H, W)
    stack with frame ordering phi-inner, chi-outer.
    """
    from dfxm_geo.io.hdf5 import ScanSpec  # local import to avoid cycle

    crystal_cfg = config.crystal
    all_planes: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, 1, 1),
    ]
    planes = (
        all_planes if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )

    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for plane in planes:
        b_table = _burgers_vectors(plane)
        b_indices = (
            crystal_cfg.b_vector_indices
            if crystal_cfg.b_vector_indices is not None
            else list(range(len(b_table)))
        )
        b_subset = b_table[b_indices]
        n_arr_unnorm = np.asarray(plane, dtype=float)
        n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)
        rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
        Ud_all = _ud_matrices(n_arr, rotated)

        for j, b_idx in enumerate(b_indices):
            if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
            ):
                continue
            for i, alpha in enumerate(angles_deg):
                Ud_mix = Ud_all[i, j]
                Fg = Fd_find_mixed(
                    fm.rl,
                    fm.Us,
                    Ud_mix=Ud_mix,
                    rotation_deg=float(alpha),
                    Theta=fm.Theta,
                )
                Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                fm.Hg = Hg
                fm.q_hkl = q_hkl

                # Build args list for this scan's frames.
                args_list, phi_pf, chi_pf = _scan_frames_args(Hg, Phi, Chi)

                burgers_int = (
                    int(round(b_table[b_idx, 0] * np.sqrt(2))),
                    int(round(b_table[b_idx, 1] * np.sqrt(2))),
                    int(round(b_table[b_idx, 2] * np.sqrt(2))),
                )
                yield ScanSpec(
                    title=_identify_title(scan_mode, n_frames, config.scan),
                    sample={
                        "name": "simulated, dislocation identification (single)",
                        "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                        "burgers": np.asarray(burgers_int, dtype=np.int32),
                        "rotation_deg": float(alpha),
                    },
                    positioners=_positioners_for_scan(phi_pf, chi_pf, config.scan),
                    dfxm_geo={
                        "Hg": Hg,
                        "q_hkl": q_hkl,
                        "theta": float(fm.theta),
                        "psize": float(fm.psize),
                        "zl_rms": float(fm.zl_rms),
                    },
                    detectors={"dfxm_sim_detector": args_list},
                    attrs={
                        "scan_mode": scan_mode,
                        "scanned_axes": scanned_axes,
                        "identify_mode": "single",
                    },
                )


def _run_identification_single(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatcher: feed `_iter_identification_single` into write_identification_h5."""
    from dfxm_geo.io.hdf5 import write_identification_h5

    output_dir.mkdir(parents=True, exist_ok=True)
    config_toml = _identification_config_to_toml_str(config)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_single(config),
        cli=" ".join(sys.argv),
        config_toml=config_toml,
        max_workers=config.io.max_workers,
    )
    return {
        "n_images": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }
```

- [ ] **Step 2: Add the small helpers used above. Append near the top of the identification helpers section in `pipeline.py` (around line ~945, just after `_slip_plane_slug`):**

```python
def _frame_grid_from_scan(
    scan: ScanConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (Phi_rad, Chi_rad, n_frames) for a single-scan rocking grid.

    Both arrays are 1-D of length n_frames, in phi-inner / chi-outer order.
    Fixed axes contribute a single repeated value.
    """
    if scan.phi.is_scanned:
        assert scan.phi.range is not None and scan.phi.steps is not None
        Phi = np.linspace(
            -np.deg2rad(scan.phi.range), np.deg2rad(scan.phi.range), scan.phi.steps
        )
    else:
        Phi = np.asarray([scan.phi.value], dtype=float)
    if scan.chi.is_scanned:
        assert scan.chi.range is not None and scan.chi.steps is not None
        Chi = np.linspace(
            -np.deg2rad(scan.chi.range), np.deg2rad(scan.chi.range), scan.chi.steps
        )
    else:
        Chi = np.asarray([scan.chi.value], dtype=float)
    return Phi, Chi, Phi.size * Chi.size


def _scan_frames_args(
    Hg: np.ndarray, Phi: np.ndarray, Chi: np.ndarray
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Build (args_list, phi_per_frame, chi_per_frame) for one ScanSpec.

    Frame order: phi-inner, chi-outer (matches forward fscan2d convention).
    """
    n = Phi.size * Chi.size
    args_list: list[tuple] = []
    phi_pf = np.empty(n, dtype=np.float64)
    chi_pf = np.empty(n, dtype=np.float64)
    for chi_idx in range(Chi.size):
        for phi_idx in range(Phi.size):
            k = chi_idx * Phi.size + phi_idx
            phi_pf[k] = float(Phi[phi_idx])
            chi_pf[k] = float(Chi[chi_idx])
            args_list.append((k, Hg, float(Phi[phi_idx]), float(Chi[chi_idx])))
    return args_list, phi_pf, chi_pf


def _positioners_for_scan(
    phi_pf: np.ndarray, chi_pf: np.ndarray, scan: ScanConfig
) -> dict[str, np.ndarray | float]:
    """Return phi/chi entries for ScanSpec.positioners.

    Fixed axes collapse to a scalar in radians; scanned axes are the
    full (N_frames,) array.
    """
    out: dict[str, np.ndarray | float] = {}
    out["phi"] = phi_pf if scan.phi.is_scanned else float(scan.phi.value)
    out["chi"] = chi_pf if scan.chi.is_scanned else float(scan.chi.value)
    return out


def _identify_title(
    scan_mode: str, n_frames: int, scan: ScanConfig
) -> str:
    """Compact human title for /N.1/title in identification masters."""
    return f"identify-{scan_mode} N_frames={n_frames}"


def _identification_config_to_toml_str(cfg: IdentificationConfig) -> str:
    """Best-effort TOML render of an IdentificationConfig (for /dfxm_geo/config_toml).

    Not round-trip-perfect; the goal is provenance, not reconstruction.
    Captures mode, reciprocal, scan, crystal (identification), noise, multi, zscan.
    """
    from dataclasses import asdict as _asdict

    lines: list[str] = [f'mode = "{cfg.mode}"']
    if cfg.reciprocal is not None:
        h, k, l = cfg.reciprocal.hkl
        lines += ["", "[reciprocal]", f"hkl = [{h}, {k}, {l}]", f"keV = {cfg.reciprocal.keV}"]
    # [crystal] (identification flavor; not the SimulationConfig crystal)
    c = cfg.crystal
    lines += [
        "",
        "[crystal]",
        f"slip_plane_normal = [{c.slip_plane_normal[0]}, "
        f"{c.slip_plane_normal[1]}, {c.slip_plane_normal[2]}]",
        f"angle_start_deg = {c.angle_start_deg}",
        f"angle_stop_deg = {c.angle_stop_deg}",
        f"angle_step_deg = {c.angle_step_deg}",
        f"sweep_all_slip_planes = {str(c.sweep_all_slip_planes).lower()}",
        f"exclude_invisibility = {str(c.exclude_invisibility).lower()}",
        f"invisibility_threshold_deg = {c.invisibility_threshold_deg}",
    ]
    if c.b_vector_indices is not None:
        lines.append(f"b_vector_indices = {list(c.b_vector_indices)}")
    # [scan.<axis>]
    for axis_name in _CANONICAL_AXES:
        axis = getattr(cfg.scan, axis_name)
        if axis.value == 0.0 and not axis.is_scanned:
            continue
        lines += ["", f"[scan.{axis_name}]"]
        if axis.value != 0.0:
            lines.append(f"value = {axis.value}")
        if axis.is_scanned:
            lines.append(f"range = {axis.range}")
            lines.append(f"steps = {axis.steps}")
    # [noise]
    lines += [
        "",
        "[noise]",
        f"poisson_noise = {str(cfg.noise.poisson_noise).lower()}",
        f"rng_seed = {cfg.noise.rng_seed}",
        f"intensity_scale = {cfg.noise.intensity_scale}",
    ]
    # [multi]
    if cfg.multi is not None:
        lines += [
            "",
            "[multi]",
            f"n_samples = {cfg.multi.n_samples}",
            f"pos_std_um = {cfg.multi.pos_std_um}",
            f"render_per_dislocation = {str(cfg.multi.render_per_dislocation).lower()}",
        ]
    # [zscan]
    if cfg.zscan is not None:
        lines += [
            "",
            "[zscan]",
            f"z_offsets_um = {list(cfg.zscan.z_offsets_um)}",
            f"include_secondary = {str(cfg.zscan.include_secondary).lower()}",
            f"secondary_rng_offset = {cfg.zscan.secondary_rng_offset}",
        ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 3: Add `from typing import Iterator` to the top imports in `pipeline.py`.**

```python
from typing import Any, Iterator, Literal
```

- [ ] **Step 4: Delete the now-dead `_save_preview_png` function** (`pipeline.py` ~lines 950-962) and remove the `import csv` (line 18) since manifest emission is dropped.

- [ ] **Step 5: Run identification HDF5 tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification_hdf5.py -v
```

Expected: 2 passed.

### Task 7.3: Rewrite the old `tests/test_pipeline_identification.py` single-mode tests

- [ ] **Step 1: Open `tests/test_pipeline_identification.py`.** For each test in single-mode that asserts the `.npy + manifest.csv` layout, rewrite it to assert the HDF5 layout (master + `scanNNNN/`). Tests that already cover behavior (sweep coverage, invisibility filter, b_vector_indices subset, etc.) should be redirected to count `/N.1` entries in the master instead of `.npy` files in `im_data/`.

Replace any check like:

```python
manifest = output_dir / "manifest.csv"
assert manifest.is_file()
n_images = sum(1 for _ in csv.DictReader(manifest.open()))
```

with:

```python
master = output_dir / "dfxm_identify.h5"
assert master.is_file()
with h5py.File(master, "r") as f:
    n_scans = sum(1 for k in f if k != "dfxm_geo")
```

- [ ] **Step 2: Run the rewritten tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification.py -v -k "single"
```

Expected: green (subject to your rewrite quality — if anything's red, fix it).

### Task 7.4: Commit Phase 7

- [ ] **Step 1: Run full suite + mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: multi-mode + z-scan-mode tests in `test_pipeline_identification.py` still red (rewritten in Phases 8-9); everything else green.

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification_hdf5.py tests/test_pipeline_identification.py
git commit -m "identify: single-mode runner yields ScanSpec to write_identification_h5"
```

---

## Phase 8: `_run_identification_multi` as a generator + `render_per_dislocation`

**Goal:** Convert multi-mode to the generator pattern. Default (`render_per_dislocation=False`) produces one detector file per MC sample. Opt-in produces three detector files (combined + per-dis). Noise behavior: combined receives Poisson; per-dis files are noiseless.

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (rewrite multi-mode tests)
- Create: `tests/test_identification_multi_per_dis.py`

### Task 8.1: Failing test — multi mode (default) writes HDF5

- [ ] **Step 1: Append to `tests/test_pipeline_identification_hdf5.py`:**

```python
def test_multi_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=__import__(
            "dfxm_geo.pipeline", fromlist=["IdentificationMonteCarloConfig"]
        ).IdentificationMonteCarloConfig(n_samples=3, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    assert (tmp_path / "dfxm_identify.h5").is_file()
    for k in range(1, 4):
        assert (tmp_path / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5").is_file()
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = [k for k in f if k != "dfxm_geo"]
        assert sorted(scan_keys) == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "multi"
            samp = scan["sample"]
            assert "dislocations" in samp
            disl = samp["dislocations"]
            assert disl.attrs["NX_class"] == "NXcollection"
            assert sorted(disl) == ["0", "1"]
            for idx in ("0", "1"):
                d = disl[idx]
                assert d.attrs["NX_class"] == "NXsample"
                assert "slip_plane_normal" in d
                assert "burgers" in d
                assert "rotation_deg" in d
                assert "position_um" in d
```

- [ ] **Step 2: Run to confirm fail.** The current multi runner still emits `.npy + manifest`.

### Task 8.2: Rewrite `_run_identification_multi` as a generator

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, replace `_run_identification_multi` (lines ~1122-1204) with:**

```python
def _build_dislocation_sample_entry(d: dict[str, Any]) -> dict[str, Any]:
    """Convert _draw_dislocation output to NXsample-shaped dict."""
    return {
        "slip_plane_normal": np.asarray(d["plane"], dtype=np.int32),
        "burgers": np.asarray(
            [int(round(c)) for c in d["b_vec"]], dtype=np.int32
        ),
        "rotation_deg": float(d["alpha_deg"]),
        "position_um": np.asarray(d["pos_um"], dtype=float),
    }


def _iter_identification_multi(
    config: IdentificationConfig,
) -> Iterator["ScanSpec"]:
    """Yield one ScanSpec per MC sample.

    `render_per_dislocation=False` (default): single detector with the
    sum of the two dislocations. `=True`: three detectors per scan
    (combined + dis0-only + dis1-only). Per-dis detectors are NOISELESS.
    """
    from dfxm_geo.io.hdf5 import ScanSpec

    assert config.multi is not None
    mc = config.multi
    noise_cfg = config.noise
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    fm.q_hkl = q_hkl

    master = np.random.default_rng(noise_cfg.rng_seed)
    param_rng, noise_rng = master.spawn(2)

    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for _ in range(mc.n_samples):
        d1 = _draw_dislocation(param_rng, mc.pos_std_um)
        d2 = _draw_dislocation(param_rng, mc.pos_std_um)

        # Combined-scene Hg
        specs = [
            MixedDislocSpec(
                Ud_mix=d1["Ud"],
                rotation_deg=d1["alpha_deg"],
                position_lab_um=d1["pos_um"],
            ),
            MixedDislocSpec(
                Ud_mix=d2["Ud"],
                rotation_deg=d2["alpha_deg"],
                position_lab_um=d2["pos_um"],
            ),
        ]
        Fg_combined = Fd_find_multi_dislocs_mixed(fm.rl, fm.Us, specs, fm.Theta)
        Hg_combined = (
            np.transpose(fast_inverse2(Fg_combined), [0, 2, 1]) - np.identity(3)
        )

        # Combined-detector args; this is the "real" measurement (with noise).
        # We render frames via the parallel writer which calls forward() —
        # there is no clean hook to inject Poisson noise inside the worker.
        # Two-pass plan: (1) the parallel writer renders the combined NOISELESS
        # stack; (2) we patch in a Poisson draw afterward if requested.
        combined_args, phi_pf, chi_pf = _scan_frames_args(Hg_combined, Phi, Chi)

        # For "noise=True", we apply Poisson noise to the combined stack only.
        # Mechanism: a wrapper detector name "dfxm_sim_detector__poisson__"
        # signals the runner to post-process the file after writing. This is
        # implemented in Phase 8.3 via a sentinel-key escape hatch in the
        # orchestrator. For now, accept that Phase 8.2 is noiseless-only;
        # noise integration is Phase 8.3.

        detectors: dict[str, list[tuple]] = {"dfxm_sim_detector": combined_args}

        sample = {
            "name": "simulated, dislocation identification (multi)",
            "dislocations": {
                "0": _build_dislocation_sample_entry(d1),
                "1": _build_dislocation_sample_entry(d2),
            },
        }

        if mc.render_per_dislocation:
            # Per-dislocation Hg: each dislocation rendered alone (other one
            # zero-strain). Noiseless by design (ground-truth labels).
            Fg_dis0 = Fd_find_mixed(
                fm.rl,
                fm.Us,
                Ud_mix=d1["Ud"],
                rotation_deg=d1["alpha_deg"],
                Theta=fm.Theta,
            )
            Hg_dis0 = (
                np.transpose(fast_inverse2(Fg_dis0), [0, 2, 1]) - np.identity(3)
            )
            Fg_dis1 = Fd_find_mixed(
                fm.rl,
                fm.Us,
                Ud_mix=d2["Ud"],
                rotation_deg=d2["alpha_deg"],
                Theta=fm.Theta,
            )
            Hg_dis1 = (
                np.transpose(fast_inverse2(Fg_dis1), [0, 2, 1]) - np.identity(3)
            )
            dis0_args, _, _ = _scan_frames_args(Hg_dis0, Phi, Chi)
            dis1_args, _, _ = _scan_frames_args(Hg_dis1, Phi, Chi)
            detectors["dfxm_sim_detector_dis0"] = dis0_args
            detectors["dfxm_sim_detector_dis1"] = dis1_args

        yield ScanSpec(
            title=_identify_title(scan_mode, n_frames, config.scan),
            sample=sample,
            positioners=_positioners_for_scan(phi_pf, chi_pf, config.scan),
            dfxm_geo={
                "Hg": Hg_combined,
                "q_hkl": q_hkl,
                "theta": float(fm.theta),
                "psize": float(fm.psize),
                "zl_rms": float(fm.zl_rms),
            },
            detectors=detectors,
            attrs={
                "scan_mode": scan_mode,
                "scanned_axes": scanned_axes,
                "identify_mode": "multi",
            },
        )


def _run_identification_multi(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    from dfxm_geo.io.hdf5 import write_identification_h5

    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_multi(config),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
    return {
        "n_samples": config.multi.n_samples if config.multi else 0,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }
```

### Task 8.3: Implement post-write Poisson noise application

The parallel writer renders noiseless frames; Poisson noise is applied as a second pass after each detector file is closed. The noise is applied only to the `dfxm_sim_detector` (combined) file; `_dis0` and `_dis1` files remain noiseless ground truth.

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, after `_run_identification_multi`, add:**

```python
def _maybe_apply_poisson_noise(
    config: IdentificationConfig, output_dir: Path, n_scans: int
) -> None:
    """Apply Poisson noise to combined-detector files (post-write).

    Per spec lockdown: combined detector receives Poisson noise (intensity-
    scaled); per-dislocation detectors (`*_dis0`, `*_dis1`) stay noiseless.
    Master HDF5 is not touched; only the per-scan detector files are
    modified in place.
    """
    import h5py

    noise_cfg = config.noise
    scale = noise_cfg.intensity_scale
    # Always scale intensity; only add Poisson if requested.
    rng = (
        np.random.default_rng(noise_cfg.rng_seed)
        .spawn(2)[1]  # second stream — matches param/noise split
        if noise_cfg.poisson_noise
        else None
    )
    for k in range(1, n_scans + 1):
        det_file = (
            output_dir
            / f"scan{k:04d}"
            / "dfxm_sim_detector_0000.h5"
        )
        if not det_file.is_file():
            continue
        with h5py.File(det_file, "a") as f:
            img = f["/entry_0000/dfxm_sim_detector/image"]
            arr = img[...] * scale
            if rng is not None:
                arr = rng.poisson(np.clip(arr, a_min=0.0, a_max=None)).astype(
                    float
                )
            img[...] = arr
```

- [ ] **Step 2: Apply the same intensity scaling for single-mode too.** In `_run_identification_single`, after the `write_identification_h5` call, add:

```python
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
```

(Same one-line addition.)

- [ ] **Step 3: Run multi smoke test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification_hdf5.py::test_multi_mode_writes_master_plus_scan_dirs -v
```

Expected: PASS.

### Task 8.4: Failing test — `render_per_dislocation=True`

- [ ] **Step 1: Create `tests/test_identification_multi_per_dis.py`:**

```python
"""Verify render_per_dislocation=true emits 3 detector files per scan."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.pipeline import (
    AxisScanConfig,
    IOConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def test_render_per_dislocation_writes_three_files(tmp_path: Path) -> None:
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(
            n_samples=2, pos_std_um=5.0, render_per_dislocation=True
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    for k in (1, 2):
        scan_dir = tmp_path / f"scan{k:04d}"
        assert (scan_dir / "dfxm_sim_detector_0000.h5").is_file()
        assert (scan_dir / "dfxm_sim_detector_dis0_0000.h5").is_file()
        assert (scan_dir / "dfxm_sim_detector_dis1_0000.h5").is_file()

    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        instr = f["/1.1/instrument"]
        for name in ("dfxm_sim_detector", "dfxm_sim_detector_dis0", "dfxm_sim_detector_dis1"):
            assert name in instr
            assert instr[name].attrs["NX_class"] == "NXdetector"


def test_per_dis_files_are_noiseless(tmp_path: Path) -> None:
    """With poisson_noise=True, dis0/dis1 stay deterministic (noiseless)."""
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=True, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(
            n_samples=1, pos_std_um=5.0, render_per_dislocation=True
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    # Re-run; per-dis files should be byte-identical (deterministic noiseless).
    out2 = tmp_path / "second"
    out2.mkdir()
    run_identification(cfg, out2)
    for f1, f2 in [
        ("dfxm_sim_detector_dis0_0000.h5", "dfxm_sim_detector_dis0_0000.h5"),
        ("dfxm_sim_detector_dis1_0000.h5", "dfxm_sim_detector_dis1_0000.h5"),
    ]:
        with h5py.File(tmp_path / "scan0001" / f1, "r") as a, h5py.File(
            out2 / "scan0001" / f2, "r"
        ) as b:
            np.testing.assert_array_equal(
                a["/entry_0000/dfxm_sim_detector/image"][...],
                b["/entry_0000/dfxm_sim_detector/image"][...],
            )
```

- [ ] **Step 2: Run the test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_multi_per_dis.py -v
```

Expected: PASS.

### Task 8.5: Commit Phase 8

- [ ] **Step 1: Run full suite + mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: z-scan tests still red (Phase 9); everything else green.

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification_hdf5.py tests/test_identification_multi_per_dis.py
git commit -m "identify: multi-mode runner + render_per_dislocation opt-in"
```

---

## Phase 9: `_run_identification_zscan` as a generator

**Goal:** Convert z-scan mode to the generator pattern; consume `config.scan.phi/chi` instead of `config.zscan.phi_*/chi_*` (already removed in Phase 5).

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (rewrite z-scan tests)
- Append to: `tests/test_pipeline_identification_hdf5.py`

### Task 9.1: Failing test — z-scan writes HDF5 layout

- [ ] **Step 1: Append to `tests/test_pipeline_identification_hdf5.py`:**

```python
def test_zscan_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    from dfxm_geo.pipeline import IdentificationZScanConfig

    cfg = IdentificationConfig(
        mode="z-scan",
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
            phi=AxisScanConfig(range=0.034377, steps=3),
            chi=AxisScanConfig(range=0.114, steps=3),
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[-1.0, 0.0, 1.0],
            include_secondary=False,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)

    # 3 z-layers × 1 plane × 1 b × 1 alpha = 3 configs
    master = tmp_path / "dfxm_identify.h5"
    assert master.is_file()
    for k in (1, 2, 3):
        assert (tmp_path / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5").is_file()

    with h5py.File(master, "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        assert scan_keys == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "z-scan"
            assert scan.attrs["scan_mode"] == "mosa"
            samp = scan["sample"]
            assert "z_offset_um" in samp
            assert "primary" in samp
            assert "secondary" not in samp  # include_secondary=False
        # Each /N.1 has phi_steps * chi_steps = 9 frames
        det = f["/1.1/instrument/dfxm_sim_detector/data"]
        assert det.shape[0] == 9
```

### Task 9.2: Rewrite `_run_identification_zscan` as a generator

- [ ] **Step 1: In `src/dfxm_geo/pipeline.py`, replace `_run_identification_zscan` (lines ~1207-1379) with:**

```python
def _iter_identification_zscan(
    config: IdentificationConfig,
) -> Iterator["ScanSpec"]:
    """Yield one ScanSpec per (z, plane, b, alpha) configuration.

    Phi/chi grid comes from `config.scan.phi / config.scan.chi` (B+C schema).
    Secondary dislocation is drawn once per config when include_secondary=True.
    """
    from dfxm_geo.io.hdf5 import ScanSpec

    assert config.zscan is not None
    zscan = config.zscan
    crystal_cfg = config.crystal
    noise_cfg = config.noise

    all_planes: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, 1, 1),
    ]
    planes = (
        all_planes if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )
    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    master_rng = np.random.default_rng(noise_cfg.rng_seed)
    spawned = master_rng.spawn(zscan.secondary_rng_offset + 1)
    secondary_rng = spawned[zscan.secondary_rng_offset]

    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())
    Phi, Chi, n_frames = _frame_grid_from_scan(config.scan)

    for z_off in zscan.z_offsets_um:
        rl_shifted = fm.Z_shift(z_off)
        for plane in planes:
            b_table = _burgers_vectors(plane)
            b_indices = (
                crystal_cfg.b_vector_indices
                if crystal_cfg.b_vector_indices is not None
                else list(range(len(b_table)))
            )
            b_subset = b_table[b_indices]
            n_arr_unnorm = np.asarray(plane, dtype=float)
            n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)
            rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
            Ud_all = _ud_matrices(n_arr, rotated)

            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                    q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_primary = Ud_all[i, j]
                    primary_spec = MixedDislocSpec(
                        Ud_mix=Ud_primary,
                        rotation_deg=float(alpha),
                        position_lab_um=(0.0, 0.0, 0.0),
                    )

                    sample: dict[str, Any] = {
                        "name": "simulated, dislocation identification (z-scan)",
                        "z_offset_um": float(z_off),
                        "primary": {
                            "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                            "burgers": np.asarray(
                                [
                                    int(round(b_table[b_idx, 0] * np.sqrt(2))),
                                    int(round(b_table[b_idx, 1] * np.sqrt(2))),
                                    int(round(b_table[b_idx, 2] * np.sqrt(2))),
                                ],
                                dtype=np.int32,
                            ),
                            "rotation_deg": float(alpha),
                            "position_um": np.asarray([0.0, 0.0, 0.0]),
                        },
                    }

                    if zscan.include_secondary:
                        sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)
                        secondary_spec = MixedDislocSpec(
                            Ud_mix=sec["Ud"],
                            rotation_deg=sec["alpha_deg"],
                            position_lab_um=sec["pos_um"],
                        )
                        Fg = Fd_find_multi_dislocs_mixed(
                            rl_shifted,
                            fm.Us,
                            [primary_spec, secondary_spec],
                            fm.Theta,
                        )
                        sample["secondary"] = _build_dislocation_sample_entry(sec)
                    else:
                        Fg = Fd_find_mixed(
                            rl_shifted,
                            fm.Us,
                            Ud_mix=Ud_primary,
                            rotation_deg=float(alpha),
                            Theta=fm.Theta,
                        )

                    Hg = (
                        np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                    )

                    args_list, phi_pf, chi_pf = _scan_frames_args(Hg, Phi, Chi)
                    yield ScanSpec(
                        title=_identify_title(scan_mode, n_frames, config.scan),
                        sample=sample,
                        positioners=_positioners_for_scan(
                            phi_pf, chi_pf, config.scan
                        ),
                        dfxm_geo={
                            "Hg": Hg,
                            "q_hkl": q_hkl,
                            "theta": float(fm.theta),
                            "psize": float(fm.psize),
                            "zl_rms": float(fm.zl_rms),
                        },
                        detectors={"dfxm_sim_detector": args_list},
                        attrs={
                            "scan_mode": scan_mode,
                            "scanned_axes": scanned_axes,
                            "identify_mode": "z-scan",
                        },
                    )


def _run_identification_zscan(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    from dfxm_geo.io.hdf5 import write_identification_h5

    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_zscan(config),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans)
    return {
        "n_configurations": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }
```

- [ ] **Step 2: Run z-scan smoke test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification_hdf5.py::test_zscan_mode_writes_master_plus_scan_dirs -v
```

Expected: PASS.

### Task 9.3: Update CLI return-value print in `cli_main_identify`

The dispatcher already returns dicts with `n_images`, `n_samples`, or `n_configurations`; the printing in `cli_main_identify` is unchanged.

- [ ] **Step 1: Sanity-check the print statements still work**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "from dfxm_geo.pipeline import cli_main_identify; print('OK')"
```

Expected: prints `OK`.

### Task 9.4: Rewrite z-scan and multi tests in `test_pipeline_identification.py`

- [ ] **Step 1: Open `tests/test_pipeline_identification.py`** and for each multi/z-scan test, replace the `.npy + manifest` assertions with HDF5 master assertions (same pattern as in Task 7.3).

- [ ] **Step 2: Run them**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_identification.py -v
```

Expected: all green.

### Task 9.5: Add `[scan.<axis>]` grid tests

- [ ] **Step 1: Create `tests/test_identification_scan_modes.py`:**

```python
"""Verify identify sub-modes consume [scan.phi] / [scan.chi] correctly."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.pipeline import (
    AxisScanConfig,
    IOConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def test_single_with_phi_scanned_produces_phi_steps_frames(tmp_path: Path) -> None:
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
        scan=ScanConfig(phi=AxisScanConfig(range=0.01, steps=4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 4
        assert f["/1.1"].attrs["scan_mode"] == "rocking"
        assert list(f["/1.1"].attrs["scanned_axes"]) == ["phi"]


def test_multi_with_phi_and_chi_scanned_produces_phi_x_chi_frames(
    tmp_path: Path,
) -> None:
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.01, steps=3),
            chi=AxisScanConfig(range=0.01, steps=2),
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=1, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 6
        assert f["/1.1"].attrs["scan_mode"] == "mosa"
        assert sorted(f["/1.1"].attrs["scanned_axes"]) == ["chi", "phi"]
        pos = f["/1.1/instrument/positioners"]
        assert pos["phi"].shape == (6,)
        assert pos["chi"].shape == (6,)
```

- [ ] **Step 2: Run it**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py -v
```

Expected: 2 passed.

### Task 9.6: Commit Phase 9

- [ ] **Step 1: Full suite + mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: green (only the not-yet-migrated `configs/identification_zscan.toml` test in `test_configs_load_under_new_schema.py` may still be red; fixed in Phase 10).

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py tests/test_pipeline_identification_hdf5.py tests/test_identification_scan_modes.py
git commit -m "identify: z-scan runner consumes [scan.<axis>]; add scan-mode tests"
```

---

## Phase 10: Migrate `configs/identification_zscan.toml`

**Goal:** Move the legacy `[zscan].phi_*/chi_*` block to `[scan.phi]/[scan.chi]` so the file loads under the new dataclass shape.

**Files:**
- Modify: `configs/identification_zscan.toml`

### Task 10.1: Edit the config

- [ ] **Step 1: Replace the contents of `configs/identification_zscan.toml` with:**

```toml
# dfxm-identify: z-scan mode (4D scan — depth × Burgers × angle × rocking-curve).
#
# Mirrors the ESRF_DTU branch's save_scan workflow: for each z layer, sweep
# (Burgers, line-direction angle), pair with a randomly-drawn secondary
# dislocation (seeded for reproducibility), then save a phi/chi rocking
# curve per configuration via the v1.2.0 master + per-scan HDF5 layout.
#
# Nsub = 1 is the codebase default (typical real-run setting). To reproduce
# Borgi 2024 (IUCrJ) publication-quality figures, flip Nsub = 1 -> 2 in
# src/dfxm_geo/direct_space/forward_model.py.
#
# v1.2.0 schema: phi_range_deg / phi_steps / chi_range_deg / chi_steps have
# moved from [zscan] into [scan.phi] / [scan.chi] (B+C shared schema).

mode = "z-scan"

[reciprocal]
hkl = [-1, 1, -1]   # Al 111 reflection
keV = 17.0

[crystal]
slip_plane_normal = [1, 1, 1]    # starting plane; sweep_all_slip_planes overrides
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

# Rocking-curve grid is now driven by the shared ScanConfig (B+C schema).
# Both values are in degrees; ranges below match dfxm-forward defaults.
[scan.phi]
range = 0.034377467707849395    # = 0.0006 rad
steps = 21                       # downscaled example; bump to 61 for a real run

[scan.chi]
range = 0.11459155902616465     # = 0.002 rad
steps = 21

[noise]
poisson_noise = false            # noise lives in the rocking-curve forward calls
rng_seed = 0
intensity_scale = 7.0

[zscan]
z_offsets_um = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 depth slices
include_secondary = true
secondary_rng_offset = 1

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

### Task 10.2: Run config-schema regression test

- [ ] **Step 1: Run**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_configs_load_under_new_schema.py -v
```

Expected: PASS for all three identify configs.

### Task 10.3: Commit Phase 10

- [ ] **Step 1: Commit**

```powershell
git add configs/identification_zscan.toml
git commit -m "configs: migrate identification_zscan.toml to v1.2.0 [scan.<axis>] schema"
```

---

## Phase 11: Migration tools — `dfxm-migrate-h5` + `dfxm-migrate-output` update

**Goal:** Add the new `dfxm-migrate-h5` CLI for v1.1.0 → v1.2.0 HDF5 conversion. Update the existing `dfxm-migrate-output` to emit the new layout. Both live in `src/dfxm_geo/io/migrate.py`.

**Files:**
- Modify: `src/dfxm_geo/io/migrate.py`
- Modify: `pyproject.toml` (add new entry point)
- Modify: `tests/test_migrate_output.py` (assert new layout)
- Create: `tests/test_migrate_h5.py`

### Task 11.1: Failing test for `dfxm-migrate-h5`

- [ ] **Step 1: Create `tests/test_migrate_h5.py`:**

```python
"""Round-trip test for dfxm-migrate-h5 (v1.1.0 single-file → v1.2.0 master+per-scan)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.migrate import migrate_h5_master_to_master
from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH


def _build_v110_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build a minimal v1.1.0 single-file dfxm_geo.h5 fixture in-place.

    Returns (dislocs_stack, perfect_stack) for later equality checks.
    """
    dislocs = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    perfect = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) * -1.0
    with h5py.File(path, "w") as f:
        # /dfxm_geo/ provenance
        g = f.create_group("dfxm_geo")
        g.create_dataset("cli", data="dfxm-forward (fixture)")
        g.create_dataset("version", data="1.1.0")
        g.create_dataset("config_toml", data="")
        # /1.1
        s1 = f.create_group("1.1")
        s1.attrs["NX_class"] = "NXentry"
        s1.create_dataset("title", data="fscan2d phi -0.01 0.01 2 chi 0 0 1 1.0")
        s1.create_dataset("start_time", data="2026-05-20T00:00:00")
        s1.create_dataset("end_time", data="2026-05-20T00:00:01")
        instr = s1.create_group("instrument")
        instr.attrs["NX_class"] = "NXinstrument"
        det = instr.create_group("dfxm_sim_detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("data", data=dislocs)
        pos = instr.create_group("positioners")
        pos.attrs["NX_class"] = "NXcollection"
        pos.create_dataset("phi", data=np.array([-0.01, 0.01]))
        pos["phi"].attrs["units"] = "degree"
        pos.create_dataset("chi", data=np.zeros(2))
        pos["chi"].attrs["units"] = "degree"
        samp = s1.create_group("sample")
        samp.attrs["NX_class"] = "NXsample"
        samp.create_dataset("name", data="simulated, dislocations")
        samp.create_dataset("dis", data=4.0)
        samp.create_dataset("ndis", data=151)
        samp.create_dataset("sample_remount", data="S1")
        d = s1.create_group("dfxm_geo")
        d.create_dataset("Hg", data=np.eye(3).reshape(1, 3, 3))
        d.create_dataset("q_hkl", data=np.array([0.0, 0.0, 1.0]))
        # /2.1 perfect crystal
        s2 = f.create_group("2.1")
        s2.attrs["NX_class"] = "NXentry"
        det2 = s2.create_group("instrument/dfxm_sim_detector")
        det2.attrs["NX_class"] = "NXdetector"
        det2.create_dataset("data", data=perfect)
    return dislocs, perfect


def test_migrate_h5_master_to_master_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "v110.h5"
    dst_dir = tmp_path / "v120"
    dislocs, perfect = _build_v110_fixture(src)
    migrate_h5_master_to_master(src, dst_dir)

    new_master = dst_dir / "dfxm_geo.h5"
    assert new_master.is_file()
    assert (dst_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (dst_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
    with h5py.File(new_master, "r") as f:
        # Pixels follow the ExternalLink to the new layout
        np.testing.assert_array_equal(
            f["/1.1/instrument/dfxm_sim_detector/data"][...], dislocs
        )
        np.testing.assert_array_equal(
            f["/2.1/instrument/dfxm_sim_detector/data"][...], perfect
        )
        # Provenance copied
        assert f["/dfxm_geo/version"][()].decode() == "1.1.0"
        # Sample, positioners, dfxm_geo nodes preserved
        assert "/1.1/sample/name" in f
        assert "/1.1/instrument/positioners/phi" in f
        assert "/1.1/dfxm_geo/Hg" in f
```

- [ ] **Step 2: Run to confirm fail**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_migrate_h5.py -v
```

Expected: ImportError on `migrate_h5_master_to_master`.

### Task 11.2: Implement `migrate_h5_master_to_master`

- [ ] **Step 1: In `src/dfxm_geo/io/migrate.py`, replace the `write_h5_scan` import with the v1.2.0 surface, and add the new function. After `migrate_npy_dir_to_h5`, append:**

```python
def migrate_h5_master_to_master(
    src: Path, dst_dir: Path
) -> None:
    """Convert a v1.1.0 single-file dfxm_geo.h5 to the v1.2.0 master+per-scan layout.

    Pixel data is moved to LIMA-style per-scan detector files under
    dst_dir/scan{N:04d}/dfxm_sim_detector_0000.h5; the new master at
    dst_dir/dfxm_geo.h5 ExternalLinks to them. All non-pixel-data nodes
    (/dfxm_geo/, /N.1/sample/, /N.1/instrument/positioners/, /N.1/dfxm_geo/,
    title/start_time/end_time) are copied losslessly.
    """
    from dfxm_geo.io.hdf5 import (
        DETECTOR_FILE_FMT,
        DETECTOR_INTERNAL_PATH,
        SCAN_DIR_FMT,
        _write_detector_file,
    )

    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_master = dst_dir / "dfxm_geo.h5"

    with h5py.File(src, "r") as fin, h5py.File(new_master, "w") as fout:
        # /dfxm_geo/ provenance — copy verbatim
        if "dfxm_geo" in fin:
            fin.copy("dfxm_geo", fout)

        # All /N.1 scan entries
        scan_ids = sorted(
            k for k in fin if k != "dfxm_geo" and "." in k
        )
        for idx, scan_id in enumerate(scan_ids, start=1):
            src_scan = fin[scan_id]
            new_scan = fout.create_group(scan_id)
            new_scan.attrs.update({k: v for k, v in src_scan.attrs.items()})
            # Copy title / start_time / end_time
            for name in ("title", "start_time", "end_time"):
                if name in src_scan:
                    fin.copy(f"{scan_id}/{name}", new_scan)
            # Copy sample/, dfxm_geo/ (and analysis/) subtrees
            for sub in ("sample", "dfxm_geo"):
                if sub in src_scan:
                    fin.copy(f"{scan_id}/{sub}", new_scan)
            # Copy positioners
            if "instrument/positioners" in src_scan:
                new_instr = new_scan.create_group("instrument")
                new_instr.attrs["NX_class"] = "NXinstrument"
                fin.copy(f"{scan_id}/instrument/positioners", new_instr)
            else:
                new_instr = new_scan.require_group("instrument")
                new_instr.attrs["NX_class"] = "NXinstrument"
            # Extract pixel data into a new per-scan detector file
            pix_path = f"{scan_id}/instrument/dfxm_sim_detector/data"
            if pix_path in fin:
                stack = fin[pix_path][...]
                scan_dir_rel = Path(SCAN_DIR_FMT.format(idx))
                det_file = dst_dir / scan_dir_rel / DETECTOR_FILE_FMT.format(
                    name="dfxm_sim_detector"
                )
                _write_detector_file(det_file, stack)
                # ExternalLink in new master
                new_det = new_instr.create_group("dfxm_sim_detector")
                new_det.attrs["NX_class"] = "NXdetector"
                new_det["data"] = h5py.ExternalLink(
                    str(scan_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")).replace(
                        "\\", "/"
                    ),
                    DETECTOR_INTERNAL_PATH,
                )
            # measurement softlinks
            meas = new_scan.create_group("measurement")
            meas.attrs["NX_class"] = "NXcollection"
            for det_name in new_instr:
                if det_name == "positioners":
                    continue
                meas[det_name] = h5py.SoftLink(
                    f"/{scan_id}/instrument/{det_name}/data"
                )
            if "instrument/positioners" in src_scan:
                for axis in src_scan["instrument/positioners"]:
                    meas[axis] = h5py.SoftLink(
                        f"/{scan_id}/instrument/positioners/{axis}"
                    )


def cli_main_h5_to_h5(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-migrate-h5`."""
    p = argparse.ArgumentParser(
        description="Convert v1.1.0 single-file dfxm_geo.h5 to v1.2.0 master+per-scan layout."
    )
    p.add_argument("input_h5", type=Path, help="v1.1.0 dfxm_geo.h5")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_h5>.v120/)",
    )
    args = p.parse_args(argv)
    dst = args.output or args.input_h5.with_suffix(args.input_h5.suffix + ".v120")
    migrate_h5_master_to_master(args.input_h5, dst)
    print(f"Wrote {dst}/")
    return 0
```

- [ ] **Step 2: At the top of `migrate.py`, update the rename `cli_main` → `cli_main_npy_to_h5` to match `pyproject.toml`'s new entry-point names:**

```python
# At end of file, after the existing cli_main definition (around line ~191):
def cli_main_npy_to_h5(argv: list[str] | None = None) -> int:
    """Renamed alias for cli_main; matches pyproject entry-point name."""
    return cli_main(argv)
```

(Keep `cli_main` for back-compat in interactive use.)

### Task 11.3: Update `pyproject.toml`

- [ ] **Step 1: Find the `[project.scripts]` block and add the new entry:**

```toml
[project.scripts]
dfxm-forward = "dfxm_geo.pipeline:cli_main"
dfxm-identify = "dfxm_geo.pipeline:cli_main_identify"
dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"
dfxm-migrate-output = "dfxm_geo.io.migrate:cli_main_npy_to_h5"
dfxm-migrate-h5 = "dfxm_geo.io.migrate:cli_main_h5_to_h5"
```

(Keep any existing entry-point lines; add only `dfxm-migrate-h5`. Verify `dfxm-migrate-output` already points to the right symbol — if it points at `cli_main`, rename it to `cli_main_npy_to_h5`.)

- [ ] **Step 2: Run the entry-point regression test**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pyproject_migrate_entry.py -v
```

Expected: PASS (after a re-install). Run `pip install -e .` if needed:

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pip install -e .
```

### Task 11.4: Update `migrate_npy_dir_to_h5` to emit new layout

- [ ] **Step 1: In `src/dfxm_geo/io/migrate.py`, replace `migrate_npy_dir_to_h5` (lines ~70-165) with a version that emits the new layout. The function reads legacy `.npy` stacks (unchanged) then writes via `MasterWriter` + `_write_detector_file`:**

```python
def migrate_npy_dir_to_h5(
    npy_dir: Path,
    h5_path: Path,
    *,
    phi_steps: int,
    chi_steps: int,
    phi_range_deg: float,
    chi_range_deg: float,
    dis: float,
    ndis: int,
    sample_remount: str,
    dislocs_dirname: str = "images10",
    perfect_dirname: str = "images10_perf_crystal",
) -> None:
    """Read legacy .npy stacks under `npy_dir` and write v1.2.0 master+per-scan."""
    from dfxm_geo.io.hdf5 import (
        DETECTOR_FILE_FMT,
        DETECTOR_INTERNAL_PATH,
        MasterWriter,
        SCAN_DIR_FMT,
        _scan_title,
        _write_detector_file,
    )

    S = SAMPLE_REMOUNT_OPTIONS[sample_remount]
    Hg, q_hkl = _fm.Find_Hg(
        dis, ndis, _fm.psize, _fm.zl_rms, S=S, remount_name=sample_remount
    )

    dislocs = _load_images_legacy(
        str(npy_dir / dislocs_dirname),
        u_steps=phi_steps,
        v_steps=chi_steps,
    )[0]

    perfect_path = npy_dir / perfect_dirname
    has_perfect = perfect_path.is_dir()
    perfect = (
        _load_images_legacy(
            str(perfect_path),
            u_steps=phi_steps,
            v_steps=chi_steps,
        )[0]
        if has_perfect
        else None
    )

    config_toml = (
        f'[crystal]\nmode = "wall"\n[crystal.wall]\ndis = {dis}\n'
        f'ndis = {ndis}\nsample_remount = "{sample_remount}"\n\n'
        f"[scan.phi]\nrange = {phi_range_deg}\nsteps = {phi_steps}\n\n"
        f"[scan.chi]\nrange = {chi_range_deg}\nsteps = {chi_steps}\n"
    )
    title = _scan_title(phi_range_deg, phi_steps, chi_range_deg, chi_steps)
    phi_pf = _phi_per_frame(phi_steps, chi_steps, phi_range_deg)
    chi_pf = _chi_per_frame(phi_steps, chi_steps, chi_range_deg)
    out_dir = h5_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel_npz = _fm._loaded_kernel_path
    if kernel_npz is None:
        raise RuntimeError(
            "no kernel loaded — migration requires a loaded kernel for provenance."
        )

    with MasterWriter(
        h5_path,
        cli="dfxm-migrate-output (legacy import)",
        config_toml=config_toml,
        kernel_npz=kernel_npz,
    ) as master:
        # /1.1
        scan1_dir_rel = Path(SCAN_DIR_FMT.format(1))
        det1_file = out_dir / scan1_dir_rel / DETECTOR_FILE_FMT.format(
            name="dfxm_sim_detector"
        )
        _write_detector_file(det1_file, dislocs)
        master.add_scan(
            scan_id="1.1",
            title=title,
            start_time="legacy",
            end_time="legacy",
            sample={
                "name": "simulated, dislocations",
                "dis": float(dis),
                "ndis": int(ndis),
                "sample_remount": sample_remount,
            },
            positioners={"phi": phi_pf, "chi": chi_pf},
            detector_links={
                "dfxm_sim_detector": (
                    scan1_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={
                "Hg": Hg,
                "q_hkl": q_hkl,
                "theta": float(_fm.theta),
                "psize": float(_fm.psize),
                "zl_rms": float(_fm.zl_rms),
            },
            attrs={},
        )
        if has_perfect and perfect is not None:
            scan2_dir_rel = Path(SCAN_DIR_FMT.format(2))
            det2_file = out_dir / scan2_dir_rel / DETECTOR_FILE_FMT.format(
                name="dfxm_sim_detector"
            )
            _write_detector_file(det2_file, perfect)
            master.add_scan(
                scan_id="2.1",
                title=title,
                start_time="legacy",
                end_time="legacy",
                sample={
                    "name": "simulated, perfect crystal",
                    "dis": float(dis),
                    "ndis": int(ndis),
                    "sample_remount": sample_remount,
                },
                positioners={"phi": phi_pf, "chi": chi_pf},
                detector_links={
                    "dfxm_sim_detector": (
                        scan2_dir_rel
                        / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                        DETECTOR_INTERNAL_PATH,
                    )
                },
                dfxm_geo={
                    "Hg": np.zeros_like(Hg),
                    "q_hkl": q_hkl,
                    "theta": float(_fm.theta),
                    "psize": float(_fm.psize),
                    "zl_rms": float(_fm.zl_rms),
                },
                attrs={},
            )
```

- [ ] **Step 2: Delete the now-unused `write_h5_scan` import** at the top of `migrate.py` and update the top-of-file import accordingly. Then also delete `write_h5_scan` from `src/dfxm_geo/io/hdf5.py` (it was kept alive only for migrate.py).

### Task 11.5: Update `test_migrate_output.py` to assert the new layout

- [ ] **Step 1: Open `tests/test_migrate_output.py`** and change every assertion that the output is a single `.h5` file with embedded pixel data to assert master + per-scan layout:

```python
out_master = out_dir / "dfxm_geo.h5"
assert out_master.is_file()
assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
if has_perfect_in_fixture:
    assert (out_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
```

- [ ] **Step 2: Run tests**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_migrate_output.py tests/test_migrate_h5.py -v
```

Expected: all green.

### Task 11.6: Commit Phase 11

- [ ] **Step 1: Full suite + mypy**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: all green; mypy 0 errors.

- [ ] **Step 2: Commit**

```powershell
git add src/dfxm_geo/io/migrate.py src/dfxm_geo/io/hdf5.py pyproject.toml tests/test_migrate_output.py tests/test_migrate_h5.py
git commit -m "io: dfxm-migrate-h5 + new-layout dfxm-migrate-output (v1.2.0 layout)"
```

---

## Phase 12: Docs — `output-format.md` rewrite + release notes

**Goal:** Replace `docs/output-format.md` with a fresh v1.2.0 description; add `docs/release-notes-1.2.0.md` covering A + D + B + C + E.

**Files:**
- Modify: `docs/output-format.md`
- Create: `docs/release-notes-1.2.0.md`

### Task 12.1: Rewrite `docs/output-format.md`

- [ ] **Step 1: Open `docs/output-format.md`** and replace its contents with the v1.2.0 layout description. Sections:

1. **Overview** — one paragraph explaining the master + per-scan-dirs design and why (BLISS fidelity).
2. **High-level structure** — the directory tree from the design spec (re-paste it).
3. **Master file (`dfxm_geo.h5` / `dfxm_identify.h5`)** — describe `/dfxm_geo/` provenance and `/N.1` BLISS scans.
4. **Per-scan LIMA-style files** — describe `scan0001/dfxm_sim_detector_0000.h5` with the `/entry_0000/` internal structure.
5. **Frame ordering** — fscan2d phi-inner / chi-outer convention; `k = chi_idx * phi_steps + phi_idx`.
6. **NX_class table** — quick reference of which groups carry which NX_class.
7. **Per-mode `sample/` layouts** — replicate from the design spec.
8. **Reading** — examples for `load_h5_scan`, h5py direct, darfix, darling.
9. **Compression** — chunks `(1, H, W)`, gzip-4, shuffle.
10. **Migration from v1.1.0** — link to `dfxm-migrate-h5` with example invocation.
11. **External link path caveat (symlinks)** — copy the spec's open-question paragraph: relative external links resolve from the master file's directory; if the user creates a symlink to the master, h5py follows the symlink and resolves links from the symlink target's directory.

Re-use full sections from `docs/superpowers/specs/2026-05-21-identification-hdf5-design.md` verbatim where they fit.

### Task 12.2: Create `docs/release-notes-1.2.0.md`

- [ ] **Step 1: Create `docs/release-notes-1.2.0.md`:**

```markdown
# Release notes — v1.2.0

**Release date:** TBD (release branch)
**Highlights:** Sub-projects A + B + C + D + E land together. Pipeline-features arc complete.

## Breaking changes

- **Output layout (forward + identify): single-file → master + per-scan-dirs.**
  v1.1.0 wrote one `dfxm_geo.h5` containing pixel data inline. v1.2.0 writes
  `dfxm_geo.h5` as a *master* that ExternalLinks to per-scan LIMA-style
  detector files in `scan0001/`, `scan0002/`, … subdirectories of the output
  directory. The shape on disk now mirrors an ESRF BLISS dataset.
  - **Migration tool:** `dfxm-migrate-h5 <v110_dfxm_geo.h5> [--output <dir>]`.
- **Identification mode: `.npy` + `manifest.csv` + `images/*.png` → HDF5.**
  All three sub-modes (`single`, `multi`, `z-scan`) write `dfxm_identify.h5`
  + per-scan detector files in the same shape as forward. Ground-truth labels
  live in `/N.1/sample/` (per-mode layout — see `docs/output-format.md`).
  Manifest/preview sidecars are dropped.
- **`IdentificationZScanConfig`: dropped `phi_range_deg / phi_steps /
  chi_range_deg / chi_steps`.** The rocking-curve grid is now read from the
  shared `[scan.phi]` / `[scan.chi]` (B+C schema). Update existing configs.
- **`IdentificationMonteCarloConfig`: dropped `n_png_previews`; added
  `render_per_dislocation: bool = False`.** When `true`, each multi-mode
  scan dir also writes per-dislocation detector files
  (`dfxm_sim_detector_dis0_0000.h5`, `..._dis1_0000.h5`) for unambiguous
  instance labels. Per-dis files are noiseless; the combined detector
  receives Poisson noise as before.

## New features (A + B + C + D, untagged on main since v1.1.0)

- **A — multi-reflection bootstrap + Bragg validity.** `[reciprocal]` block in
  TOML carries hkl + keV; kernel lookup validates Bragg-satisfiability before
  load.
- **D — multi-reflection kernel lookup in forward + identify.** Kernels
  bundle their `hkl`/`keV` metadata; runtime picks the matching one and
  loads it on the fly.
- **B — per-axis `[scan.<axis>]` schema + derived mode names.** `single`,
  `rocking`, `rolling`, `mosa`, etc. derived from which axes carry
  range+steps.
- **C — crystal layouts: discriminated union.** `[crystal] mode = "centered"
  | "wall" | "random_dislocations"` with matching sub-block.
- **E — identification → HDF5 (this release).**

## Out of scope / deferred (v1.3.0+)

- Wiring `[scan.two_dtheta]` / `[scan.z]` into forward + identification
  kernels (eager `ValueError` for now).
- z-scan mode consolidation into `single + [scan.z]`.
- Pixel-level segmentation masks for multi mode.
- `render_per_dislocation` analogue for z-scan's primary/secondary pair.
- `_SLIP_SYSTEM_111` table extension (6/12 → 12/12 FCC slip systems).
- `sample_dis = -1.0` sentinel cleanup (centered + random_dislocations).
- `/2.1` HDF5 attrs `scan_mode`/`scanned_axes`/`crystal_mode` (currently
  only `/1.1` carries them).

## Migration checklist for existing users

1. **HDF5 outputs from v1.1.0:** run `dfxm-migrate-h5 <old.h5>` to convert.
2. **TOML configs for z-scan mode:** move `[zscan].phi_*/chi_*` to
   `[scan.phi]/[scan.chi]`. See `configs/identification_zscan.toml`.
3. **TOML configs for multi mode:** remove `n_png_previews`. To enable
   per-dis rendering, add `render_per_dislocation = true`.
4. **Cluster bootstrap:** `git pull && pip install -e ".[dev]" &&
   dfxm-bootstrap --config configs/default.toml` to refresh the kernel
   with the v1.2.0 metadata.
```

### Task 12.3: Commit Phase 12

- [ ] **Step 1: Commit**

```powershell
git add docs/output-format.md docs/release-notes-1.2.0.md
git commit -m "docs: rewrite output-format.md for v1.2.0; add release notes"
```

---

## Phase 13: Update CLAUDE.md working notes

**Goal:** Refresh CLAUDE.md (one level up from the repo) for the post-E state.

**Files:**
- Modify: `C:\Users\borgi\Documents\GM-reworked\CLAUDE.md`

### Task 13.1: Update the pipeline-features arc table

- [ ] **Step 1: In `CLAUDE.md`, update the arc table (line ~85-91) to mark E as shipped and adjust dates/commits accordingly.** Replace the existing table with:

```markdown
| # | Sub-project | Status |
|---|---|---|
| **A** | Bootstrap multi-reflection + Bragg validity | ✅ SHIPPED (merge `dbc3ecc`) |
| **D** | Multi-reflection kernel lookup in forward + identify | ✅ SHIPPED (merge `327b766`) |
| **B + C** | Scan modes + crystal layouts | ✅ SHIPPED (merge `43059bc`) |
| **E** | Identification mode → HDF5 (BLISS schema) | ✅ SHIPPED (merge `<TBD>`, v1.2.0) |
| **F** | Default config flip to "simple" | Pending; v2.0.0 candidate |
```

(Fill in the actual merge SHA after the final commit.)

- [ ] **Step 2: Update the tag-chain reference (line ~52-58) to add v1.2.0.**

```markdown
- **Latest release tag**: `v1.2.0` (2026-05-XX, merge commit `<TBD>` —
  identification → HDF5 + master/per-scan layout; sub-projects A-E
  bundled). `main` HEAD matches.
  Tag chain: `v0.9.0` → `v1.0.2` → `v1.0.3` → `v1.1.0` → `v1.2.0`.
```

- [ ] **Step 3: Remove now-resolved items from "Sub-projects B + C follow-ups"** that were addressed in E (e.g. the `sample_dis = -1.0` sentinel mention can stay since E didn't fix it; `/2.1` HDF5 attrs likewise still open).

- [ ] **Step 4: Add a new entry to "Already resolved (no action needed)":**

```markdown
- Identification mode HDF5 + master/per-scan layout — shipped in v1.2.0
  (sub-project E). Forward mode retrofitted to the same layout.
```

### Task 13.2: Commit (NOT in repo — CLAUDE.md is private)

CLAUDE.md is private and lives outside the repo; no git operation here. Just save the file.

---

## Phase 14: Manual verification (silx / darfix / darling)

**Goal:** Confirm the v1.2.0 layout opens correctly in the same tooling that v1.1.0 was validated against.

**Files:** None (verification only).

### Task 14.1: Generate a small forward output for inspection

- [ ] **Step 1: Pick a quick config:**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.pipeline --config configs/default.toml --output C:\Users\borgi\tmp\v120_forward_smoke
```

Expected: produces `dfxm_geo.h5` + `scan0001/` + `scan0002/`.

### Task 14.2: silx view

- [ ] **Step 1: Open in silx**

```powershell
silx view C:\Users\borgi\tmp\v120_forward_smoke\dfxm_geo.h5
```

Verify the BLISS scan navigator shows `/1.1` and `/2.1`, that detector images render, and that the external link resolves transparently. Take a screenshot for the PR description if useful.

### Task 14.3: darfix

- [ ] **Step 1: In Python:**

```python
from darfix.dtypes import ImageDataset
from silx.io.url import DataUrl
ds = ImageDataset(
    [DataUrl(file_path="C:/Users/borgi/tmp/v120_forward_smoke/dfxm_geo.h5", data_path="/1.1/instrument/dfxm_sim_detector/data", scheme="silx")]
)
print(ds.data.shape)
```

Expected: a 3-D shape, e.g. `(N_frames, H, W)`.

### Task 14.4: darling

- [ ] **Step 1: In Python:**

```python
from darling.io import load
ds = load("C:/Users/borgi/tmp/v120_forward_smoke/dfxm_geo.h5", "1.1")
print(ds.shape)  # should auto-reshape to (phi_steps, chi_steps, H, W)
```

Expected: a 4-D shape.

### Task 14.5: Identify quick smoke

- [ ] **Step 1: Generate a tiny identify run:**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.pipeline --config configs/identification_single.toml --output C:\Users\borgi\tmp\v120_identify_smoke
```

Then run silx view on `dfxm_identify.h5` and confirm `/N.1` browsability.

---

## Phase 15: Final commit, tag, and follow-up handoff

**Goal:** Land the final commit, tag v1.2.0, and write the session-handoff memory file.

**Files:** None (git + memory only).

### Task 15.1: Final full-suite check

- [ ] **Step 1:**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: all green; 0 mypy errors. Total test count = baseline 430 + ~25 new tests from this plan.

### Task 15.2: Tag v1.2.0

- [ ] **Step 1: Confirm `main` is clean:**

```powershell
git status
git log --oneline -5
```

- [ ] **Step 2: Bump the version in `pyproject.toml`** from `1.1.0` to `1.2.0`. Then:

```powershell
git add pyproject.toml
git commit -m "release: v1.2.0"
git tag -a v1.2.0 -m "v1.2.0 — sub-projects A + B + C + D + E"
```

**Confirm with Sina before** running `git push origin main --tags` (CLAUDE.md rule: confirm before pushing).

### Task 15.3: Update auto-memory

- [ ] **Step 1: Create the session handoff file** at
`C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\session_handoff_2026-05-XX.md`
(use the actual date) capturing: what shipped in E, what's deferred to v1.3.0, what changes were untagged-pre-tag vs. tagged.

- [ ] **Step 2: Update `cleanup_session_state.md`** with Round 32 (sub-project E shipped, v1.2.0 tagged).

- [ ] **Step 3: Update `MEMORY.md` index** with the new handoff entry.

---

## Self-review checklist

Before declaring the plan complete, the engineer should run through:

- [ ] Every spec section in `2026-05-21-identification-hdf5-design.md` has a phase in this plan.
  - Master + per-scan-dirs layout → Phase 1-3
  - Forward retrofit → Phase 3-4
  - Identify single → Phase 7
  - Identify multi + `render_per_dislocation` → Phase 8
  - Identify z-scan → Phase 9
  - `[scan.<axis>]` consumption (all three sub-modes) → Phases 7-9 (specifically `_frame_grid_from_scan` and `_positioners_for_scan` helpers in Phase 7)
  - `IdentificationZScanConfig` simplification → Phase 5
  - `IdentificationMonteCarloConfig` extension → Phase 5
  - Migration tools (`dfxm-migrate-h5` + `dfxm-migrate-output` update) → Phase 11
  - Sidecar drops (`manifest.csv`, `images/*.png`, `im_data/*.npy`) → Phase 7 (single) + Phase 8 (multi) + Phase 9 (z-scan)
  - Docs (output-format.md + release notes) → Phase 12
- [ ] Every new helper function in Phase 7 (`_frame_grid_from_scan`, `_scan_frames_args`, `_positioners_for_scan`, `_identify_title`, `_identification_config_to_toml_str`, `_build_dislocation_sample_entry`) is used by exactly one runner (single/multi/z-scan) — DRY.
- [ ] Per-`/N.1` attrs (`scan_mode`, `scanned_axes`, `identify_mode`) are written by every runner.
- [ ] `crystal_mode` is still written for forward only (not identify), matching the spec attrs table.
- [ ] `_maybe_apply_poisson_noise` is called by all three runners (single, multi, z-scan) — confirmed in Phases 7, 8, 9.
- [ ] Per-dis files (Phase 8) skip the noise pass — verified by `_maybe_apply_poisson_noise` opening only `dfxm_sim_detector_0000.h5` (not the `_dis0/_dis1` variants).
- [ ] `MasterWriter` flushes provenance on `__exit__` even when no `add_scan` was called (Phase 2).
- [ ] `dfxm-migrate-h5` round-trip preserves all non-pixel-data nodes byte-for-byte (test in Phase 11).
- [ ] `pyproject.toml` `dfxm-migrate-h5` entry maps to `cli_main_h5_to_h5`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-21-identification-hdf5-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per phase (or per task within a phase for the larger ones), with two-stage review between phases. Phase 7 (single runner + helpers) is a good size for one subagent; Phases 8 and 9 likely each merit their own; Phases 4 and 11 are mostly mechanical and can be batched.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review at each phase boundary.

**Which approach?**
