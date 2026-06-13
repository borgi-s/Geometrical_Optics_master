"""M4 Stage 4.3a Task 11: FCC bit-identity / determinism e2e gate.

The STRONG algebraic bit-identity proof lives in ``test_forward_dispatch.py``
(Task 8: ``test_fcc_random_draw_byte_identical_to_legacy_slip_table`` replays
the exact per-dislocation ``rng.integers(0, 12)`` draw against the deleted
hand-authored ``_SLIP_SYSTEM_111`` table and asserts the realized ``Ud`` +
sidecar (b, n, t) sequence matches byte-for-byte) and in
``test_identify_structure_aware.py`` (Task 9: FCC identify reuses the exact
v2.x ``_ALL_111_PLANES`` / ``_burgers_vectors`` objects by identity).

This file adds the two complementary E2E guards that the algebraic tests don't
cover:

  1. Full-pipeline DETERMINISM: a seeded FCC ``random_dislocations`` forward run
     twice produces byte-identical detector image arrays.  Guards the whole
     forward path (population draw -> Find_Hg -> kernel -> detector write), not
     just the per-dislocation slip draw.
  2. No-new-attrs at the OUTPUT level: a default FCC simplified forward writes
     NO ``structure_type`` provenance attr on ``/1.1`` — the byte-identity gate
     where it matters for downstream readers (v2.5.x FCC outputs unchanged).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    CrystalConfig,
    IOConfig,
    RandomDislocationsConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    run_simulation,
)


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk.

    Mirrors test_detector_dtype_float32._kernel_or_skip — the default FCC
    simplified path uses the MC resolution kernel, not the analytic backend.
    """
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def _fcc_random_cfg() -> SimulationConfig:
    """Default FCC (mount=None -> simplified) random_dislocations, seeded.

    Small population (ndis=3) + single frame keeps the run fast; the seed is
    what the determinism assertion turns on.
    """
    return SimulationConfig(
        crystal=CrystalConfig(
            mode="random_dislocations",
            random_dislocations=RandomDislocationsConfig(ndis=3, sigma=5.0, seed=2024),
        ),
        scan=ScanConfig(phi=AxisScanConfig(value=0.0)),
        io=IOConfig(include_perfect_crystal=False, write_strain_provenance=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )


def _detector_image(out_dir: Path) -> np.ndarray:
    det = out_dir / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file(), f"no detector file written under {out_dir}"
    with h5py.File(det, "r") as f:
        return f["/entry_0000/dfxm_sim_detector/image"][...]


@pytest.mark.slow
def test_fcc_random_dislocations_forward_deterministic(tmp_path: Path) -> None:
    """A seeded FCC random_dislocations forward is byte-identical across runs.

    End-to-end determinism guard: runs the SAME seeded config into two output
    dirs and asserts the detector image arrays are EXACTLY equal. This covers
    the full pipeline (seeded population draw -> Hg -> kernel -> detector),
    complementing the algebraic per-draw proof in test_forward_dispatch.py.
    """
    _require_kernel()

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    run_simulation(_fcc_random_cfg(), out_a)
    run_simulation(_fcc_random_cfg(), out_b)

    img_a = _detector_image(out_a)
    img_b = _detector_image(out_b)
    assert img_a.shape == img_b.shape
    assert np.isfinite(img_a).all()
    assert float(img_a.max()) > 0.0
    assert np.array_equal(img_a, img_b), "FCC seeded forward is not deterministic end-to-end"


def test_fcc_forward_writes_no_structure_attrs(tmp_path: Path) -> None:
    """A default FCC simplified forward writes NO structure_type attr on /1.1.

    Output-level byte-identity gate: the v2.5.x FCC simplified path
    (mount=None) must not start emitting structure-family provenance, so
    existing FCC outputs / downstream readers are unchanged. (The structure
    attrs only appear on structure-aware / oblique runs — see
    test_pipeline_writes_oblique_provenance.py.)
    """
    _require_kernel()

    out = tmp_path / "run"
    run_simulation(_fcc_random_cfg(), out)

    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["geometry_mode"] == "simplified"
    for key in ("structure_type", "poisson_ratio", "poisson_source", "burgers_magnitude_um"):
        assert key not in attrs, f"unexpected structure attr {key!r} on FCC simplified /1.1"
