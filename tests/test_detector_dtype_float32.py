"""Forward-mode detector output under the detector-noise model (Task 10).

With the detector model ON by default, forward ``scanXXXX`` detector
datasets are realistic uint16 ADU; ``model="ideal"`` keeps the original
float32 passthrough (the historical intent of this file — see
``test_detector_image_is_float32_ideal``). The perfect-crystal ``scan0002``
gets the same model when ``include_perfect_crystal=True``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.detector import PCO_EDGE_4P2_ID03
from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH
from dfxm_geo.pipeline import (
    AxisScanConfig,
    CrystalConfig,
    DetectorConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    WallCrystalConfig,
    run_simulation,
)


def _kernel_or_skip() -> None:
    kernel_dir = Path(fm.pkl_fpath)
    # #16 Slice 5: fm.Hg is no longer set by the loader; gate only on file presence.
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping.")


def _forward(
    tmp_path: Path,
    *,
    detector: DetectorConfig,
    include_perfect_crystal: bool = False,
) -> Path:
    """Smallest wall forward run; return the run output dir."""
    _kernel_or_skip()
    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        ),
        scan=ScanConfig(phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3)),
        io=IOConfig(include_perfect_crystal=include_perfect_crystal),
        detector=detector,
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    return out


@pytest.mark.slow
def test_detector_image_is_float32_ideal(tmp_path: Path) -> None:
    """model='ideal' keeps the float32 passthrough (this file's original intent)."""
    out = _forward(tmp_path, detector=DetectorConfig(model="ideal"))
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f[DETECTOR_INTERNAL_PATH]
        assert img.dtype == np.float32, f"expected float32, got {img.dtype}"
        data = img[()]
        assert "detector_model" not in img.attrs
    assert np.isfinite(data).all()
    assert float(data.sum()) > 0.0


@pytest.mark.slow
def test_forward_default_model_is_uint16_with_provenance(tmp_path: Path) -> None:
    """Default [detector] (model on) -> forward scan0001 is uint16 ADU + attrs."""
    out = _forward(tmp_path, detector=DetectorConfig())
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16, f"expected uint16, got {ds.dtype}"
        vals = ds[...]
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
        assert ds.attrs["exposure_time"] == DetectorConfig().exposure_time
    assert np.isfinite(vals).all()
    assert int(vals.min()) >= 0


@pytest.mark.slow
def test_forward_uint16_dark_floor_at_measured_offset(tmp_path: Path) -> None:
    # counts_scale=1.0 (not the 1e4 default): the wall's contrast nearly fills
    # the frame, so at the default anchor even nominally dark pixels carry
    # ~1000 ADU of signal and swamp the ~110 ADU offset floor. A tiny
    # counts_scale yields a dark-dominated scene that isolates the measured
    # offset floor — same approach as the Task 9 identify floor test.
    out = _forward(
        tmp_path,
        detector=DetectorConfig(exposure_time=1.0, counts_scale=1.0, rng_seed=0),
    )
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        vals = ds[...].astype(np.float64)
        m = PCO_EDGE_4P2_ID03
        # dark pixels sit at the measured floor: offset(1s) ~ 110 ADU.
        dark = np.percentile(vals, 5)
        assert m.offset(1.0) - 10 < dark < m.offset(1.0) + 15
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
        assert ds.attrs["exposure_time"] == 1.0


@pytest.mark.slow
def test_forward_perfect_crystal_scan_also_gets_model(tmp_path: Path) -> None:
    """include_perfect_crystal=True -> scan0002 detector is uint16 too."""
    out = _forward(tmp_path, detector=DetectorConfig(rng_seed=0), include_perfect_crystal=True)
    for scan in ("scan0001", "scan0002"):
        det = out / scan / "dfxm_sim_detector_0000.h5"
        assert det.is_file(), f"missing {det}"
        with h5py.File(det, "r") as f:
            ds = f[DETECTOR_INTERNAL_PATH]
            assert ds.dtype == np.uint16, f"{scan}: expected uint16, got {ds.dtype}"
            assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
            assert int(ds[...].min()) >= 0
