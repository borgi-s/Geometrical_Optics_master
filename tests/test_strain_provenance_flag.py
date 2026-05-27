"""io.write_strain_provenance gates the large per-scan Hg array in the master."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    run_simulation,
)


def _kernel_or_skip() -> None:
    # Only the on-disk LUT npz is needed; run_simulation computes Hg itself.
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping.")


def _tiny_cfg(write_strain_provenance: bool) -> SimulationConfig:
    # Default centered crystal (omit crystal block via defaults), tiny phi scan.
    return SimulationConfig(
        scan=ScanConfig(phi=AxisScanConfig(range=0.001, steps=2)),
        io=IOConfig(
            include_perfect_crystal=False,
            write_strain_provenance=write_strain_provenance,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )


def test_strain_provenance_written_by_default(tmp_path: Path) -> None:
    _kernel_or_skip()
    run_simulation(_tiny_cfg(True), tmp_path / "on")
    with h5py.File(tmp_path / "on" / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/Hg" in f
        assert "/1.1/dfxm_geo/q_hkl" in f  # scalars always present


def test_strain_provenance_omitted_when_disabled(tmp_path: Path) -> None:
    _kernel_or_skip()
    run_simulation(_tiny_cfg(False), tmp_path / "off")
    with h5py.File(tmp_path / "off" / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/Hg" not in f  # big array dropped
        assert "/1.1/dfxm_geo/q_hkl" in f  # tiny scalars retained


def test_write_strain_provenance_round_trips_from_toml(tmp_path: Path) -> None:
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text("[io]\nwrite_strain_provenance = false\n", encoding="utf-8")
    cfg = SimulationConfig.from_toml(toml_path)
    assert cfg.io.write_strain_provenance is False
