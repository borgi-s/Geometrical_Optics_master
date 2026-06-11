"""Verify render_per_dislocation=True emits 3 detector files per scan.

The combined detector (`dfxm_sim_detector`) carries the sum of both
dislocations and is subject to optional Poisson noise post-write. The
per-dislocation detectors (`*_dis0`, `*_dis1`) are noiseless by
design — they bypass the noise pass and stay deterministic across runs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    matches = sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))
    if not matches:
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def test_render_per_dislocation_writes_three_files(tmp_path: Path) -> None:
    _require_kernel()
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
        for name in (
            "dfxm_sim_detector",
            "dfxm_sim_detector_dis0",
            "dfxm_sim_detector_dis1",
        ):
            assert name in instr
            assert instr[name].attrs["NX_class"] == "NXdetector"


def test_per_dis_renders_are_positioned(tmp_path: Path, monkeypatch) -> None:
    """Per-instance renders must be drawn at each dislocation's scene position.

    Regression: the combined scene threads ``position_lab_um`` into
    ``Fd_find_multi_dislocs_mixed``, but the per-dislocation renders called
    ``Fd_find_mixed`` *without* it, so ``dis0`` / ``dis1`` were rendered at the
    origin and could not overlay the combined image as instance labels. The
    render call must receive the same position stored under
    ``/N.1/sample/dislocations/<k>/position_um``.

    W2 (v2.6.0): the orchestrator now routes through ``find_hg_scene`` which
    carries all positions inside the ``specs`` list — spying on the seam and
    checking each spec's ``position_lab_um`` verifies the contract. End-to-end
    position fidelity (that the rendered Hg was actually computed at each spec's
    position) is covered by the bit-identity tests in ``tests/test_hg_scene.py``;
    this test verifies positions flow into the seam's specs and match the
    HDF5-stored values.
    """
    _require_kernel()
    import dfxm_geo.orchestrator as pipeline
    from dfxm_geo.crystal.dislocations import find_hg_scene as _real_scene

    captured_specs: list = []

    def spy(rl_um, Us, specs, Theta, **kwargs):
        captured_specs.extend(specs)
        return _real_scene(rl_um, Us, specs, Theta, **kwargs)

    monkeypatch.setattr(pipeline, "find_hg_scene", spy)

    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(
            n_samples=1, pos_std_um=5.0, render_per_dislocation=True
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)

    # One sample with render_per_dislocation -> find_hg_scene receives 2 specs.
    assert len(captured_specs) == 2
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        pos0 = f["/1.1/sample/dislocations/0/position_um"][()]
        pos1 = f["/1.1/sample/dislocations/1/position_um"][()]
    np.testing.assert_allclose(captured_specs[0].position_lab_um, pos0, atol=1e-9)
    np.testing.assert_allclose(captured_specs[1].position_lab_um, pos1, atol=1e-9)


def test_per_dis_files_are_noiseless(tmp_path: Path) -> None:
    """With poisson_noise=True, dis0/dis1 stay deterministic (noiseless).

    The post-write Poisson pass touches only `dfxm_sim_detector_0000.h5`;
    `*_dis0` and `*_dis1` files are written once and never modified, so
    two runs at the same seed must produce byte-identical per-dis arrays.
    """
    _require_kernel()
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
    out1 = tmp_path / "first"
    out2 = tmp_path / "second"
    out1.mkdir()
    out2.mkdir()
    run_identification(cfg, out1)
    run_identification(cfg, out2)
    for name in (
        "dfxm_sim_detector_dis0_0000.h5",
        "dfxm_sim_detector_dis1_0000.h5",
    ):
        with (
            h5py.File(out1 / "scan0001" / name, "r") as a,
            h5py.File(out2 / "scan0001" / name, "r") as b,
        ):
            np.testing.assert_array_equal(
                a["/entry_0000/dfxm_sim_detector/image"][...],
                b["/entry_0000/dfxm_sim_detector/image"][...],
            )
