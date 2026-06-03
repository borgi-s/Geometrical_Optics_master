"""S2b (#16): the simplified-theta fix propagates end-to-end to HDF5 provenance.

After S2a, run_simulation builds its ForwardContext via
``fm.build_forward_context(run_theta(config), res, config.reciprocal.hkl)``.
For the default reflection ``(-1,1,-1) @ 17 keV``, ``run_theta`` returns the
true Bragg angle (≈ 0.15661142 rad, 8.97317°) which DIFFERS from the legacy
module global ``fm.theta_0`` (≈ 0.15666948 rad, 8.97650°).

This test pins that propagation: the θ persisted in both storage locations
inside the master HDF5 must equal ``run_theta(config)``, not the legacy global.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


@pytest.fixture
def _kernel_on_disk() -> None:
    """Skip if the canonical (-1,1,-1) 17 keV kernel npz is not on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no (-1,1,-1) 17 keV kernel npz found in {kernel_dir}")


def test_pipeline_persists_true_bragg_theta_not_legacy_global(
    tmp_path: Path, _kernel_on_disk: None
) -> None:
    """run_simulation writes the true Bragg θ to HDF5, not the legacy global.

    Verifies both storage locations:
      - the attr ``/1.1.attrs["theta"]``       (written by write_simulation_h5)
      - the dataset ``/1.1/dfxm_geo/theta``    (written by MasterWriter.add_scan)
    """
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        CrystalConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
        SimulationConfig,
        WallCrystalConfig,
        run_simulation,
        run_theta,
    )

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=10, sample_remount="S1"),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3),
            chi=AxisScanConfig(range=0.002 * 180 / np.pi, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=True, max_workers=1),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)

    expected = run_theta(cfg)  # true Bragg ≈ 0.15661142 rad (8.97317°)
    legacy = fm.theta_0  # import-time global ≈ 0.15666948 rad (8.97650°)

    # Sanity: the two values must actually differ so the assertion is meaningful.
    assert abs(expected - legacy) > 1e-5, (
        f"expected ({expected}) and legacy ({legacy}) are too close; "
        "the theta-fix may not have an observable effect"
    )

    h5_path = out / "dfxm_geo.h5"
    assert h5_path.exists(), f"master HDF5 not created: {h5_path}"

    with h5py.File(h5_path, "r") as f:
        # Location 1: /1.1 attribute (written by write_simulation_h5 via attrs_1_1)
        theta_attr = float(f["1.1"].attrs["theta"])
        # Location 2: /1.1/dfxm_geo/theta dataset (written by MasterWriter.add_scan)
        theta_ds = float(f["1.1/dfxm_geo/theta"][()])

    assert theta_attr == pytest.approx(expected, abs=1e-9), (
        f"HDF5 attr /1.1.attrs['theta']={theta_attr:.10f} does not match "
        f"true Bragg {expected:.10f} (legacy={legacy:.10f})"
    )
    assert theta_ds == pytest.approx(expected, abs=1e-9), (
        f"HDF5 dataset /1.1/dfxm_geo/theta={theta_ds:.10f} does not match "
        f"true Bragg {expected:.10f} (legacy={legacy:.10f})"
    )
