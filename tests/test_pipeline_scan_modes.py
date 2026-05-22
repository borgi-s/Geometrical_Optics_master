"""Unit tests for the per-axis scan primitives (sub-project B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import AxisScanConfig, ScanConfig


class TestAxisScanConfig:
    def test_default_is_fixed_at_zero(self) -> None:
        axis = AxisScanConfig()
        assert axis.value == 0.0
        assert axis.range is None
        assert axis.steps is None
        assert not axis.is_scanned

    def test_fixed_with_nonzero_value(self) -> None:
        axis = AxisScanConfig(value=1.5e-4)
        assert axis.value == 1.5e-4
        assert not axis.is_scanned

    def test_scanned_centered_on_zero(self) -> None:
        axis = AxisScanConfig(range=1e-3, steps=61)
        assert axis.is_scanned
        assert axis.value == 0.0
        assert axis.range == 1e-3
        assert axis.steps == 61

    def test_scanned_with_offset(self) -> None:
        axis = AxisScanConfig(value=1.5e-4, range=1e-3, steps=61)
        assert axis.is_scanned
        assert axis.value == 1.5e-4

    def test_range_without_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="both `range` and `steps`"):
            AxisScanConfig(range=1e-3)

    def test_steps_without_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="both `range` and `steps`"):
            AxisScanConfig(steps=61)

    def test_zero_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            AxisScanConfig(range=0.0, steps=61)

    def test_negative_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            AxisScanConfig(range=-1e-3, steps=61)

    def test_steps_below_two_rejected(self) -> None:
        with pytest.raises(ValueError, match="`steps` must be >= 2"):
            AxisScanConfig(range=1e-3, steps=1)


class TestScanConfigFromDict:
    def test_empty_dict_all_axes_fixed_at_zero(self) -> None:
        cfg = ScanConfig.from_dict({})
        assert not cfg.phi.is_scanned
        assert not cfg.chi.is_scanned
        assert not cfg.two_dtheta.is_scanned
        assert not cfg.z.is_scanned

    def test_none_dict_all_axes_fixed_at_zero(self) -> None:
        cfg = ScanConfig.from_dict(None)
        assert not cfg.phi.is_scanned

    def test_mosa_grid(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 61},
                "chi": {"range": 2e-3, "steps": 61},
            }
        )
        assert cfg.phi.is_scanned and cfg.phi.range == 6e-4
        assert cfg.chi.is_scanned and cfg.chi.range == 2e-3
        assert not cfg.two_dtheta.is_scanned
        assert not cfg.z.is_scanned

    def test_single_image_with_phi_offset(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"value": 1.5e-4}})
        assert not cfg.phi.is_scanned
        assert cfg.phi.value == 1.5e-4

    def test_rocking_strain(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 61},
                "two_dtheta": {"range": 5e-4, "steps": 41},
            }
        )
        assert cfg.phi.is_scanned
        assert cfg.two_dtheta.is_scanned
        assert not cfg.chi.is_scanned

    def test_unknown_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown scan axis 'omega'"):
            ScanConfig.from_dict({"omega": {"range": 1e-3, "steps": 21}})

    def test_axis_value_propagated_to_axis_config(self) -> None:
        cfg = ScanConfig.from_dict({"chi": {"value": 1e-5, "range": 2e-3, "steps": 41}})
        assert cfg.chi.value == 1e-5
        assert cfg.chi.is_scanned

    def test_invalid_axis_data_propagates_post_init(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            ScanConfig.from_dict({"phi": {"range": 0.0, "steps": 21}})


class TestDerivedModeName:
    def _scanned(self, **axes_with_range: tuple[float, int]) -> ScanConfig:
        """Build a ScanConfig where named axes are scanned, others fixed."""
        data = {name: {"range": r, "steps": s} for name, (r, s) in axes_with_range.items()}
        return ScanConfig.from_dict(data)

    def test_no_axes_scanned_is_single(self) -> None:
        assert ScanConfig().derived_mode_name() == "single"

    def test_phi_only_is_rocking(self) -> None:
        assert self._scanned(phi=(6e-4, 61)).derived_mode_name() == "rocking"

    def test_chi_only_is_rolling(self) -> None:
        assert self._scanned(chi=(2e-3, 61)).derived_mode_name() == "rolling"

    def test_two_dtheta_only_is_strain(self) -> None:
        assert self._scanned(two_dtheta=(5e-4, 41)).derived_mode_name() == "strain"

    def test_z_only_is_layer(self) -> None:
        assert self._scanned(z=(1e-6, 5)).derived_mode_name() == "layer"

    def test_phi_chi_is_mosa(self) -> None:
        assert self._scanned(phi=(6e-4, 61), chi=(2e-3, 61)).derived_mode_name() == "mosa"

    def test_phi_chi_two_dtheta_is_mosa_strain(self) -> None:
        assert (
            self._scanned(phi=(6e-4, 61), chi=(2e-3, 61), two_dtheta=(5e-4, 41)).derived_mode_name()
            == "mosa_strain"
        )

    def test_phi_chi_z_is_mosa_layer(self) -> None:
        assert (
            self._scanned(phi=(6e-4, 61), chi=(2e-3, 61), z=(1e-6, 5)).derived_mode_name()
            == "mosa_layer"
        )

    def test_phi_chi_two_dtheta_z_is_mosa_strain_layer(self) -> None:
        assert (
            self._scanned(
                phi=(6e-4, 61), chi=(2e-3, 61), two_dtheta=(5e-4, 41), z=(1e-6, 5)
            ).derived_mode_name()
            == "mosa_strain_layer"
        )

    def test_non_canonical_combo_concatenates_in_axis_order(self) -> None:
        # phi + two_dtheta = "rocking_strain"  (chi missing → not mosa)
        assert (
            self._scanned(phi=(6e-4, 61), two_dtheta=(5e-4, 41)).derived_mode_name()
            == "rocking_strain"
        )
        # chi + z = "rolling_layer"
        assert self._scanned(chi=(2e-3, 61), z=(1e-6, 5)).derived_mode_name() == "rolling_layer"
        # phi + z (no chi, no two_dtheta) is "rocking_layer"
        assert self._scanned(phi=(6e-4, 61), z=(1e-6, 5)).derived_mode_name() == "rocking_layer"


class TestScannedAxesAndIsScanned:
    def test_scanned_axes_empty(self) -> None:
        assert ScanConfig().scanned_axes() == ()

    def test_scanned_axes_ordered_canonically(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                # Insertion order: z, phi, two_dtheta (deliberately not canonical)
                "z": {"range": 1e-6, "steps": 5},
                "phi": {"range": 6e-4, "steps": 61},
                "two_dtheta": {"range": 5e-4, "steps": 41},
            }
        )
        assert cfg.scanned_axes() == ("phi", "two_dtheta", "z")

    def test_is_scanned_per_axis(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"range": 6e-4, "steps": 61}})
        assert cfg.is_scanned("phi")
        assert not cfg.is_scanned("chi")
        assert not cfg.is_scanned("two_dtheta")
        assert not cfg.is_scanned("z")

    def test_is_scanned_unknown_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown axis 'omega'"):
            ScanConfig().is_scanned("omega")


# ---------------------------------------------------------------------------
# Integration test: run_simulation with two_dtheta scan no longer raises
# ---------------------------------------------------------------------------


def test_run_simulation_two_dtheta_scan_lifts_value_error(tmp_path: Path) -> None:
    """Setting [scan.two_dtheta] no longer raises; produces a 4D scan."""
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        CenteredCrystalConfig,
        CrystalConfig,
        IOConfig,
        PostprocessConfig,
        ReciprocalConfig,
        ScanConfig,
        SimulationConfig,
        run_simulation,
    )

    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            chi=AxisScanConfig(range=2e-3, steps=2),
            two_dtheta=AxisScanConfig(range=1e-4, steps=2),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)

    import h5py

    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        # 2 phi * 2 chi * 2 two_dtheta = 8 frames
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 8
        assert f["/1.1"].attrs["scan_mode"] == "mosa_strain"
        assert sorted(f["/1.1"].attrs["scanned_axes"]) == ["chi", "phi", "two_dtheta"]
        # two_dtheta is a per-frame positioner array
        assert f["/1.1/instrument/positioners/two_dtheta"].shape == (8,)


def test_run_simulation_z_scan_recomputes_hg_per_z(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [scan.z] config triggers one Find_Hg(_from_population) call per unique z."""
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        CenteredCrystalConfig,
        CrystalConfig,
        IOConfig,
        PostprocessConfig,
        ReciprocalConfig,
        ScanConfig,
        SimulationConfig,
        run_simulation,
    )

    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")

    # Spy on Find_Hg_from_population to count calls.
    calls: list[float] = []
    real = fm.Find_Hg_from_population

    def spy(pop, *args, rl=None, **kwargs):  # type: ignore[no-untyped-def]
        # Reverse-engineer z from rl identity: rl is None -> z=0; rl is shifted otherwise.
        if rl is None:
            calls.append(0.0)
        else:
            # We can't easily reverse the offset from rl alone; just record "non-zero".
            calls.append(float("nan"))
        return real(pop, *args, rl=rl, **kwargs)

    monkeypatch.setattr(fm, "Find_Hg_from_population", spy)

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            z=AxisScanConfig(range=5.0, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)
    # 3 unique z values + 1 baseline call (Hg_provider(0.0) for q_hkl provenance)
    assert len(calls) >= 3, f"expected >=3 Find_Hg_from_population calls, got {len(calls)}"

    import h5py

    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        # 2 phi * 3 z = 6 frames
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 6
        # z is a per-frame positioner
        assert f["/1.1/instrument/positioners/z"].shape == (6,)
