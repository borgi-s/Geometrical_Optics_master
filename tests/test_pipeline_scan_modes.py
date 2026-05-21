"""Unit tests for the per-axis scan primitives (sub-project B)."""

from __future__ import annotations

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
