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
