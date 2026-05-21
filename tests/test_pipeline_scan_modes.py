"""Unit tests for the per-axis scan primitives (sub-project B)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import AxisScanConfig


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
