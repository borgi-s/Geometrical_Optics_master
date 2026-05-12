"""Unit tests for dfxm_geo.analysis.mosaicity."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.analysis.mosaicity import compute_chi_shift


class TestComputeChiShift:
    def test_corner_peaked_at_middle_returns_baseline_shift(self) -> None:
        """A perfect-crystal stack whose corner pixel peaks at the middle chi
        cell should return a small baseline shift (the systematic discretization
        offset). Asserts against the analytical formula in the docstring."""
        chi_steps = 11  # odd → integer middle index
        phi_steps = 5
        chi_range = 0.1  # degrees
        stack = np.zeros((chi_steps, phi_steps, 4, 4))
        # corner pixel (-1, -1) peaks at chi-index 5 (middle of 11)
        stack[5, :, -1, -1] = 1.0

        shift = compute_chi_shift(stack, chi_steps, chi_range, oversample=100)

        # com[0] = 5.0; shift_idx = 5*100 - 11*100/2 = -50; abs = 50
        # chi_high has 1100 elements over [-0.1, 0.1] → step = 0.2/1099
        expected = 50 * (2 * chi_range / (chi_steps * 100 - 1))
        assert shift == pytest.approx(expected, rel=1e-9)

    def test_shift_grows_when_corner_pixel_shifts(self) -> None:
        """Moving the corner-pixel peak by one chi-grid cell increases |shift| by
        exactly that one-cell width in degrees."""
        chi_steps = 11
        chi_range = 0.1
        oversample = 100
        one_cell_deg = oversample * (2 * chi_range / (chi_steps * oversample - 1))

        stack_centered = np.zeros((chi_steps, 5, 4, 4))
        stack_centered[5, :, -1, -1] = 1.0
        stack_shifted = np.zeros((chi_steps, 5, 4, 4))
        stack_shifted[4, :, -1, -1] = 1.0

        s_centered = compute_chi_shift(stack_centered, chi_steps, chi_range, oversample=oversample)
        s_shifted = compute_chi_shift(stack_shifted, chi_steps, chi_range, oversample=oversample)
        assert (s_shifted - s_centered) == pytest.approx(one_cell_deg, rel=1e-9)

    def test_sign_loss_peaks_above_center_collapse_to_baseline(self) -> None:
        """The abs() sign-loss means a peak one step above center gives the
        same magnitude as the centered baseline (shift_idx sign is discarded)."""
        chi_steps, chi_range, oversample = 11, 0.1, 100
        stack_centered = np.zeros((chi_steps, 5, 4, 4))
        stack_centered[5, :, -1, -1] = 1.0
        stack_above = np.zeros((chi_steps, 5, 4, 4))
        stack_above[6, :, -1, -1] = 1.0  # one step above center
        s_centered = compute_chi_shift(stack_centered, chi_steps, chi_range, oversample=oversample)
        s_above = compute_chi_shift(stack_above, chi_steps, chi_range, oversample=oversample)
        # Because the implementation does int(abs(shift_idx)), peaks at index 4
        # and index 6 produce the same magnitude (mirror around the center).
        assert s_above == pytest.approx(s_centered, rel=1e-9)
