"""Unit tests for dfxm_geo.analysis.mosaicity."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.analysis.mosaicity import compute_chi_shift, compute_com_maps


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


class TestComputeComMaps:
    def test_planted_centroids_recovered(self) -> None:
        """Each pixel has a planted intensity peak at a known (chi, phi) cell.
        compute_com_maps must return the corresponding chi_high / phi_high
        radian values."""
        chi_steps = 11
        phi_steps = 7
        chi_range = 0.1
        phi_range = 0.05
        oversample = 20
        H, W = 3, 4
        stack = np.zeros((chi_steps, phi_steps, H, W))
        # plant pixel (i, j) → peak at chi=i+1, phi=j+1 (modulo grid)
        for i in range(H):
            for j in range(W):
                stack[(i + 1) % chi_steps, (j + 1) % phi_steps, i, j] = 1.0

        phi_list, chi_list = compute_com_maps(
            stack,
            phi_range,
            phi_steps,
            chi_range,
            chi_steps,
            chi_shift=0.0,
            oversample=oversample,
        )

        phi_high = np.deg2rad(np.linspace(-phi_range, phi_range, phi_steps * oversample))
        chi_high = np.deg2rad(np.linspace(-chi_range, chi_range, chi_steps * oversample))
        assert phi_list.shape == (H, W)
        assert chi_list.shape == (H, W)
        for i in range(H):
            for j in range(W):
                expected_phi = phi_high[((j + 1) % phi_steps) * oversample]
                expected_chi = chi_high[((i + 1) % chi_steps) * oversample]
                assert phi_list[i, j] == pytest.approx(expected_phi)
                assert chi_list[i, j] == pytest.approx(expected_chi)

    def test_chi_shift_is_applied_additively(self) -> None:
        """Passing chi_shift shifts the χ output by that amount (degrees → radians)."""
        chi_steps = 11
        phi_steps = 7
        chi_range = 0.1
        phi_range = 0.05
        oversample = 20
        stack = np.zeros((chi_steps, phi_steps, 1, 1))
        stack[5, 3, 0, 0] = 1.0

        _, chi_zero = compute_com_maps(
            stack,
            phi_range,
            phi_steps,
            chi_range,
            chi_steps,
            chi_shift=0.0,
            oversample=oversample,
        )
        _, chi_shifted = compute_com_maps(
            stack,
            phi_range,
            phi_steps,
            chi_range,
            chi_steps,
            chi_shift=0.02,
            oversample=oversample,
        )
        # delta on the chi grid equals deg2rad(0.02)
        assert (chi_shifted[0, 0] - chi_zero[0, 0]) == pytest.approx(np.deg2rad(0.02), rel=1e-9)

    def test_non_square_grid(self) -> None:
        """COM extraction must not assume H == W (regression guard for the
        latent fastgrainplot non-square-grid bug class)."""
        chi_steps = 5
        phi_steps = 5
        H, W = 3, 7  # asymmetric
        stack = np.zeros((chi_steps, phi_steps, H, W))
        stack[2, 2, :, :] = 1.0

        phi_list, chi_list = compute_com_maps(
            stack,
            phi_range=0.1,
            phi_steps=phi_steps,
            chi_range=0.1,
            chi_steps=chi_steps,
            chi_shift=0.0,
            oversample=10,
        )
        assert phi_list.shape == (H, W)
        assert chi_list.shape == (H, W)

    def test_zero_intensity_pixel_produces_nan(self) -> None:
        """Dead detector pixels (all-zero rocking curve) must produce NaN in
        both phi and chi maps, not a ValueError from int(NaN)."""
        chi_steps = 5
        phi_steps = 5
        H, W = 2, 2
        stack = np.zeros((chi_steps, phi_steps, H, W))
        # Plant a peak at one pixel only; the rest stay zero (dead pixels).
        stack[2, 2, 0, 0] = 1.0

        phi_list, chi_list = compute_com_maps(
            stack,
            phi_range=0.1,
            phi_steps=phi_steps,
            chi_range=0.1,
            chi_steps=chi_steps,
            chi_shift=0.0,
            oversample=10,
        )

        # Live pixel is finite
        assert np.isfinite(phi_list[0, 0])
        assert np.isfinite(chi_list[0, 0])
        # Dead pixels are NaN, not a crash
        assert np.isnan(phi_list[0, 1])
        assert np.isnan(chi_list[0, 1])
        assert np.isnan(phi_list[1, 0])
        assert np.isnan(chi_list[1, 0])
        assert np.isnan(phi_list[1, 1])
        assert np.isnan(chi_list[1, 1])
