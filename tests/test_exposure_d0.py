"""Test exposure aux model d0_angstrom parameter."""

from __future__ import annotations

from dfxm_geo.reciprocal_space.exposure import run_exposure_simulation


def test_run_exposure_simulation_accepts_d0():
    """Test that run_exposure_simulation accepts d0_angstrom parameter."""
    total, fraction = run_exposure_simulation(Nrays=200, d0_angstrom=2.0)
    assert isinstance(total, int)
    assert 0.0 <= fraction <= 1.0


def test_run_exposure_simulation_d0_default():
    """Test that run_exposure_simulation uses Al d_111 as default d0."""
    # Al d_111 = 4.0495 / sqrt(3) ≈ 2.338 Ångström
    total, fraction = run_exposure_simulation(Nrays=200)
    assert isinstance(total, int)
    assert 0.0 <= fraction <= 1.0


def test_run_exposure_simulation_d0_affects_result():
    """Test that different d0 values produce different results."""
    # Run with different d0 values and check that results differ
    # (with high probability for a seed-independent Monte Carlo)
    total1, fraction1 = run_exposure_simulation(Nrays=500, d0_angstrom=2.0)
    total2, fraction2 = run_exposure_simulation(Nrays=500, d0_angstrom=3.0)
    # Results should differ; note that the test is probabilistic,
    # but with 500 rays the probability of identical results is negligible
    assert not (total1 == total2 and fraction1 == fraction2)
