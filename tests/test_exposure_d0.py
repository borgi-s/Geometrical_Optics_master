"""Test exposure aux model d0_angstrom parameter."""

from __future__ import annotations

import numpy as np

from dfxm_geo.reciprocal_space.exposure import run_exposure_simulation


def test_run_exposure_simulation_accepts_d0():
    """run_exposure_simulation accepts d0_angstrom and returns sane stats."""
    total, fraction = run_exposure_simulation(Nrays=200, d0_angstrom=2.0)
    assert isinstance(total, int)
    assert 0.0 <= fraction <= 1.0


def test_run_exposure_simulation_d0_affects_result(monkeypatch):
    """Different d0 values change the result under an identical RNG stream.

    The simulation is intentionally unseeded in production; to make this
    comparison deterministic (no flake), pin ``np.random.default_rng`` so
    both calls consume the same random stream and only d0 differs.
    """
    monkeypatch.setattr(np.random, "default_rng", lambda: np.random.Generator(np.random.PCG64(42)))
    total1, fraction1 = run_exposure_simulation(Nrays=500, d0_angstrom=2.0)
    total2, fraction2 = run_exposure_simulation(Nrays=500, d0_angstrom=3.0)
    assert (total1, fraction1) != (total2, fraction2)
