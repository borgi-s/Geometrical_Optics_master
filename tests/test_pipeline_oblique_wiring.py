"""Task 16.5: pipeline-side oblique-geometry wiring.

The bootstrap (kernel.cli_main) already threads [geometry]/[crystal] into the
LUT. These tests pin the *consumer* side in pipeline.py:

  1. SimulationConfig.from_toml parses the [geometry] block into a GeometryConfig
     and propagates eta onto ReciprocalConfig (so the analytic backend gets it).
  2. simplified configs default to mode='simplified', eta=0.0.
  3. _lookup_and_load_kernel dispatches the oblique LUT pattern when given an
     oblique GeometryConfig (theta_validated/eta), ...
  4. ... and stays on the legacy simplified pattern otherwise (back-compat).
  5. The Figure 3B path (beamstop off -> analytic backend) hands the validated
     eta to _load_analytic_resolution without touching the LUT lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.data import configs_root
from dfxm_geo.pipeline import (
    GeometryConfig,
    SimulationConfig,
    _load_resolution,
    _lookup_and_load_kernel,
)

_FIG3B = configs_root() / "al_oblique_figure3.toml"


def test_from_toml_parses_oblique_geometry() -> None:
    """The shipped Fig 3B config loads with a populated oblique GeometryConfig."""
    cfg = SimulationConfig.from_toml(_FIG3B)

    assert cfg.geometry.mode == "oblique"
    assert cfg.geometry.eta == pytest.approx(0.353140, abs=1e-6)
    # theta from compute_omega_eta(mount, hkl, keV) ~ 15.417 deg (paper Table A.2).
    assert cfg.geometry.theta_validated == pytest.approx(np.deg2rad(15.417), abs=1e-3)
    # Mount carried for provenance; paper uses a = 4.0493 angstrom.
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.a == pytest.approx(4.0493e-10)
    # eta propagated to the reciprocal config consumed by the analytic backend.
    assert cfg.reciprocal.eta == pytest.approx(0.353140, abs=1e-6)


def test_simplified_config_geometry_defaults(tmp_path: Path) -> None:
    """A config with no [geometry] block defaults to simplified / eta=0."""
    p = tmp_path / "simple.toml"
    p.write_text(
        "[reciprocal]\n"
        "hkl = [-1, 1, -1]\n"
        "keV = 17.0\n"
        "\n"
        "[crystal]\n"
        'mode = "centered"\n'
        "[crystal.centered]\n"
        "b = [1, 0, -1]\n"
        "n = [1, 1, 1]\n"
        "t = [1, -2, 1]\n"
    )
    cfg = SimulationConfig.from_toml(p)

    assert cfg.geometry.mode == "simplified"
    assert cfg.geometry.eta == 0.0
    assert cfg.geometry.theta_validated is None
    assert cfg.reciprocal.eta == 0.0


def test_lookup_and_load_kernel_dispatches_oblique(monkeypatch: pytest.MonkeyPatch) -> None:
    """An oblique GeometryConfig routes the lookup to the (theta, eta, keV) pattern."""
    calls: dict[str, dict] = {}

    def fake_lookup(*, directory, mode, keV, hkl=None, theta=None, eta=None, tol=1e-6):
        calls["lookup"] = dict(mode=mode, hkl=hkl, theta=theta, eta=eta, keV=keV)
        return Path("oblique_fake.npz")

    def fake_load(path, *, expected_hkl=None, expected_keV=None, compute_Hg=True):
        calls["load"] = dict(path=path, expected_hkl=expected_hkl, expected_keV=expected_keV)

    monkeypatch.setattr(fm, "_lookup_kernel_path", fake_lookup)
    monkeypatch.setattr(fm, "_load_default_kernel", fake_load)
    monkeypatch.setattr(fm, "_loaded_kernel_path", None)

    geom = GeometryConfig(mode="oblique", eta=0.3531, theta_validated=0.2691)
    _lookup_and_load_kernel((-1, -1, 3), 19.1, geometry=geom)

    assert calls["lookup"]["mode"] == "oblique"
    assert calls["lookup"]["theta"] == pytest.approx(0.2691)
    assert calls["lookup"]["eta"] == pytest.approx(0.3531)
    assert calls["lookup"]["keV"] == pytest.approx(19.1)
    # The oblique LUT still carries hkl/keV metadata; verify it is checked.
    assert calls["load"]["path"] == "oblique_fake.npz"
    assert calls["load"]["expected_hkl"] == (-1, -1, 3)
    assert calls["load"]["expected_keV"] == pytest.approx(19.1)


def test_lookup_and_load_kernel_simplified_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No geometry (or simplified) keeps the legacy (hkl, keV) lookup — back-compat."""
    calls: dict[str, dict] = {}

    def fake_lookup(*, directory, mode, keV, hkl=None, theta=None, eta=None, tol=1e-6):
        calls["lookup"] = dict(mode=mode, hkl=hkl, theta=theta, eta=eta, keV=keV)
        return Path("simplified_fake.npz")

    def fake_load(path, *, expected_hkl=None, expected_keV=None, compute_Hg=True):
        calls["load"] = True

    monkeypatch.setattr(fm, "_lookup_kernel_path", fake_lookup)
    monkeypatch.setattr(fm, "_load_default_kernel", fake_load)
    monkeypatch.setattr(fm, "_loaded_kernel_path", None)

    _lookup_and_load_kernel((-1, 1, -1), 17.0)  # no geometry kwarg

    assert calls["lookup"]["mode"] == "simplified"
    assert calls["lookup"]["hkl"] == (-1, 1, -1)
    assert calls["lookup"]["keV"] == pytest.approx(17.0)
    assert calls["lookup"]["theta"] is None


def test_load_resolution_oblique_analytic_passes_eta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fig 3B (beamstop off -> analytic) hands the validated eta to the analytic backend."""
    cfg = SimulationConfig.from_toml(_FIG3B)
    captured: dict[str, float] = {}

    def fake_analytic(config) -> None:
        captured["eta"] = float(config.eta)

    def boom_lookup(*args, **kwargs):
        raise AssertionError("LUT lookup must not run on the analytic Fig 3B path")

    monkeypatch.setattr(fm, "_load_analytic_resolution", fake_analytic)
    monkeypatch.setattr(fm, "_lookup_kernel_path", boom_lookup)

    _load_resolution(cfg.reciprocal, cfg.geometry)

    assert captured["eta"] == pytest.approx(0.353140, abs=1e-6)
