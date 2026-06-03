"""Oblique-geometry wiring for the IDENTIFICATION path (v2.3.0).

The forward path (`SimulationConfig` / `run_simulation`) already threads the
TOML `[geometry]` block into a `GeometryConfig`, routes it to the resolution
backend, and builds a ForwardContext via ``build_forward_context(run_theta(config),
...)`` so the ray grid + imaging rotation use the run reflection's Bragg angle.
The identification path had none of this: `IdentificationConfig` carried no
geometry field, `load_identification_config` ignored `[geometry]`, and
`run_identification` called `_load_resolution` without geometry — so an oblique
`dfxm-identify` run silently fell back to the simplified Al-111 kernel.

These tests pin the identify-side wiring as the mirror of the forward side:
  1. IdentificationConfig defaults to simplified geometry (back-compat).
  2. A config with no [geometry] block loads as simplified / eta=0.
  3. An oblique [geometry] block loads into a populated oblique GeometryConfig
     (eta + validated theta + mount), propagates eta onto ReciprocalConfig,
     and still builds the IdentificationCrystalConfig (mount keys stripped).
  4. run_identification threads config.geometry into _load_resolution AND
     builds a ForwardContext whose geometry.theta_0 equals run_theta(config)
     (S3 of #16: the reflection_theta_if_oblique CM call site is retired;
     geometry is sourced from build_forward_context, not module globals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dfxm_geo.pipeline import (
    GeometryConfig,
    IdentificationConfig,
    _identification_config_to_toml_str,
    load_identification_config,
    run_identification,
)

# Known-good oblique geometry (paper Fig 3B / al_oblique_figure3.toml):
# eta=0.353140 rad (20.233 deg), reflection (-1,-1,3) @ 19.1 keV, identity
# mount, a=4.0493 angstrom -> theta ~ 15.417 deg (Table A.2).
_OBLIQUE_ETA = 0.353140
_OBLIQUE_THETA = np.deg2rad(15.417)


def _write_oblique_identify_toml(path: Path) -> None:
    path.write_text(
        'mode = "single"\n'
        "\n"
        "[crystal]\n"
        "slip_plane_normal = [1, 1, -1]\n"
        "angle_start_deg = 0.0\n"
        "angle_stop_deg = 0.0\n"
        "angle_step_deg = 10.0\n"
        "b_vector_indices = [0]\n"
        "sweep_all_slip_planes = false\n"
        "exclude_invisibility = false\n"
        "# oblique mount schema (consumed by _build_geometry_config, not the\n"
        "# IdentificationCrystalConfig):\n"
        'lattice = "cubic"\n'
        "a = 4.0493e-10\n"
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_OBLIQUE_ETA}\n"
        "\n"
        "[reciprocal]\n"
        "hkl = [-1, -1, 3]\n"
        "keV = 19.1\n"
    )


def test_identification_config_default_geometry_is_simplified() -> None:
    """A bare IdentificationConfig defaults to simplified geometry."""
    cfg = IdentificationConfig()
    assert cfg.geometry.mode == "simplified"
    assert cfg.geometry.eta == 0.0
    assert cfg.geometry.theta_validated is None


def test_load_identification_config_simplified_default(tmp_path: Path) -> None:
    """An identify config with no [geometry] block loads as simplified."""
    p = tmp_path / "simple_identify.toml"
    p.write_text('mode = "single"\n[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n')
    cfg = load_identification_config(p)
    assert cfg.geometry.mode == "simplified"
    assert cfg.geometry.eta == 0.0
    assert cfg.geometry.theta_validated is None
    assert cfg.reciprocal.eta == 0.0


def test_load_identification_config_parses_oblique_geometry(tmp_path: Path) -> None:
    """An oblique identify config loads with a populated oblique GeometryConfig
    and a valid IdentificationCrystalConfig (mount keys stripped)."""
    p = tmp_path / "oblique_identify.toml"
    _write_oblique_identify_toml(p)
    cfg = load_identification_config(p)

    assert cfg.geometry.mode == "oblique"
    assert cfg.geometry.eta == pytest.approx(_OBLIQUE_ETA, abs=1e-6)
    assert cfg.geometry.theta_validated == pytest.approx(_OBLIQUE_THETA, abs=1e-3)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.a == pytest.approx(4.0493e-10)
    # eta propagated to the reciprocal config consumed by the analytic backend.
    assert cfg.reciprocal.eta == pytest.approx(_OBLIQUE_ETA, abs=1e-6)
    # The identify crystal config still parses (mount keys must not leak into it).
    assert cfg.crystal.slip_plane_normal == (1, 1, -1)
    assert cfg.crystal.b_vector_indices == [0]
    assert cfg.crystal.sweep_all_slip_planes is False


def test_run_identification_threads_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_identification passes config.geometry to _load_resolution and
    builds a ForwardContext whose geometry.theta_0 equals run_theta(config).

    S3 of #16: the reflection_theta_if_oblique CM call site is retired;
    geometry now comes from build_forward_context(run_theta(config), ...) so
    the ctx passed to the runners carries the correct oblique Bragg angle.
    """
    captured: dict = {}

    # Build a stub ResolutionContext so build_forward_context can be called
    # without needing a kernel on disk.  We only care that ctx.geometry.theta_0
    # is set correctly; the resolution fields are not exercised by this test.
    from dfxm_geo.direct_space.forward_model import ResolutionContext

    stub_resolution = ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=1.0,
        qi2_start=0.0,
        qi2_step=1.0,
        qi3_start=0.0,
        qi3_step=1.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=None,
        loaded_kernel_path=None,
    )

    def fake_load_resolution(reciprocal, geometry=None):
        captured["geometry"] = geometry
        return stub_resolution

    def fake_single(config, output_dir, ctx):
        captured["ran"] = True
        captured["ctx_theta"] = ctx.geometry.theta_0
        return {"n_images": 0, "output_dir": output_dir}

    monkeypatch.setattr("dfxm_geo.pipeline._load_resolution", fake_load_resolution)
    monkeypatch.setattr("dfxm_geo.pipeline._run_identification_single", fake_single)

    cfg = IdentificationConfig(
        mode="single",
        geometry=GeometryConfig(mode="oblique", eta=_OBLIQUE_ETA, theta_validated=_OBLIQUE_THETA),
    )
    run_identification(cfg, tmp_path)

    # Geometry config threaded through to _load_resolution.
    assert captured["geometry"].mode == "oblique"
    # ctx passed to the runner carries the oblique Bragg angle (not the
    # simplified default ~0.1567 rad = 8.98 deg).
    assert captured["ctx_theta"] == pytest.approx(_OBLIQUE_THETA, abs=1e-6)
    assert captured["ran"] is True


def test_oblique_identify_provenance_round_trips_through_toml_str(tmp_path: Path) -> None:
    """The provenance TOML embedded in the HDF5 master for an oblique identify
    run must round-trip back to oblique (not silently 'simplified')."""
    p = tmp_path / "oblique_identify.toml"
    _write_oblique_identify_toml(p)
    cfg = load_identification_config(p)
    assert cfg.geometry.mode == "oblique"  # precondition

    rt = tmp_path / "provenance_round_trip.toml"
    rt.write_text(_identification_config_to_toml_str(cfg))
    reloaded = load_identification_config(rt)

    assert reloaded.geometry.mode == "oblique"
    assert reloaded.geometry.eta == pytest.approx(_OBLIQUE_ETA, abs=1e-6)
    assert reloaded.geometry.theta_validated == pytest.approx(_OBLIQUE_THETA, abs=1e-3)
