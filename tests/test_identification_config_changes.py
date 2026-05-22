"""Verify v1.2.0 IdentificationZScanConfig / IdentificationMonteCarloConfig shape."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import (
    IdentificationMonteCarloConfig,
    IdentificationZScanConfig,
)


def test_zscan_config_drops_phi_chi_duplicates() -> None:
    cfg = IdentificationZScanConfig(
        z_offsets_um=[-1.0, 0.0, 1.0],
        include_secondary=True,
        secondary_rng_offset=1,
    )
    # These fields must no longer exist on the dataclass:
    for stale in ("phi_range_deg", "phi_steps", "chi_range_deg", "chi_steps"):
        assert not hasattr(cfg, stale), f"{stale} should be removed in v1.2.0"


def test_zscan_config_rejects_old_kwargs() -> None:
    with pytest.raises(TypeError):
        IdentificationZScanConfig(  # type: ignore[call-arg]
            z_offsets_um=[0.0],
            phi_range_deg=0.034,  # removed
            phi_steps=21,
            chi_range_deg=0.114,
            chi_steps=21,
        )


def test_multi_config_adds_render_per_dislocation_default_false() -> None:
    cfg = IdentificationMonteCarloConfig(n_samples=10, pos_std_um=5.0)
    assert cfg.render_per_dislocation is False


def test_multi_config_render_per_dislocation_opt_in() -> None:
    cfg = IdentificationMonteCarloConfig(n_samples=10, pos_std_um=5.0, render_per_dislocation=True)
    assert cfg.render_per_dislocation is True


def test_multi_config_drops_n_png_previews() -> None:
    cfg = IdentificationMonteCarloConfig(n_samples=10, pos_std_um=5.0)
    assert not hasattr(cfg, "n_png_previews")


@pytest.mark.parametrize("axis_name", ["two_dtheta"])
def test_run_identification_eager_guards_unwired_axes(axis_name, tmp_path) -> None:
    # NOTE: `z` was removed from this parametrize in v1.3.0-A — it is now a
    # valid scan axis in identification single/multi modes (see Task 11).
    # `two_dtheta` is still rejected eagerly because the kernel can't yet
    # consume it in the identification path.
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        IdentificationConfig,
        IdentificationCrystalConfig,
        IdentificationNoiseConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
        run_identification,
    )

    scan = ScanConfig(
        phi=AxisScanConfig(value=1e-4),
        **{axis_name: AxisScanConfig(value=0.0, range=1e-3, steps=3)},
    )
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        noise=IdentificationNoiseConfig(),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    with pytest.raises(ValueError, match=axis_name):
        run_identification(cfg, tmp_path)
