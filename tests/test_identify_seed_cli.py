"""TDD tests for `--seed` on `dfxm-identify` (cli_main_identify).

Tests the seed-override path without requiring an on-disk kernel by:
  1. Checking the RNG draw directly from IdentificationNoiseConfig
  2. Verifying cli_main_identify's argparse wires the seed into cfg.noise.rng_seed

For the determinism test we use _draw_dislocation (the actual sampler used by
_iter_identification_multi) seeded with two different seeds from noise.rng_seed.
This tests the full contract — same seed → same draws, different seed → different
draws — without invoking the kernel-dependent forward model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dfxm_geo.pipeline import (
    IdentificationConfig,
    _draw_dislocation,
    cli_main_identify,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_rng(rng_seed: int) -> np.random.Generator:
    """Return the *param* child RNG for the given seed (same split as pipeline)."""
    param_rng, _noise_rng = np.random.default_rng(rng_seed).spawn(2)
    return param_rng


def _first_draw(rng_seed: int) -> dict:
    """Draw a single dislocation from param_rng at the given seed."""
    rng = _param_rng(rng_seed)
    return _draw_dislocation(rng, pos_std_um=5.0)


# ---------------------------------------------------------------------------
# Seed determinism contract (no kernel needed)
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_dislocation_draw() -> None:
    """Same rng_seed → identical drawn dislocation parameters."""
    d1 = _first_draw(42)
    d2 = _first_draw(42)
    assert d1["plane"] == d2["plane"]
    assert d1["b_idx"] == d2["b_idx"]
    assert d1["alpha_deg"] == pytest.approx(d2["alpha_deg"])
    np.testing.assert_array_equal(d1["pos_um"], d2["pos_um"])


def test_different_seed_produces_different_dislocation_draw() -> None:
    """Different rng_seed → different drawn parameters (very high probability)."""
    d1 = _first_draw(0)
    d2 = _first_draw(99)
    # At least one of plane, b_idx, alpha must differ
    differs = (
        d1["plane"] != d2["plane"]
        or d1["b_idx"] != d2["b_idx"]
        or d1["alpha_deg"] != pytest.approx(d2["alpha_deg"])
    )
    assert differs, "Two distinct seeds produced identical first draws (collision)"


# ---------------------------------------------------------------------------
# cli_main_identify --seed argument wiring
# ---------------------------------------------------------------------------


def test_cli_identify_accepts_seed_argument(tmp_path: Path) -> None:
    """--seed N is accepted and does not cause argparse errors.

    We use a minimal TOML that exercises the multi code path up to the
    point of loading the kernel, then expect a SystemExit from the missing
    kernel or a clean 0 return. Either way --seed must NOT cause an
    argparse error.
    """
    # Write a minimal multi config (no kernel available; the test gates on that)
    cfg_toml = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n"
        "[scan.phi]\nrange = 0.001\nsteps = 2\n\n"
        "[noise]\npoisson_noise = false\nrng_seed = 0\n\n"
        "[multi]\nn_samples = 1\n"
    )
    cfg_file = tmp_path / "id_multi.toml"
    cfg_file.write_text(cfg_toml, encoding="utf-8")
    out_dir = tmp_path / "out"

    # Either completes (if kernel is present), raises SystemExit from the
    # missing kernel, or raises KeyError / Exception from _load_resolution.
    # All are fine — what must NOT happen is argparse SystemExit(2).
    try:
        rc = cli_main_identify(
            [
                "--config",
                str(cfg_file),
                "--output",
                str(out_dir),
                "--mode",
                "multi",
                "--seed",
                "7",
            ]
        )
        # If it reaches here the kernel was found and the run succeeded
        assert rc == 0
    except SystemExit as exc:
        raise AssertionError("argparse rejected --seed") from exc
    except Exception:
        # KeyError from missing kernel, etc. — acceptable; the point is that
        # argparse passed without error (we got past parse_args).
        pass


def test_cli_identify_seed_overrides_config_rng_seed(tmp_path: Path) -> None:
    """--seed overrides noise.rng_seed in the loaded IdentificationConfig.

    We monkey-patch run_identification to capture the config it receives,
    then verify noise.rng_seed was overridden by --seed.
    """
    import dfxm_geo.pipeline as pipeline_mod

    cfg_toml = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n"
        "[scan.phi]\nrange = 0.001\nsteps = 2\n\n"
        "[noise]\npoisson_noise = false\nrng_seed = 0\n\n"
        "[multi]\nn_samples = 1\n"
    )
    cfg_file = tmp_path / "id_multi.toml"
    cfg_file.write_text(cfg_toml, encoding="utf-8")
    out_dir = tmp_path / "out"

    captured: list[IdentificationConfig] = []

    def fake_run(config: IdentificationConfig, output_dir):  # type: ignore[override]
        captured.append(config)
        # Return a valid result dict to avoid KeyError in cli_main_identify
        return {"n_samples": 0, "output_dir": output_dir, "master_path": output_dir / "x.h5"}

    original = pipeline_mod.run_identification
    pipeline_mod.run_identification = fake_run  # type: ignore[assignment]
    try:
        cli_main_identify(
            [
                "--config",
                str(cfg_file),
                "--output",
                str(out_dir),
                "--mode",
                "multi",
                "--seed",
                "42",
            ]
        )
    finally:
        pipeline_mod.run_identification = original

    assert len(captured) == 1, "run_identification was not called"
    assert captured[0].noise.rng_seed == 42, (
        f"Expected rng_seed=42 but got {captured[0].noise.rng_seed}"
    )


def test_cli_identify_seed_none_leaves_config_rng_seed(tmp_path: Path) -> None:
    """Without --seed the config's own rng_seed is preserved."""
    import dfxm_geo.pipeline as pipeline_mod

    cfg_toml = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n\n"
        "[scan.phi]\nrange = 0.001\nsteps = 2\n\n"
        "[noise]\npoisson_noise = false\nrng_seed = 77\n\n"
        "[multi]\nn_samples = 1\n"
    )
    cfg_file = tmp_path / "id_multi.toml"
    cfg_file.write_text(cfg_toml, encoding="utf-8")
    out_dir = tmp_path / "out"

    captured: list[IdentificationConfig] = []

    def fake_run(config: IdentificationConfig, output_dir):  # type: ignore[override]
        captured.append(config)
        return {"n_samples": 0, "output_dir": output_dir, "master_path": output_dir / "x.h5"}

    original = pipeline_mod.run_identification
    pipeline_mod.run_identification = fake_run  # type: ignore[assignment]
    try:
        cli_main_identify(
            [
                "--config",
                str(cfg_file),
                "--output",
                str(out_dir),
                "--mode",
                "multi",
                # no --seed
            ]
        )
    finally:
        pipeline_mod.run_identification = original

    assert len(captured) == 1
    assert captured[0].noise.rng_seed == 77, (
        f"Expected rng_seed=77 (from config) but got {captured[0].noise.rng_seed}"
    )
