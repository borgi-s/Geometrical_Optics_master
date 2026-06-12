"""[detector] config block parsing + [noise] rejection."""

import pytest

from dfxm_geo.config import DetectorConfig, load_identification_config
from dfxm_geo.pipeline import SimulationConfig

# Sub-project F made every key optional: an empty TOML parses to all-defaults
# for both load_identification_config and SimulationConfig.from_toml (see
# tests/test_empty_toml_runs.py). The empty string is therefore the smallest
# valid identification config; appending a [detector]/[noise] block to it
# exercises ONLY the detector parse path.
MINIMAL_IDENTIFY = ""


def test_detector_defaults_are_on_everywhere():
    cfg = DetectorConfig()
    assert cfg.model == "pco_edge_4.2_id03"
    assert cfg.exposure_time == 1.0
    assert cfg.rng_seed == 0
    assert cfg.counts_scale > 0


def test_detector_block_round_trips_from_toml(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(
        MINIMAL_IDENTIFY + '\n[detector]\nmodel = "ideal"\nexposure_time = 0.25\nrng_seed = 9\n'
    )
    cfg = load_identification_config(p)
    assert cfg.detector.model == "ideal"
    assert cfg.detector.exposure_time == 0.25
    assert cfg.detector.rng_seed == 9


def test_unknown_model_rejected(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(MINIMAL_IDENTIFY + '\n[detector]\nmodel = "bogus"\n')
    with pytest.raises(ValueError, match="unknown detector model"):
        load_identification_config(p)


def test_noise_block_rejected_with_pointer(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(MINIMAL_IDENTIFY + "\n[noise]\npoisson_noise = true\n")
    with pytest.raises(ValueError, match=r"\[noise\] was removed.*\[detector\]"):
        load_identification_config(p)


def test_simulation_config_also_parses_detector(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text('[detector]\nmodel = "ideal"\n')
    cfg = SimulationConfig.from_toml(p)
    assert cfg.detector.model == "ideal"


def test_simulation_config_rejects_noise_block(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[noise]\npoisson_noise = true\n")
    with pytest.raises(ValueError, match=r"\[noise\] was removed.*\[detector\]"):
        SimulationConfig.from_toml(p)
