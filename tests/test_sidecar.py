"""Tests for the random_dislocations sidecar JSON writer (sub-project C)."""

from __future__ import annotations

import json
from pathlib import Path

from dfxm_geo.io.sidecar import write_random_dislocations_sidecar


def test_sidecar_written_next_to_stem(tmp_path: Path) -> None:
    metadata = {
        "ndis": 2,
        "sigma_um": 5.1,
        "sigma_source": "default-fov",
        "min_distance_um": None,
        "seed": 42,
        "seed_source": "user",
        "dislocations": [
            {
                "index": 0,
                "x_um": 1.0,
                "y_um": 2.0,
                "z_um": 0.0,
                "b": [1, -1, 0],
                "n": [1, 1, 1],
                "t": [1, 1, -2],
            },
            {
                "index": 1,
                "x_um": -3.0,
                "y_um": 4.0,
                "z_um": 0.0,
                "b": [0, 1, -1],
                "n": [1, 1, 1],
                "t": [-2, 1, 1],
            },
        ],
    }
    stem = tmp_path / "dfxm_geo"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    assert out_path == stem.with_name("dfxm_geo_random_dislocations.json")
    assert out_path.exists()


def test_sidecar_round_trips_via_json(tmp_path: Path) -> None:
    metadata = {
        "ndis": 1,
        "sigma_um": 5.0,
        "sigma_source": "user",
        "min_distance_um": 2.0,
        "seed": 42,
        "seed_source": "user",
        "dislocations": [
            {
                "index": 0,
                "x_um": 0.5,
                "y_um": -0.5,
                "z_um": 0.0,
                "b": [1, -1, 0],
                "n": [1, 1, 1],
                "t": [1, 1, -2],
            },
        ],
    }
    stem = tmp_path / "run"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    with open(out_path) as fh:
        loaded = json.load(fh)
    assert loaded == metadata


def test_sidecar_pretty_printed(tmp_path: Path) -> None:
    metadata = {"ndis": 1, "dislocations": []}
    stem = tmp_path / "run"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    text = out_path.read_text()
    assert "\n" in text  # not single-line
    assert "  " in text  # indented
