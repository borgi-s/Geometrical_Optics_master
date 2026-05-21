"""Sidecar JSON writer for random_dislocations realized parameters.

Sub-project C: when a forward run uses `mode="random_dislocations"`,
the realized (positions, Ud) per dislocation are written to a sidecar
JSON file next to the HDF5 output so users (notably ML-training
consumers) can recover the random draw without re-running.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_random_dislocations_sidecar(
    output_stem: Path,
    metadata: dict,
) -> Path:
    """Serialize realized random_dislocations params to JSON.

    Args:
        output_stem: Path stem (no `_random_dislocations.json` suffix).
            E.g. `/runs/2026-05-21/dfxm_geo` -> writes
            `/runs/2026-05-21/dfxm_geo_random_dislocations.json`.
        metadata: Dict produced by `build_dislocation_population` for
            `mode="random_dislocations"`. See the spec for schema.

    Returns:
        The path the JSON was written to.
    """
    out_path = output_stem.with_name(output_stem.name + "_random_dislocations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=False)
    return out_path
