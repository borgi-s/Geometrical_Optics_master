"""dfxm_geo.pipeline — stable facade over the split modules (refactor gate, 2026-06-11).

The 2700-line pipeline module was split into:
  dfxm_geo.config        — config dataclasses, TOML loaders/serializers, run_theta
  dfxm_geo.orchestrator  — run_simulation / run_identification + helpers
  dfxm_geo.cli           — dfxm-forward / dfxm-identify argparse entry points

This module re-exports the complete historical surface (public AND underscore
names, plus the `fm` module alias) so every existing import path, pyproject
entry point, fanout template, and notebook keeps working. New code should
import from the specific module; tests that monkeypatch orchestration
internals patch `dfxm_geo.orchestrator.*` (the module where the bare-name
call sites execute) — patching the facade binding does NOT reach them.
"""

from __future__ import annotations

import sys

# --- refactor gate (2026-06-11): CLI entry points extracted to dfxm_geo.cli.
# pipeline remains the stable facade: import cli names from here as before.
from dfxm_geo.cli import cli_main, cli_main_identify  # noqa: F401

# --- refactor gate (2026-06-11): config layer extracted to dfxm_geo.config.
# pipeline remains the stable facade: import config names from here as before.
from dfxm_geo.config import (  # noqa: F401
    _AXIS_TO_LABEL,
    _CANONICAL_AXES,
    _CRYSTAL_MODE_NAMES,
    AxisScanConfig,
    CenteredCrystalConfig,
    CrystalConfig,
    DetectorConfig,
    DetectorGeometryConfig,
    GeometryConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationZScanConfig,
    IOConfig,
    PostprocessConfig,
    RandomDislocationsConfig,
    ReciprocalConfig,
    ScanConfig,
    ScanFrames,
    SimulationConfig,
    WallCrystalConfig,
    _build_geometry_config,
    _dataclass_to_toml_str,
    _identification_config_to_toml_str,
    _parse_reflections_tables,
    load_identification_config,
    run_theta,
)

# --- refactor gate (2026-06-11): run orchestration extracted to
# dfxm_geo.orchestrator. pipeline remains the stable facade: import orchestration
# names (public AND underscore, plus the `fm` module alias and the io.hdf5
# writers tests reach through the facade) from here as before. Tests that
# monkeypatch orchestration internals patch `dfxm_geo.orchestrator.*`.
from dfxm_geo.orchestrator import (  # noqa: F401
    _ALL_111_PLANES,
    _KERNEL_CTX_CACHE,
    _apply_detector_model,
    _build_dislocation_sample_entry,
    _build_scan_frames,
    _build_scan_frames_at_z,
    _context_for_run,
    _dispatch_identification,
    _draw_dislocation,
    _identify_geometry_attrs,
    _identify_title,
    _iter_identification_multi,
    _iter_identification_single,
    _iter_identification_zscan,
    _iterate_simulation_frames,
    _load_resolution,
    _lookup_and_load_kernel,
    _passes_invisibility,
    _positioners_for_scan_frames,
    _q_unit,
    _resolution_for_run,
    _resolve_postprocess_Hg,
    _run_identification_multi,
    _run_identification_single,
    _run_identification_zscan,
    _run_simulation_inner,
    _scan_frames_args,
    find_hg_scene,
    fm,
    run_identification,
    run_postprocess,
    run_simulation,
    write_identification_h5,
    write_multi_reflection_master,
    write_simulation_h5,
)

if __name__ == "__main__":
    sys.exit(cli_main())
