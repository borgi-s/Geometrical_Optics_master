"""dfxm-forward / dfxm-identify argparse entry points — extracted from pipeline.py
(refactor gate, 2026-06-11). pyproject targets the pipeline facade, which re-exports
these.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for ``dfxm-forward`` and ``python scripts/run_forward.py``.

    Default behavior: run simulation, then post-processing.
    """
    # Lazy import: pipeline (the facade) re-imports this module, so a
    # module-top import would be circular. Resolving at call time keeps
    # `monkeypatch.setattr("dfxm_geo.pipeline.run_simulation", ...)` working
    # — cli routes ALL pipeline names through the facade at call time.
    import dfxm_geo.pipeline as _pipeline  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="DFXM forward simulation")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Run simulation only; skip post-processing (Phase 6 behavior).",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip simulation; run post-processing against an existing output dir.",
    )
    args = parser.parse_args(argv)

    config = _pipeline.SimulationConfig.from_toml(args.config)

    if args.postprocess_only:
        _pipeline.run_postprocess(args.output, config)
    else:
        _pipeline.run_simulation(config, args.output)
        if config.postprocess.enabled and not args.no_postprocess:
            if config.reflections:
                # run_postprocess requires config.reciprocal.hkl (not set for
                # multi-reflection configs) and calls _load_resolution / build_forward_context
                # using the top-level config rather than per-run geometry.  Looping it
                # per-subdir would produce wrong geometry for every reflection.
                # Per-reflection figures are a Task 7 / polish item (M3 plan 2).
                print(
                    "postprocess: skipped for multi-reflection runs "
                    "(per-reflection figures land with plan-2 polish)",
                    file=sys.stderr,
                )
            else:
                _pipeline.run_postprocess(args.output, config)
    return 0


def cli_main_identify(argv: list[str] | None = None) -> int:
    """Argparse-driven entry point for `dfxm-identify`."""
    # Lazy import: pipeline (the facade) re-imports this module, so a
    # module-top import would be circular. Resolving at call time keeps
    # `monkeypatch.setattr("dfxm_geo.pipeline.run_identification", ...)` working.
    import dfxm_geo.pipeline as _pipeline  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="DFXM dislocation identification simulation")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to identification TOML config"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "z-scan"],
        default=None,
        help="Override the config's mode field",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config's [detector] rng_seed (integer). Useful for per-task "
        "reproducibility in array jobs (e.g. --seed $LSB_JOBINDEX).",
    )
    args = parser.parse_args(argv)

    cfg = _pipeline.load_identification_config(args.config)
    if args.mode is not None and args.mode != cfg.mode:
        cfg = replace(cfg, mode=args.mode)
        cfg.__post_init__()  # re-run validation

    if args.seed is not None:
        cfg = replace(cfg, detector=replace(cfg.detector, rng_seed=args.seed))
        cfg.__post_init__()  # re-run validation

    result = _pipeline.run_identification(cfg, args.output)
    count_key, noun = {
        "single": ("n_images", "images"),
        "multi": ("n_samples", "samples"),
        "z-scan": ("n_configurations", "configurations"),
    }[cfg.mode]
    if "n_reflections" in result:
        # Multi-reflection return shape: per-reflection results nested under
        # "reflections" (one reflection_NNN/ master each) + a super-master.
        total = sum(r[count_key] for r in result["reflections"])
        print(
            f"Wrote {total} {noun} across {result['n_reflections']} reflections "
            f"to {args.output} (per-reflection masters in reflection_NNN/)"
        )
    else:
        print(f"Wrote {result[count_key]} {noun} to {result['output_dir']}")
    return 0
