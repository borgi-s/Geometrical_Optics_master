#!/usr/bin/env python
"""Summarise a completed identify-sweep output tree.

Walks the sweep output, tallies per-scene label distributions, and saves
up to ``--n-frames`` example detector images as PNGs for the presentation.

Usage::

    python tools/sweep_summary_identify.py \
        --glob "output/idsweep_*/**/dfxm_identify.h5" \
        --out presentation_assets \
        --n-frames 8

Outputs (all under ``--out``):

* ``frames/frame_NNNN.png``   — up to ``--n-frames`` detector images
* ``rotations.npy``           — 1-D array of all sampled ``rotation_deg`` values
* ``summary.json``            — headline counts + label histograms
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("h5py is required: pip install h5py") from e

_SCENE_RE = re.compile(r"^\d+\.1$")


def _iter_scenes(f: h5py.File) -> list[str]:
    """Return scene group names matching the ``N.1`` pattern."""
    return [k for k in f if _SCENE_RE.fullmatch(k)]


def _read_frame(scene: h5py.Group) -> np.ndarray | None:
    """Read the detector frame from a scene group, or return None on error."""
    try:
        data = scene["instrument/dfxm_sim_detector/data"][()]
        return np.squeeze(np.asarray(data, dtype=float))
    except (KeyError, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        default="output/idsweep_*/**/dfxm_identify.h5",
        help="Glob pattern to find master HDF5 files (recursive).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("presentation_assets"),
        help="Output directory for summary files and example frames.",
    )
    ap.add_argument(
        "--n-frames",
        type=int,
        default=8,
        help="Maximum number of example detector frames to save as PNG.",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Discover master HDF5 files.
    master_files = sorted(Path(".").glob(args.glob))
    if not master_files:
        print(f"No HDF5 files found matching: {args.glob}")
        return 1

    n_images = 0
    n_frames_saved = 0
    all_rotations: list[float] = []
    slip_plane_counter: Counter[str] = Counter()

    for master_path in master_files:
        try:
            with h5py.File(master_path, "r") as f:
                scenes = _iter_scenes(f)
                n_images += len(scenes)

                for scene_key in scenes:
                    scene = f[scene_key]

                    # Collect labels from all dislocations in this scene.
                    dislocations_grp = scene.get("sample/dislocations")
                    if dislocations_grp is not None:
                        for dis_key in dislocations_grp:
                            dis = dislocations_grp[dis_key]
                            # rotation_deg
                            try:
                                rot = float(np.asarray(dis["rotation_deg"]).ravel()[0])
                                all_rotations.append(rot)
                            except (KeyError, IndexError):
                                pass
                            # slip_plane_normal -> string key for counting
                            try:
                                spn = tuple(
                                    int(round(x))
                                    for x in np.asarray(dis["slip_plane_normal"]).ravel()
                                )
                                slip_plane_counter[str(spn)] += 1
                            except (KeyError, IndexError):
                                pass

                    # Save up to --n-frames example images.
                    if n_frames_saved < args.n_frames:
                        frame = _read_frame(scene)
                        if frame is not None:
                            png_path = frames_dir / f"frame_{n_frames_saved:04d}.png"
                            plt.imsave(str(png_path), frame, cmap="magma")
                            n_frames_saved += 1

        except OSError:
            print(f"  WARNING: could not open {master_path} — skipping")
            continue

    # Save rotations array.
    rotations_arr = np.array(all_rotations, dtype=float)
    np.save(str(out_dir / "rotations.npy"), rotations_arr)

    # Build and save summary JSON.
    summary = {
        "n_configs": len(master_files),
        "n_images": n_images,
        "dislocations_per_frame": 2,
        "slip_plane_counts": dict(slip_plane_counter),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"summary: {len(master_files)} configs, {n_images} images, "
        f"{len(all_rotations)} rotation samples, {n_frames_saved} frames saved"
    )
    print(f"  -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
