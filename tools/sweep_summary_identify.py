#!/usr/bin/env python
"""Summarise a completed identify-sweep output tree.

Walks the sweep output, tallies per-scene label distributions, and saves
up to ``--n-frames`` example detector images as PNGs for the presentation.

Usage (the default glob finds both the single-node fanout and array layouts)::

    python tools/sweep_summary_identify.py --out presentation_assets --n-frames 8

For a rocking scan, each scene holds ``phi_steps`` frames; the saved example
PNG is the middle (~peak) frame, and ``summary.json`` reports both ``n_scenes``
and ``n_images`` (= total frames = scenes x phi_steps).

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
    """Read a single 2-D detector frame from a scene group, or None on error.

    A rocking scan stores ``(n_frames, H, W)``; pick the middle (~peak) frame
    so a multi-frame scene still yields a 2-D image that ``imsave`` can write.
    """
    try:
        data = np.asarray(scene["instrument/dfxm_sim_detector/data"][()], dtype=float)
    except (KeyError, OSError):
        return None
    data = np.squeeze(data)
    if data.ndim == 3:  # (n_frames, H, W) rocking scan -> middle frame
        data = data[data.shape[0] // 2]
    if data.ndim != 2:
        return None
    return data


def _scene_n_frames(scene: h5py.Group) -> int:
    """Detector frame count for a scene (1 for fixed phi, N for an N-step rock)."""
    try:
        shape = scene["instrument/dfxm_sim_detector/data"].shape
    except (KeyError, OSError):
        return 0
    return int(shape[0]) if len(shape) >= 3 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        default="output/**/dfxm_identify.h5",
        help="Glob to find master HDF5 files (recursive). Matches both the "
        "single-node fanout (output/fanout_*/) and array (output/idsweep_*/) layouts.",
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

    n_scenes = 0
    n_frames_total = 0
    n_frames_saved = 0
    all_rotations: list[float] = []
    slip_plane_counter: Counter[str] = Counter()

    for master_path in master_files:
        try:
            with h5py.File(master_path, "r") as f:
                scenes = _iter_scenes(f)

                for scene_key in scenes:
                    scene = f[scene_key]
                    n_scenes += 1
                    n_frames_total += _scene_n_frames(scene)

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
        "n_scenes": n_scenes,
        "n_images": n_frames_total,  # total detector frames (scenes x phi_steps)
        "dislocations_per_frame": 2,
        "slip_plane_counts": dict(slip_plane_counter),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"summary: {len(master_files)} configs, {n_scenes} scenes, "
        f"{n_frames_total} images (frames), {len(all_rotations)} rotation samples, "
        f"{n_frames_saved} frames saved"
    )
    print(f"  -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
