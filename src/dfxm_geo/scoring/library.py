# src/dfxm_geo/scoring/library.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from .types import CandidateLabel, CandidateLibrary, GridSpec

_DET_LINK = "instrument/dfxm_sim_detector/data"

PathLike = str | Path


def _expand_paths(paths: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return sorted(p.glob("**/dfxm_identify.h5"))
        return [p]
    out: list[Path] = []
    for item in paths:
        out.extend(_expand_paths(item))
    return out


def _identify_entries(h: h5py.File) -> list[str]:
    entries = [k for k in h if k[:1].isdigit() and k.endswith(".1")]
    return sorted(entries, key=lambda k: int(k.split(".")[0]))


def _read_label(h: h5py.File, entry: str, fp: Path) -> CandidateLabel:
    s = h[entry]["sample"]
    g = h[entry]["dfxm_geo"]
    return CandidateLabel(
        slip_plane_normal=cast(
            "tuple[int, int, int]",
            tuple(int(x) for x in np.asarray(s["slip_plane_normal"])),
        ),
        burgers=cast(
            "tuple[int, int, int]",
            tuple(int(x) for x in np.asarray(s["burgers"])),
        ),
        rotation_deg=float(s["rotation_deg"][()]),
        gb_cos=float(g["gb_cos"][()]),
        gb_visible=bool(int(g["gb_visible"][()])),
        q_hkl=cast(
            "tuple[float, float, float]",
            tuple(float(x) for x in np.asarray(g["q_hkl"])),
        ),
        scan_index=int(entry.split(".")[0]),
        source_file=str(fp),
    )


def _read_frame(h: h5py.File, entry: str, reduction: str) -> np.ndarray:
    data = np.asarray(h[entry][_DET_LINK], dtype=np.float64)
    if data.ndim == 2:
        return data
    n = data.shape[0]
    if n == 1:
        return data[0]
    if reduction in ("auto", "max"):
        return data.max(axis=0)
    if reduction == "mean":
        return data.mean(axis=0)
    if reduction == "single":
        raise ValueError(f"{entry} has {n} frames but frame_reduction='single'")
    raise ValueError(f"unknown frame_reduction: {reduction!r}")


def _read_grid(h: h5py.File, entry: str, shape: tuple[int, ...]) -> GridSpec:
    g = h[entry]["dfxm_geo"]
    if "psize" not in g:
        raise ValueError(f"{entry}/dfxm_geo lacks 'psize'; cannot derive object-plane grid")
    pitch_um = round(
        float(g["psize"][()]) * 1.0e6, 9
    )  # metres -> micrometres (square object-plane pixels)
    return GridSpec(pitch_um=(pitch_um, pitch_um), shape=(int(shape[-2]), int(shape[-1])))


def load_library(
    paths: PathLike | Sequence[PathLike],
    *,
    include_invisible: bool = False,
    frame_reduction: str = "auto",
) -> CandidateLibrary:
    """Load candidate frames + labels from one or more identify masters."""
    files = _expand_paths(paths)
    if not files:
        raise ValueError(f"no identify masters found at {paths!r}")
    frames: list[np.ndarray] = []
    labels: list[CandidateLabel] = []
    grid: GridSpec | None = None
    for fp in files:
        with h5py.File(fp, "r") as h:
            for entry in _identify_entries(h):
                label = _read_label(h, entry, fp)
                if not include_invisible and not label.gb_visible:
                    continue
                arr = _read_frame(h, entry, frame_reduction)
                this_grid = _read_grid(h, entry, arr.shape)
                if grid is None:
                    grid = this_grid
                elif this_grid != grid:
                    raise ValueError(f"non-uniform grid at {fp}:{entry}: {this_grid} vs {grid}")
                frames.append(arr.astype(np.float32))
                labels.append(label)
    if not frames:
        raise ValueError("library is empty (no visible candidates after filtering)")
    assert grid is not None
    return CandidateLibrary(frames=np.stack(frames), labels=labels, grid=grid)
