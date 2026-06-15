from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Object-plane sampling of a candidate/target image."""

    pitch_um: tuple[float, float]  # (dy, dx) micrometres per pixel
    shape: tuple[int, int]  # (H, W)


@dataclass(frozen=True)
class CandidateLabel:
    slip_plane_normal: tuple[int, int, int]
    burgers: tuple[int, int, int]
    rotation_deg: float
    gb_cos: float
    gb_visible: bool
    q_hkl: tuple[float, float, float]
    scan_index: int
    source_file: str

    def class_key(self, mode: str = "plane_burgers") -> tuple:
        if mode == "plane_burgers":
            return (self.slip_plane_normal, self.burgers)
        if mode == "burgers":
            return (self.burgers,)
        if mode == "plane_burgers_alpha":
            return (self.slip_plane_normal, self.burgers, self.rotation_deg)
        raise ValueError(f"unknown class_key mode: {mode!r}")


@dataclass
class CandidateLibrary:
    frames: np.ndarray  # (N, H, W) float32
    labels: list[CandidateLabel]
    grid: GridSpec

    def __len__(self) -> int:
        return len(self.labels)
