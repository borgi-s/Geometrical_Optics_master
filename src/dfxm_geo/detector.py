"""Realistic detector model: measured pco_ff noise + ADU conversion.

Converts sampling-normalized forward-model intensity into uint16 ADU frames
that are statistically indistinguishable from raw ID03 PCO Edge 4.2 bi
(scintillator-coupled sCMOS) frames.

Calibration provenance — fitted 2026-06-12 from Sina's ID03 beamtime data
(true darks + 61-frame photon-transfer sweep); method and fit figures in
docs/detector-noise-model.md, scripts in docs/calibration/, design in
docs/superpowers/specs/2026-06-12-detector-noise-model-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

FULL_WELL: Final[int] = 65535  # uint16 saturation, matches the real LIMA frames

# Maximum number of frames processed in a single DetectorModel.apply() call.
# At (510, 170) pixels per frame, 256 frames ≈ 112 MB float64 -- well under
# typical RAM limits. Reducing this from n_frames to chunk_frames keeps the
# peak float64 footprint bounded regardless of how large a scan stack grows.
# (A 3721-frame 61×61 config at (510, 170) would otherwise allocate ~2.4 GiB.)
DETECTOR_APPLY_CHUNK_FRAMES: Final[int] = 256


# eq=False: the fpn_offset ndarray field would make an auto-generated __eq__
# raise on truth-value ambiguity; identity equality is what we want (constructed once per run).
@dataclass(frozen=True, kw_only=True, eq=False)
class SensorMap:
    """Per-pixel fixed-pattern state for one synthetic 'camera'.

    Generated once per run (same map for every scan and reflection — it is
    *fixed*-pattern), reproducible from the config seed.
    """

    fpn_offset: np.ndarray  # (ny, nx) float64 ADU, added to the time-dependent offset


@dataclass(frozen=True, kw_only=True)
class DetectorModel:
    """Noise/conversion parameters of a physical detector. All ADU units."""

    name: str
    offset_base: float  # ADU at t=0
    dark_rate: float  # ADU per second of exposure
    read_noise_var_base: float  # ADU^2 at t=0
    read_noise_var_rate: float  # ADU^2 per second (dark shot noise)
    gain: float  # ADU per incident photon-equivalent
    fpn_sigma: float  # Gaussian core of the fixed-pattern offset
    tail_fraction: float  # fraction of warm/hot pixels (exponential tail)
    tail_scale: float  # ADU, exponential tail scale
    edge_rows: int  # rows at each frame edge with elevated offset
    edge_peak: float  # ADU boost at the outermost row
    edge_decay: float  # e-folding of the boost, in rows

    def offset(self, exposure_time: float) -> float:
        """Mean dark level in ADU at the given exposure."""
        return self.offset_base + self.dark_rate * exposure_time

    def noise_sigma(self, exposure_time: float) -> float:
        """Temporal (read + dark shot) noise sigma in ADU at the given exposure."""
        return float(np.sqrt(self.read_noise_var_base + self.read_noise_var_rate * exposure_time))

    def make_sensor_map(self, shape: tuple[int, int], rng: np.random.Generator) -> SensorMap:
        """Synthesize a fixed-pattern offset map by sampling the measured
        distributions (spec §3.1): Gaussian core + exponential warm/hot tail
        + elevated edge rows. Synthetic rather than the literal 2048^2
        calibration map so it scales to any simulated detector size.
        """
        ny, nx = shape
        fpn = rng.normal(0.0, self.fpn_sigma, size=shape)
        tail = rng.random(shape) < self.tail_fraction
        fpn[tail] += rng.exponential(self.tail_scale, size=int(tail.sum()))
        rows = np.arange(ny, dtype=np.float64)
        dist = np.minimum(rows, ny - 1 - rows)  # distance to nearest edge
        boost = self.edge_peak * np.exp(-dist / self.edge_decay)
        boost[dist >= self.edge_rows] = 0.0
        return SensorMap(fpn_offset=fpn + boost[:, None])

    def apply(
        self,
        ideal_adu: np.ndarray,
        exposure_time: float,
        rng: np.random.Generator,
        sensor: SensorMap,
    ) -> np.ndarray:
        """Ideal expected-signal ADU (above offset) → noisy uint16 frames.

        Noise composition (spec §2): gain-scaled Poisson on the photon
        count, plus the exposure-dependent offset, the fixed-pattern map,
        and Gaussian read/dark-shot noise; rounded and clamped to uint16.
        ``gain * Poisson(s / gain)`` reproduces the measured variance
        ``gain * s`` exactly (scintillator excess noise is inside the
        fitted gain by construction).
        """
        photons = np.clip(ideal_adu, 0.0, None) / self.gain
        signal = self.gain * rng.poisson(photons).astype(np.float64)
        noisy = (
            signal
            + self.offset(exposure_time)
            + sensor.fpn_offset
            + rng.normal(0.0, self.noise_sigma(exposure_time), size=ideal_adu.shape)
        )
        return np.clip(np.rint(noisy), 0, FULL_WELL).astype(np.uint16)


def apply_in_chunks(
    model: DetectorModel,
    ideal_adu: np.ndarray,
    exposure_time: float,
    rng: np.random.Generator,
    sensor: SensorMap,
    *,
    chunk_frames: int = DETECTOR_APPLY_CHUNK_FRAMES,
) -> np.ndarray:
    """Call ``model.apply()`` in frame-contiguous chunks, threading ``rng``.

    Produces byte-identical output to ``model.apply(ideal_adu, ...)`` because
    NumPy's PCG64 Generator advances its internal state sequentially in flat
    C-order. Feeding chunks [0:k], [k:2k], … through the same stateful ``rng``
    object is equivalent to one call over all frames: each Poisson/Normal draw
    pulls the next value from the same deterministic sequence.

    The caller must NOT re-seed or replace ``rng`` between chunks.

    Args:
        model: DetectorModel whose apply() is called per chunk.
        ideal_adu: float64 array of shape (n_frames, H, W).
        exposure_time: exposure in seconds (passed through to apply()).
        rng: stateful Generator; advanced in-place across chunks.
        sensor: SensorMap (fixed-pattern, shared across all chunks).
        chunk_frames: number of frames per chunk (default DETECTOR_APPLY_CHUNK_FRAMES).

    Returns:
        uint16 array of shape (n_frames, H, W), byte-identical to a single
        ``model.apply(ideal_adu, exposure_time, rng, sensor)`` call.
    """
    n_frames = ideal_adu.shape[0]
    if n_frames == 0:
        return np.empty((0,) + ideal_adu.shape[1:], dtype=np.uint16)
    out = np.empty_like(ideal_adu, dtype=np.uint16)
    for start in range(0, n_frames, chunk_frames):
        end = min(start + chunk_frames, n_frames)
        out[start:end] = model.apply(ideal_adu[start:end], exposure_time, rng, sensor)
    return out


PCO_EDGE_4P2_ID03 = DetectorModel(
    name="pco_edge_4.2_id03",
    offset_base=102.5,
    dark_rate=7.5,
    read_noise_var_base=6.3,
    read_noise_var_rate=11.0,
    gain=2.14,
    fpn_sigma=1.8,
    tail_fraction=0.015,
    tail_scale=10.0,
    edge_rows=16,
    edge_peak=20.0,
    edge_decay=4.0,
)

_MODELS = {PCO_EDGE_4P2_ID03.name: PCO_EDGE_4P2_ID03}


def resolve_model(name: str) -> DetectorModel | None:
    """Map a config model name to a DetectorModel; ``"ideal"`` → None."""
    if name == "ideal":
        return None
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown detector model {name!r}; expected 'ideal' or one of {sorted(_MODELS)}"
        ) from None
