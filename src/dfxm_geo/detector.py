"""Realistic detector model: measured pco_ff noise + ADU conversion.

Converts sampling-normalized forward-model intensity into uint16 ADU frames
that are statistically indistinguishable from raw ID03 PCO Edge 4.2 bi
(scintillator-coupled sCMOS) frames.

Calibration provenance — fitted 2026-06-12 from Sina's ID03 beamtime data
(true darks + 61-frame photon-transfer sweep); method and fit figures in
docs/detector-noise-model.md, scripts in docs/calibration/, design in
docs/superpowers/specs/2026-06-12-detector-noise-model-design.md.
"""

from dataclasses import dataclass

import numpy as np

FULL_WELL = 65535  # uint16 saturation, matches the real LIMA frames


@dataclass(frozen=True)
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


def resolve_model(name: str) -> "DetectorModel | None":
    """Map a config model name to a DetectorModel; ``"ideal"`` → None."""
    if name == "ideal":
        return None
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown detector model {name!r}; expected 'ideal' or one of {sorted(_MODELS)}"
        ) from None
