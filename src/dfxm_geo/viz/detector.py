"""Detector-frame display helpers.

The weak-beam reduction implemented here mirrors standard experimental
post-processing of DFXM detector images: estimate the background level as
``mean + k*std`` over the frame and clamp what remains at zero. Validated
against ID03 pco_ff weak-beam frames (subtracts ~117 ADU at 1 s exposure);
see docs/detector-noise-model.md. In strong-beam condition this zeros
nearly the whole image by construction — the GO model is weak-beam-only.
"""

import numpy as np


def subtract_background(image: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Subtract ``mean + k*std`` and floor at zero. Returns a new float array.

    Works on a single frame or a stack; statistics are computed over the
    whole input (per-run background, matching how the reduction is applied
    at the beamline viewer).
    """
    img = np.asarray(image, dtype=np.float64)
    threshold = img.mean() + k * img.std()
    return np.clip(img - threshold, 0.0, None)
