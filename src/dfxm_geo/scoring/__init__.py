"""Cross-correlation dislocation identification scorer."""

from .engine import cross_correlation_peak, preprocess, score_matrix, score_target
from .identify import IdentifiabilityResult, Identifier, RankedMatch
from .library import load_library
from .target import resample_to_grid
from .types import CandidateLabel, CandidateLibrary, GridSpec

__all__ = [
    "GridSpec",
    "CandidateLabel",
    "CandidateLibrary",
    "preprocess",
    "cross_correlation_peak",
    "score_matrix",
    "score_target",
    "resample_to_grid",
    "load_library",
    "Identifier",
    "IdentifiabilityResult",
    "RankedMatch",
]
