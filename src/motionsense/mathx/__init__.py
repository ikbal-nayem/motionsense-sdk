"""Numerical building blocks shared by the feature extractor and recognizers.

Nothing in here imports MediaPipe or OpenCV, so the signal-processing layer can
be unit tested (and reused) without a camera or a model download.
"""

from .filters import ExpSmoother, OneEuroFilter, alpha_for_cutoff, alpha_for_tau
from .gates import SchmittGate
from .geometry import angle_at, angle_at3, direction, procrustes_scale, signed_angle_from_up
from .oscillation import OscillationDetector, OscillationResult
from .series import RingSeries, WindowedSlope

__all__ = [
    "ExpSmoother",
    "OneEuroFilter",
    "OscillationDetector",
    "OscillationResult",
    "RingSeries",
    "SchmittGate",
    "WindowedSlope",
    "alpha_for_cutoff",
    "alpha_for_tau",
    "angle_at",
    "angle_at3",
    "direction",
    "procrustes_scale",
    "signed_angle_from_up",
]
