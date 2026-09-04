"""Built-in recognizers, and the base classes for writing your own.

Order matters: the engine runs recognizers in the order given by
:func:`default_recognizers`, and each one can see what earlier ones decided.
Posture is resolved before motion so that jump detection can mute itself around
a squat.
"""

from .arms import ArmsRecognizer
from .base import (
    PredicateRecognizer,
    Recognizer,
    Sink,
    margin,
    margin_at_least,
    min_margin,
    nanmax,
    nanmean,
)
from .hands import HandsRecognizer
from .motion import MotionRecognizer
from .posture import PostureRecognizer

__all__ = [
    "ArmsRecognizer",
    "HandsRecognizer",
    "MotionRecognizer",
    "PostureRecognizer",
    "PredicateRecognizer",
    "Recognizer",
    "Sink",
    "default_recognizers",
    "margin",
    "margin_at_least",
    "min_margin",
    "nanmax",
    "nanmean",
]


def default_recognizers() -> list[Recognizer]:
    """A fresh set of the built-in recognizers, in dependency order."""
    return [ArmsRecognizer(), PostureRecognizer(), MotionRecognizer(), HandsRecognizer()]
