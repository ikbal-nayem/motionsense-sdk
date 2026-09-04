"""Landmark provider protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..types import HandSample

__all__ = ["LandmarkProvider", "Landmarks"]


@dataclass(slots=True)
class Landmarks:
    """Raw detector output for one frame, already in canonical orientation.

    Canonical means: mirrored so that ``+x`` is the subject's right, and hand
    sides relabelled to match. Everything downstream can then assume the subject
    is facing it, whatever the camera did.
    """

    #: ``(33, 4)`` x, y, z, visibility in normalised image coordinates, or None.
    pose: np.ndarray | None = None
    #: ``(33, 4)`` metric coordinates with the origin between the hips, or None.
    #: Joint angles taken from these are view-independent.
    pose_world: np.ndarray | None = None
    hands: tuple[HandSample, ...] = field(default_factory=tuple)


class LandmarkProvider(ABC):
    """Detects body landmarks in a frame.

    Swap in your own to use a different pose model. Produce the 33-point
    BlazePose layout (see :mod:`motionsense.landmarks`) and every recognizer
    works unchanged.
    """

    @abstractmethod
    def process(self, image: np.ndarray, t: float, frame_index: int) -> Landmarks:
        """Detect landmarks in a BGR frame."""

    @abstractmethod
    def close(self) -> None:
        """Release models. Must be safe to call more than once."""

    def set_hands_enabled(self, enabled: bool) -> None:
        """Enable or disable the hand model. Ignored if unsupported."""
