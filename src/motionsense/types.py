"""Public data types: activities, events, and per-frame results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "ActivityDef",
    "Event",
    "EventHandler",
    "FrameResult",
    "HandSample",
    "Phase",
    "Signal",
    "Snapshot",
    "Trigger",
]


class Trigger(str, Enum):
    """How an activity behaves over time."""

    #: True for as long as the condition holds (a pose). Produces START/UPDATE/END.
    LEVEL = "level"
    #: Fires once when a motion is recognised (a gesture). Produces TRIGGER.
    EDGE = "edge"


class Phase(str, Enum):
    """Which moment in an activity's life an event describes."""

    START = "start"
    UPDATE = "update"
    END = "end"
    TRIGGER = "trigger"


class Signal(str, Enum):
    """Input a recognizer needs. The engine uses this to decide which models to
    run: the hand model is roughly as expensive as the pose model, so it stays
    off unless something actually subscribes to a hand activity."""

    POSE = "pose"
    HANDS = "hands"


@dataclass(frozen=True, slots=True)
class ActivityDef:
    """Static description of something the SDK can recognise."""

    id: str
    name: str
    category: str
    trigger: Trigger
    description: str
    requires: frozenset[Signal] = frozenset({Signal.POSE})

    @property
    def is_level(self) -> bool:
        return self.trigger is Trigger.LEVEL

    @property
    def is_edge(self) -> bool:
        return self.trigger is Trigger.EDGE


@dataclass(frozen=True, slots=True)
class Event:
    """Delivered to listeners when an activity starts, updates, ends or fires."""

    #: Activity id, e.g. ``"jump"`` or ``"left_hand_up"``.
    activity: str
    phase: Phase
    #: Seconds since the engine started (monotonic).
    timestamp: float
    #: ``time.time()`` at the same moment, for logging and correlation.
    wall_time: float
    #: 0..1. For level activities, how far past the opening threshold the
    #: measurement sits; for gestures, how strongly the pattern matched.
    confidence: float = 1.0
    #: For ``Phase.END`` and ``Phase.UPDATE``, how long the activity has been held.
    duration: float = 0.0
    #: Recognizer-specific extras (e.g. ``{"peak_velocity": 2.4}`` on a jump).
    data: Mapping[str, Any] = field(default_factory=dict)
    frame_index: int = -1

    @property
    def definition(self) -> ActivityDef | None:
        from .catalog import get

        return get(self.activity)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.timestamp:7.3f}s] {self.activity}.{self.phase.value} ({self.confidence:.2f})"


EventHandler = Callable[[Event], None]


@dataclass(slots=True)
class HandSample:
    """One detected hand, in the canonical (subject-facing) coordinate frame."""

    #: ``"left"`` or ``"right"`` -- the subject's own hand, already corrected for
    #: mirroring.
    side: str
    #: ``(21, 3)`` array of x, y, z. x/y are aspect-corrected image coordinates.
    points: np.ndarray
    #: Detector confidence for the handedness classification.
    score: float = 1.0


@dataclass(slots=True)
class FrameResult:
    """Everything produced from a single frame.

    Handed to ``engine.on_frame`` listeners for preview rendering or custom
    analysis. ``image`` is the source frame in BGR; it is *not* copied, so treat
    it as read-only unless you copy it first.
    """

    index: int
    timestamp: float
    wall_time: float
    #: ``(33, 4)`` x, y, z, visibility in normalised image coordinates, or None.
    pose: np.ndarray | None
    hands: tuple[HandSample, ...]
    #: Activity ids currently held.
    active: frozenset[str]
    #: Events produced by this frame, in order.
    events: tuple[Event, ...]
    image: np.ndarray | None = None
    #: Seconds spent between receiving the frame and finishing dispatch.
    latency: float = 0.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Cheap, thread-safe view of engine state, for status bars and dashboards."""

    running: bool
    active: frozenset[str]
    fps: float
    #: Mean seconds from frame arrival to event dispatch.
    latency: float
    frames: int
    dropped: int
    tracking: bool
