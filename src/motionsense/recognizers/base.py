"""Recognizer protocol and the per-frame output sink."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

from ..catalog import register
from ..config import Tuning
from ..features import BodyFeatures
from ..mathx import SchmittGate
from ..types import ActivityDef, Signal, Trigger

__all__ = [
    "PredicateRecognizer",
    "Recognizer",
    "Sink",
    "margin",
    "margin_at_least",
    "min_margin",
    "nanmax",
    "nanmean",
]


class Sink:
    """Collects one frame's recognizer output.

    Recognizers run in a fixed order and each can see what earlier ones decided
    via :meth:`is_active`. That is how mutually-informed rules stay simple --
    the jump detector suppresses itself while a squat is held rather than
    re-deriving posture.
    """

    __slots__ = ("levels", "edges", "enabled")

    def __init__(self, enabled: frozenset[str] | None = None):
        self.levels: dict[str, tuple[bool, float, dict]] = {}
        self.edges: list[tuple[str, float, dict]] = []
        self.enabled = enabled

    def wants(self, activity_id: str) -> bool:
        return self.enabled is None or activity_id in self.enabled

    def level(self, activity_id: str, active: bool, confidence: float = 1.0, **data: Any) -> None:
        if self.enabled is not None and activity_id not in self.enabled:
            return
        self.levels[activity_id] = (bool(active), _clamp01(confidence), data)

    def edge(self, activity_id: str, confidence: float = 1.0, **data: Any) -> None:
        if self.enabled is not None and activity_id not in self.enabled:
            return
        self.edges.append((activity_id, _clamp01(confidence), data))

    def is_active(self, activity_id: str) -> bool:
        entry = self.levels.get(activity_id)
        return bool(entry and entry[0])

    def active_ids(self) -> frozenset[str]:
        return frozenset(k for k, v in self.levels.items() if v[0])


class Recognizer(ABC):
    """Turns :class:`~motionsense.features.BodyFeatures` into activity output.

    Subclass this to add activities the SDK does not ship with, then register the
    instance with :meth:`~motionsense.engine.MotionEngine.add_recognizer`. The
    engine handles threading, model selection and event dispatch; a recognizer
    only has to be a pure-ish function of features plus its own state.
    """

    #: Activities this recognizer can produce. Registered into the catalog on attach.
    activities: tuple[ActivityDef, ...] = ()

    def signals(self) -> frozenset[Signal]:
        """Inputs this recognizer needs, unioned over its activities."""
        needed: set[Signal] = set()
        for a in self.activities:
            needed |= a.requires
        return frozenset(needed)

    def configure(self, tuning: Tuning) -> None:
        """Called when the engine starts, before the first frame."""

    @abstractmethod
    def update(self, f: BodyFeatures, out: Sink) -> None:
        """Inspect one frame and report to ``out``."""

    def reset(self) -> None:
        """Drop temporal state. Called when tracking is lost or the engine stops."""

    def on_tracking_lost(self, t: float, out: Sink) -> None:
        """Called on frames with no detectable body. Default: release everything."""
        for a in self.activities:
            if a.is_level:
                out.level(a.id, False, 0.0)


class PredicateRecognizer(Recognizer):
    """Wraps a plain function into a fully-behaved activity.

    This is the low-ceremony extension path::

        engine.define(
            "reach_forward",
            lambda f: f.left_hand_rise,
            trigger="level", enter=-0.1, exit=-0.2,
        )

    The function receives :class:`BodyFeatures` and returns a number (compared
    against ``enter``/``exit`` with hysteresis) or a bool. Returning ``NaN``
    means "cannot tell from this frame", which releases the activity rather than
    latching it. Custom activities get the same debouncing, visibility handling
    and event semantics as the built-ins.
    """

    def __init__(
        self,
        activity: ActivityDef,
        fn: Callable[[BodyFeatures], Any],
        *,
        enter: float = 0.5,
        exit: float = 0.4999,
        min_on: float | None = None,
        min_off: float | None = None,
        cooldown: float = 0.4,
    ):
        self.activities = (activity,)
        self._def = activity
        self._fn = fn
        self._enter = enter
        self._exit = exit
        self._min_on = min_on
        self._min_off = min_off
        self._cooldown = cooldown
        self._gate: SchmittGate | None = None
        self._last_fire = float("-inf")

    def configure(self, tuning: Tuning) -> None:
        self._gate = SchmittGate(
            self._enter,
            self._exit,
            self._min_on if self._min_on is not None else tuning.level_min_on,
            self._min_off if self._min_off is not None else tuning.level_min_off,
        )

    def reset(self) -> None:
        if self._gate is not None:
            self._gate.reset()
        self._last_fire = float("-inf")

    def update(self, f: BodyFeatures, out: Sink) -> None:
        if self._gate is None:
            self.configure(Tuning())
        raw = self._fn(f)
        value = float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)
        state = self._gate.update(value, f.t)

        if self._def.trigger is Trigger.LEVEL:
            out.level(self._def.id, state, self._gate.confidence, value=value)
        elif state and f.t - self._last_fire >= self._cooldown:
            # Edge activities fire on the gate's rising transition only.
            if self._gate.changed_at == f.t:
                self._last_fire = f.t
                out.edge(self._def.id, self._gate.confidence, value=value)


# -- helpers shared by the built-in recognizers ------------------------------


def margin(value: float, limit: float, width: float) -> float:
    """Normalised distance inside a limit: 1.0 well within, 0.0 at it, negative past.

    Putting every constraint on this common scale is what lets a multi-condition
    pose (a T-pose needs four things at once) collapse into one number that can
    drive a single hysteresis gate and yield a meaningful confidence.
    """
    if math.isnan(value):
        return float("nan")
    if width <= 0.0:
        return 1.0 if value <= limit else -1.0
    return (limit - value) / width


def margin_at_least(value: float, limit: float, width: float) -> float:
    """Mirror of :func:`margin` for constraints of the form ``value >= limit``."""
    if math.isnan(value):
        return float("nan")
    if width <= 0.0:
        return 1.0 if value >= limit else -1.0
    return (value - limit) / width


def nanmean(*values: float) -> float:
    """Mean of the non-NaN arguments, or NaN if there are none.

    Used to pool the left and right sides of a symmetric measurement: if one leg
    is occluded the other still carries the signal, at reduced confidence,
    instead of the whole feature going dark.
    """
    total = 0.0
    count = 0
    for v in values:
        if not math.isnan(v):
            total += v
            count += 1
    return total / count if count else float("nan")


def nanmax(*values: float) -> float:
    """Largest non-NaN argument, or NaN if there are none.

    For fusing cues that each fail *toward zero* in the view where they are
    blind: taking the max means "whichever cue can currently see this".
    """
    best = float("-inf")
    for v in values:
        if not math.isnan(v) and v > best:
            best = v
    return best if best != float("-inf") else float("nan")


def min_margin(*values: float) -> float:
    """Weakest of several margins, propagating NaN.

    ``min()`` cannot be used directly: Python's ``min`` returns NaN or not
    depending on argument order, because every comparison with NaN is False.
    A pose whose constraints are unmeasurable must read as unknown, not as
    satisfied.
    """
    worst = float("inf")
    for v in values:
        if math.isnan(v):
            return float("nan")
        if v < worst:
            worst = v
    return worst if worst != float("inf") else float("nan")


def register_all(activities: Iterable[ActivityDef]) -> None:
    for a in activities:
        try:
            register(a)
        except ValueError:
            pass  # already known (the built-in catalog, or a re-attach)


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
