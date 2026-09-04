"""Finger states from the hand model: fist open/closed, pinch."""

from __future__ import annotations

from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures
from ..mathx import SchmittGate
from .base import Recognizer, Sink

__all__ = ["HandsRecognizer"]

_IDS = (
    "left_fist_closed",
    "left_fist_open",
    "right_fist_closed",
    "right_fist_open",
    "left_pinch",
    "right_pinch",
)

_SIDES = ("left", "right")

NAN = float("nan")


class HandsRecognizer(Recognizer):
    """Per-hand finger states.

    Open and closed are separate gates sharing one hysteresis band, so between
    them there is a dead zone where neither is reported. That is intentional: a
    half-curled hand is genuinely neither, and reporting nothing is better than
    alternating between two mapped keys while the user's hand rests midway.
    """

    activities = tuple(d for d in (get(i) for i in _IDS) if d is not None)

    def __init__(self):
        self._tuning = Tuning()
        self._closed: dict[str, SchmittGate] = {}
        self._open: dict[str, SchmittGate] = {}
        self._pinch: dict[str, SchmittGate] = {}

    def configure(self, tuning: Tuning) -> None:
        self._tuning = tuning
        on, off = tuning.level_min_on, tuning.level_min_off
        # Falling polarity: curl below `enter` means the fingertips are inside
        # their own knuckles.
        self._closed = {
            s: SchmittGate(tuning.fist_closed_enter, tuning.fist_closed_exit, on, off) for s in _SIDES
        }
        # Rising polarity over the same pair, so open and closed can never both
        # be true and can never flip without crossing the whole band.
        self._open = {
            s: SchmittGate(tuning.fist_closed_exit, tuning.fist_closed_enter, on, off) for s in _SIDES
        }
        self._pinch = {s: SchmittGate(tuning.pinch_enter, tuning.pinch_exit, on, off) for s in _SIDES}

    def reset(self) -> None:
        for group in (self._closed, self._open, self._pinch):
            for gate in group.values():
                gate.reset()

    def update(self, f: BodyFeatures, out: Sink) -> None:
        if not self._closed:
            self.configure(self._tuning)

        for side in _SIDES:
            hand = f.hand(side)
            # No hand this frame -> NaN, which the gates read as "no evidence"
            # and release after the debounce rather than latching on.
            curl = hand.curl if hand is not None else NAN
            pinch = hand.pinch if hand is not None else NAN

            closed = self._closed[side].update(curl, f.t)
            opened = self._open[side].update(curl, f.t)
            pinched = self._pinch[side].update(pinch, f.t)

            out.level(f"{side}_fist_closed", closed, self._closed[side].confidence)
            out.level(f"{side}_fist_open", opened, self._open[side].confidence)
            out.level(f"{side}_pinch", pinched, self._pinch[side].confidence)
