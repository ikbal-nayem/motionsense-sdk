"""Finger states from the hand model: fist open/closed, pinch."""

from __future__ import annotations

from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures, HandFeatures
from ..mathx import SchmittGate
from .base import Recognizer, Sink, margin, margin_at_least, min_margin

__all__ = ["HandsRecognizer"]

_IDS = (
    "left_fist_closed",
    "left_fist_open",
    "right_fist_closed",
    "right_fist_open",
    "left_pinch",
    "right_pinch",
    "left_thumbs_up",
    "right_thumbs_up",
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
        self._thumbs_up: dict[str, SchmittGate] = {}

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
        # Thumbs-up needs three conditions at once, so it goes through the
        # margin scheme the multi-condition poses use: one number, 0 at the
        # limit, driving one gate.
        self._thumbs_up = {
            s: SchmittGate(0.0, tuning.pose_margin_exit, on, off) for s in _SIDES
        }

    def reset(self) -> None:
        for group in (self._closed, self._open, self._pinch, self._thumbs_up):
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

            closed = self._closed[side].update(curl, f.t)
            opened = self._open[side].update(curl, f.t)
            pinched = self._pinch[side].update(self._pinch_value(hand), f.t)
            thumbs_up = self._thumbs_up[side].update(self._thumbs_up_score(hand), f.t)

            # A thumbs-up curls the same four fingers a fist does, so the fist
            # gate is right about the fingers and wrong about the gesture. The
            # more specific reading wins: without this, one gesture presses two
            # mapped keys.
            out.level(f"{side}_fist_closed", closed and not thumbs_up, self._closed[side].confidence)
            out.level(f"{side}_fist_open", opened, self._open[side].confidence)
            out.level(f"{side}_pinch", pinched, self._pinch[side].confidence)
            out.level(f"{side}_thumbs_up", thumbs_up, self._thumbs_up[side].confidence)

    # -- scores --------------------------------------------------------------
    def _pinch_value(self, hand: HandFeatures | None) -> float:
        """Thumb-index gap, or NaN once the hand is too closed to mean it.

        The gap alone cannot separate a pinch from a fist: a fist folds the
        thumb across the curled index, which puts the tips as close together as
        a deliberate pinch does. The fingers a pinch does not use are what tell
        them apart, and while they are curled into a fist there is no pinch to
        measure -- which is what NaN says, and the gate releases on it.
        """
        if hand is None:
            return NAN
        if hand.outer_curl < self._tuning.pinch_outer_curl_min:
            return NAN
        return hand.pinch

    def _thumbs_up_score(self, hand: HandFeatures | None) -> float:
        """Fingers closed, thumb extended, thumb pointing up -- as one margin."""
        if hand is None:
            return NAN
        tn = self._tuning
        return min_margin(
            margin(hand.curl, tn.fist_closed_enter, 0.20),
            margin_at_least(hand.thumb_extension, tn.thumbs_up_extension_min, 0.15),
            margin_at_least(hand.thumb_up, tn.thumbs_up_direction_min, 0.20),
        )
