"""Teach the SDK an activity it does not ship with.

    python examples/03_custom_activity.py

Two ways to extend, in increasing order of power:

1. ``engine.define`` -- a function of the body features. Enough for most poses.
2. A ``Recognizer`` subclass -- when you need state across frames.

Both get the same hysteresis, debouncing, visibility handling and event
semantics as the built-in activities.
"""

from __future__ import annotations

import math
import time

from motionsense import MotionEngine, Pose, Trigger
from motionsense.mathx import WindowedSlope
from motionsense.recognizers import Recognizer
from motionsense.types import ActivityDef


def hand_over_heart(features) -> float:
    """How close the right hand is to the left side of the chest.

    Features arrive normalised: the origin is the hip centre, +x is the
    subject's right, +y is up, and one unit is about a torso. So "chest, left
    side" is a fixed point in these coordinates regardless of who the subject
    is, how far away they stand, or where in the frame they are.

    Returns *negative distance*, so larger is better and the gate thresholds
    read as "within this far".
    """
    if not features.visible(Pose.RIGHT_WRIST):
        return float("nan")  # cannot tell from this frame -> releases, not latches
    wrist = features.P[Pose.RIGHT_WRIST]
    target = (-0.18, 0.62)
    return -math.hypot(float(wrist[0]) - target[0], float(wrist[1]) - target[1])


class LeanOscillation(Recognizer):
    """Fires when the subject rocks side to side -- state across frames, so a
    predicate cannot express it."""

    activities = (
        ActivityDef(
            id="rocking",
            name="Rocking",
            category="Custom",
            trigger=Trigger.EDGE,
            description="Torso swaying repeatedly from side to side.",
        ),
    )

    def __init__(self):
        self._lean = WindowedSlope(window=1.4)
        self._crossings = 0
        self._sign = 0
        self._last_fire = float("-inf")

    def reset(self) -> None:
        self._lean.clear()
        self._crossings = 0
        self._sign = 0

    def update(self, features, out) -> None:
        lean = features.torso_lean
        if math.isnan(lean):
            return
        self._lean.push(features.t, lean)

        # Count sign changes outside a dead band, the same Schmitt-trigger idea
        # the wave detector uses: noise near zero must not count as a crossing.
        sign = 1 if lean > 8.0 else (-1 if lean < -8.0 else 0)
        if sign and sign != self._sign:
            if self._sign:
                self._crossings += 1
            self._sign = sign

        if self._crossings >= 3 and features.t - self._last_fire > 2.0:
            self._last_fire = features.t
            self._crossings = 0
            out.edge("rocking", 1.0, amplitude=round(self._lean.span(features.t), 1))


def main() -> None:
    engine = MotionEngine()

    engine.define(
        "hand_over_heart",
        hand_over_heart,
        trigger="level",
        name="Hand Over Heart",
        description="Right hand resting on the left side of the chest.",
        # Distances are in body units, so these mean "within 0.20 torsos" to
        # enter and "further than 0.30" to release.
        enter=-0.20,
        exit=-0.30,
    )
    engine.add_recognizer(LeanOscillation())

    engine.on("hand_over_heart", lambda e: print("hand over heart"))
    engine.on_end("hand_over_heart", lambda e: print(f"released after {e.duration:.1f}s"))
    engine.on("rocking", lambda e: print(f"rocking, {e.data['amplitude']} degrees"))

    print("Try: right hand on your chest, or rock side to side. Ctrl-C to stop.\n")
    engine.start()
    try:
        while engine.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
