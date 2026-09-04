"""Why isn't my pose detecting?

    python examples/07_diagnose.py                 # watch every pose
    python examples/07_diagnose.py arms_crossed    # focus on one

Prints, live, what the recognizers actually measured. A pose that is not firing
fails in one of two ways, and they need opposite fixes:

  score is NEGATIVE   the geometry was measured and rejected. You are not far
                      enough into the pose, or the threshold wants loosening.
                      The number is in units of the hysteresis band, so -0.4
                      means you are 40% of a band short.

  score is NaN        it could not be measured at all -- a landmark the pose
                      needs is not visible. Fix the framing or the lighting;
                      loosening a threshold will not help, because nothing was
                      compared to it.

Without this the two look identical from outside: the pose just never fires.
"""

from __future__ import annotations

import math
import sys
import threading

from motionsense import EngineConfig, MotionEngine, Trigger, catalog


def main(focus: str | None) -> int:
    if focus:
        catalog.require(focus)  # fail now on a typo, with a suggestion

    engine = MotionEngine(EngineConfig(preset="fast", enable_hands=True))
    latest: dict = {}
    arrived = threading.Event()

    @engine.on_frame
    def capture(result) -> None:
        latest["result"] = result
        arrived.set()

    engine.on_error(lambda exc: print(f"! {exc}"))

    level_ids = [a.id for a in catalog.all_activities() if a.trigger is Trigger.LEVEL]
    watched = [focus] if focus else level_ids

    print(f"Watching: {', '.join(watched)}")
    print("Ctrl-C to stop.\n")
    engine.start()

    try:
        while engine.running:
            if not arrived.wait(0.5):
                continue
            arrived.clear()
            result = latest.get("result")
            if result is None:
                continue

            if not result.levels:
                print("\rno body detected" + " " * 60, end="")
                continue

            parts = []
            for activity in watched:
                entry = result.levels.get(activity)
                if entry is None:
                    continue
                active, confidence, data = entry
                score = data.get("score")
                if score is None:
                    detail = f"{confidence:.2f}"
                elif isinstance(score, float) and math.isnan(score):
                    detail = "NOT VISIBLE"
                else:
                    detail = f"{score:+.2f}"
                mark = "ON " if active else "   "
                parts.append(f"{mark}{activity}={detail}")

            line = "  ".join(parts) if focus else "  ".join(p for p in parts if "ON " in p or focus)
            print(f"\r{line[:200]:<200}", end="")
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
