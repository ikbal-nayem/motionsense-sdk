"""Control several different things, with a gesture set per context.

    python examples/05_control_profiles.py

The pattern for applications where the mapping is *data* rather than code --
loaded from a config file, edited in a UI, switched when the user changes
context. Here the same body gestures drive a media player, then smart lights,
then a slide deck, with a T-pose cycling between them.

Nothing about the detection changes when the profile does; only the mapping
does. That separation is the point of the SDK: activities are facts about the
body, and what they *mean* belongs to the application.
"""

from __future__ import annotations

import time

from motionsense import EngineConfig, MotionEngine
from motionsense.bindings import ActionRouter


# -- the things being controlled (stubs; swap for real device calls) ---------
def say(message: str):
    return lambda: print(f"  -> {message}")


MEDIA = {
    "swipe_left": say("previous track"),
    "swipe_right": say("next track"),
    "left_pinch": say("play / pause"),
    "lean_left": {"start": say("volume down (held)"), "end": say("volume steady")},
    "lean_right": {"start": say("volume up (held)"), "end": say("volume steady")},
}

LIGHTS = {
    "both_hands_up": {"start": say("lights on"), "end": say("lights hold")},
    "hands_down": say("lights off"),
    "wave_left_hand": say("warmer"),
    "wave_right_hand": say("cooler"),
    "left_pinch": say("dim"),
}

SLIDES = {
    "swipe_left": say("previous slide"),
    "swipe_right": say("next slide"),
    "jump": say("black screen"),
    "arms_crossed": say("presenter notes"),
}

PROFILES = [("media", MEDIA), ("lights", LIGHTS), ("slides", SLIDES)]


def main() -> None:
    # enable_hands is "auto" by default and these profiles use `left_pinch`, so
    # the hand model switches itself on. Drop the pinch bindings and it stays
    # off, and the pipeline runs roughly twice as fast.
    engine = MotionEngine(EngineConfig(preset="fast"))
    router = ActionRouter(engine)

    state = {"index": 0}

    def activate(index: int) -> None:
        state["index"] = index % len(PROFILES)
        name, mapping = PROFILES[state["index"]]
        # `load` swaps the whole mapping atomically -- no window where half the
        # old profile and half the new one are both live.
        router.load(mapping)
        print(f"\n=== profile: {name} ===")
        for activity in mapping:
            print(f"    {activity}")

    # T-pose cycles profiles. Subscribed directly rather than through the
    # router, so switching profiles is never itself remapped away.
    engine.on("t_pose", lambda e: activate(state["index"] + 1))
    engine.on_error(lambda exc: print(f"! {exc}"))

    activate(0)
    print("\nHold a T-pose to switch profiles. Ctrl-C to stop.")

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
