"""Drive a game (or any application) with body movement.

    pip install motionsense[keys]
    python examples/02_keyboard.py

Poses are held and gestures are tapped, so the mapping is just
``{activity: key}``. On Windows the `keyboard` package needs administrator
privileges to inject keystrokes into another application; on Linux it needs
root or membership of the `input` group.
"""

from __future__ import annotations

import time

from motionsense import EngineConfig, MotionEngine
from motionsense.bindings import KeyBindings

# Poses become held keys, gestures become taps. Nothing here says which is
# which -- the catalog already knows, and KeyBindings reads it.
MAPPING = {
    "left_hand_up": "a",
    "right_hand_up": "d",
    "both_hands_up": "w",
    "squat": "ctrl",
    "lean_left": "q",
    "lean_right": "e",
    "jump": "space",
    "wave_right_hand": "r",
    "swipe_left": "left",
    "swipe_right": "right",
}


def main() -> None:
    # "fast" trades a little landmark precision for roughly double the frame
    # rate, which is the right trade for control: latency is what you feel.
    engine = MotionEngine(EngineConfig(preset="fast"))
    keys = KeyBindings(engine, MAPPING)

    engine.on_any(lambda e: print(f"{e.activity:<18} {e.phase.value:<8} -> {MAPPING.get(e.activity, '')}"))
    engine.on_error(lambda exc: print(f"! {exc}"))

    print("Mapped:")
    for activity, key in MAPPING.items():
        print(f"  {activity:<18} {key}")
    print("\nCtrl-C to stop.\n")

    engine.start()
    try:
        while engine.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters: stopping the engine emits END for everything still
        # held, which is what lifts the keys. Clearing the bindings first would
        # leave one pressed.
        engine.stop()
        keys.clear()
        print("\nStopped, all keys released.")


if __name__ == "__main__":
    main()
