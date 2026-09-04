"""The smallest useful program: print activities as they happen.

    python examples/01_listen.py

Stand back far enough that your hips and knees are in frame, then wave, raise a
hand, lean, or crouch.
"""

from __future__ import annotations

import time

from motionsense import Event, MotionEngine


def main() -> None:
    engine = MotionEngine()

    # `on_any` receives every phase of every activity. For real applications
    # subscribe to what you need instead -- it lets the engine skip models.
    @engine.on_any
    def show(event: Event) -> None:
        definition = event.definition
        name = definition.name if definition else event.activity
        print(f"{event.timestamp:7.2f}s  {name:<18} {event.phase.value:<8} {event.confidence:.2f}")

    engine.on_error(lambda exc: print(f"! {exc}"))

    print("Starting camera. Ctrl-C to stop.\n")
    engine.start()
    try:
        while engine.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
