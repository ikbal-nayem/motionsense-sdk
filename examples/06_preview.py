"""Live preview window with the skeleton and current activities drawn on it.

    python examples/06_preview.py

Shows the two things worth knowing when integrating into a GUI:

* Rendering happens on *your* thread, not the engine's. ``on_frame`` hands you
  the result; here it is put in a one-slot mailbox and the main thread draws it.
  Doing OpenCV window work inside the callback would stall detection, and on
  most platforms GUI calls off the main thread are undefined behaviour anyway.
* ``deliver_frames=True`` is what attaches the image. It is off by default
  because most applications never need the pixels.

Press `c` to calibrate (stand neutral first), `q` to quit.
"""

from __future__ import annotations

import threading

from motionsense import EngineConfig, MotionEngine, catalog
from motionsense.draw import render


def main() -> None:
    import cv2

    engine = MotionEngine(EngineConfig(preset="balanced", deliver_frames=True, enable_hands=True))

    latest: dict = {}
    arrived = threading.Event()

    @engine.on_frame
    def capture(result) -> None:
        latest["result"] = result
        arrived.set()

    engine.on_error(lambda exc: print(f"! {exc}"))
    labels = {a.id: a.name for a in catalog.all_activities()}

    engine.start()
    print("q quit, c calibrate")
    try:
        while engine.running:
            if not arrived.wait(0.5):
                continue
            arrived.clear()
            result = latest.get("result")
            if result is None:
                continue

            canvas = render(result, mirror=True, snapshot=engine.snapshot(), labels=labels)
            if canvas is None:
                continue
            cv2.imshow("motionsense", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                print("hold still...")
                engine.calibrate(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cv2.destroyAllWindows()
        if engine.calibration.is_set:
            print("calibration:", engine.calibration.neutral)


if __name__ == "__main__":
    main()
