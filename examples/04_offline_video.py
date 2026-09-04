"""Analyse a recorded video, with no camera and no threads.

    python examples/04_offline_video.py path/to/clip.mp4

Two things make this reproducible: every frame is processed (a recording never
drops frames the way a live camera does), and timestamps come from the file's
own timeline, so velocities are correct whether the machine runs faster or
slower than real time. The same file always produces the same events -- which is
what makes this useful for regression testing a gesture set.
"""

from __future__ import annotations

import sys
from collections import Counter

from motionsense import EngineConfig, MotionEngine, Phase, VideoFileSource


def main(path: str) -> int:
    # "accurate" uses the heavy model. Offline there is no latency budget to
    # respect, so there is no reason not to.
    engine = MotionEngine(EngineConfig(preset="accurate", enable_hands=True))

    timeline = []
    engine.on_any(timeline.append)
    failure = []
    engine.on_error(failure.append)

    engine.run(VideoFileSource(path))

    if failure:
        print(f"error: {failure[0]}", file=sys.stderr)
        return 1

    counts = Counter(e.activity for e in timeline if e.phase in (Phase.START, Phase.TRIGGER))
    snapshot = engine.snapshot()

    print(f"{snapshot.frames} frames analysed\n")
    if not counts:
        print("no activities detected")
        return 0

    print(f"{'activity':<20} {'count':>6}")
    for activity, count in counts.most_common():
        print(f"{activity:<20} {count:>6}")

    print("\ntimeline:")
    for event in timeline:
        if event.phase is Phase.TRIGGER:
            print(f"  {event.timestamp:7.2f}s  {event.activity}")
        elif event.phase is Phase.END and event.duration > 0.5:
            start = event.timestamp - event.duration
            print(f"  {start:7.2f}s  {event.activity} (held {event.duration:.1f}s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
