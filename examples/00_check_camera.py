"""Check that a camera works, before blaming the detector.

    python examples/00_check_camera.py          # probe indices 0-3
    python examples/00_check_camera.py 1        # just index 1

Reports, per camera index: whether it opens, whether it actually delivers
frames, at what resolution and frame rate, and how many reads it drops. Opening
a camera only reserves the handle -- a device held by another application, or a
ghost index left behind by a virtual-camera driver, opens perfectly well and
then produces nothing. Those two cases look identical until you try to read.

No detection runs here, so a failure points at the camera, its drivers, or
another application holding it.
"""

from __future__ import annotations

import sys
import time

from motionsense import CameraSource, SourceError


def probe(index: int) -> bool:
    print(f"\n--- camera {index} ---")
    source = CameraSource(index, warmup=4.0)

    started = time.perf_counter()
    try:
        source.open()
    except SourceError as exc:
        print(f"  FAILED: {exc}")
        return False
    print(f"  opened and delivered its first frame in {time.perf_counter() - started:.2f}s")

    try:
        frames, first = 0, None
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            frame = source.read(timeout=1.0)
            if frame is None:
                continue
            if first is None:
                first = frame
            frames += 1

        if first is None:
            print("  FAILED: opened but no frames arrived")
            return False

        height, width = first.image.shape[:2]
        elapsed = 3.0
        print(f"  resolution : {width} x {height}")
        print(f"  delivered  : {frames} frames in {elapsed:.0f}s  (~{frames / elapsed:.0f} fps to the consumer)")
        print(f"  dropped    : {source.dropped} (frames captured while the consumer was busy -- normal)")
        print("  OK")
        return True
    except SourceError as exc:
        print(f"  FAILED mid-stream: {exc}")
        return False
    finally:
        source.close()


def main(argv: list[str]) -> int:
    indices = [int(a) for a in argv] if argv else [0, 1, 2, 3]
    working = [i for i in indices if probe(i)]

    print()
    if working:
        print(f"Working camera indices: {working}")
        print(f"Use one with:  MotionEngine().start({working[0]})")
        return 0

    print("No working camera found. Things to check:")
    print("  - another application (Teams, Zoom, OBS, the Camera app) holding the device")
    print("  - OS camera privacy settings blocking desktop apps")
    print("  - on Windows, a virtual-camera driver occupying a low index")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
