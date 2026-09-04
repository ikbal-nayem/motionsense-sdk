"""Video file playback."""

from __future__ import annotations

import time

from .base import Frame, SourceError, VideoSource

__all__ = ["VideoFileSource"]


class VideoFileSource(VideoSource):
    """Reads a video file frame by frame.

    ``drop_stale`` is ``False``: for a recording every frame must be processed,
    because timestamps come from the file's own timeline and skipping frames
    would silently distort every velocity the recognizers compute.

    By default it runs as fast as the machine allows, which makes it useful for
    offline analysis and for reproducible tests -- the same file always produces
    the same events. Set ``realtime=True`` to pace playback at the file's frame
    rate when you want to watch it happen.
    """

    drop_stale = False
    is_live = False

    def __init__(self, path: str, *, realtime: bool = False, loop: bool = False):
        self.path = str(path)
        self.realtime = realtime
        self.loop = loop
        self._cap = None
        self._counter = 0
        self._fps = 30.0
        self._started: float | None = None

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"could not open video file: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 1e-3 else 30.0
        self._cap = cap
        self._counter = 0
        self._started = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, timeout: float = 1.0) -> Frame | None:
        if self._cap is None:
            return None
        ok, image = self._cap.read()
        if not ok:
            if self.loop:
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._counter = 0
                self._started = None
                ok, image = self._cap.read()
            if not ok:
                return None

        # Timestamps follow the file, not the wall clock, so velocities are
        # correct whether playback is faster or slower than real time.
        stamp = self._counter / self._fps
        self._counter += 1

        if self.realtime:
            if self._started is None:
                self._started = time.perf_counter()
            wait = self._started + stamp - time.perf_counter()
            if wait > 0:
                time.sleep(min(wait, timeout))

        return Frame(image=image, captured_at=stamp, index=self._counter)
