"""Live camera capture with a latest-frame policy."""

from __future__ import annotations

import sys
import threading
import time

from .base import Frame, SourceError, VideoSource

__all__ = ["CameraSource"]


class CameraSource(VideoSource):
    """Webcam capture that always hands over the *newest* frame.

    ``VideoCapture.read()`` returns the oldest frame in the driver's queue. If
    inference takes longer than the frame interval -- which it usually does --
    that queue fills, and every frame processed is progressively more out of
    date. The lag compounds: within a few seconds a gesture registers a
    noticeable moment after it happened, and no amount of detector tuning fixes
    it, because the detector is looking at the past.

    A dedicated grab thread drains the queue continuously into a single-slot
    mailbox. The engine always reads a frame that is at most one capture
    interval old, and the frames in between are dropped rather than queued.
    Dropping is the right call for interactive control: a stale frame has no
    value once a newer one exists.
    """

    drop_stale = True

    def __init__(
        self,
        index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        api_preference: int | None = None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.api_preference = api_preference

        self._cap = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: Frame | None = None
        self._running = False
        self._error: str | None = None
        self._counter = 0
        self._dropped = 0
        self._consumed = -1

    @property
    def dropped(self) -> int:
        return self._dropped

    # -- lifecycle -------------------------------------------------------------
    def open(self) -> None:
        import cv2

        candidates = []
        if self.api_preference is not None:
            candidates.append(self.api_preference)
        elif sys.platform == "win32":
            # DirectShow opens in a fraction of the time MSMF takes on Windows
            # and is less prone to stalling on cheap webcams.
            candidates.append(cv2.CAP_DSHOW)
        candidates.append(cv2.CAP_ANY)

        cap = None
        for api in candidates:
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                break
            cap.release()
            cap = None

        if cap is None:
            raise SourceError(f"could not open camera {self.index}")

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Ask the driver for a shallow queue too, where the backend honours it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cap = cap
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._grab_loop, name="motionsense-capture", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        with self._new_frame:
            self._new_frame.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._latest = None

    # -- capture ---------------------------------------------------------------
    def _grab_loop(self) -> None:
        cap = self._cap
        while self._running and cap is not None:
            ok, image = cap.read()
            if not ok:
                with self._new_frame:
                    self._error = "camera feed ended"
                    self._running = False
                    self._new_frame.notify_all()
                return
            now = time.perf_counter()
            with self._new_frame:
                if self._latest is not None and self._latest.index > self._consumed:
                    self._dropped += 1
                self._counter += 1
                self._latest = Frame(image=image, captured_at=now, index=self._counter)
                self._new_frame.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        deadline = time.perf_counter() + timeout
        with self._new_frame:
            while True:
                if self._error:
                    raise SourceError(self._error)
                frame = self._latest
                if frame is not None and frame.index > self._consumed:
                    self._consumed = frame.index
                    return frame
                if not self._running:
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)
