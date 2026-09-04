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
        warmup: float = 4.0,
        read_timeout: float = 2.0,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.api_preference = api_preference
        #: How long to wait for the camera's first frame. A webcam that reports
        #: itself open is often not yet delivering: DirectShow devices in
        #: particular fail their first reads for up to a couple of seconds while
        #: exposure settles.
        self.warmup = warmup
        #: How long reads may keep failing before the feed is declared lost.
        #: Individual failures are normal and transient; a sustained run of them
        #: is not.
        self.read_timeout = read_timeout

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
        # Prove the device actually delivers before reporting success. Opening
        # only reserves the handle -- a camera held by another application, or a
        # ghost index, opens happily and then never produces a frame. Failing
        # here gives the caller an accurate message instead of a mysterious
        # "feed ended" a second later.
        primed = self._prime(cap)
        if primed is None:
            cap.release()
            self._cap = None
            raise SourceError(
                f"camera {self.index} opened but produced no frames within "
                f"{self.warmup:.0f}s. It may be in use by another application."
            )

        self._counter = 1
        self._latest = primed
        self._consumed = 0
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._grab_loop, name="motionsense-capture", daemon=True)
        self._thread.start()

    def _prime(self, cap) -> Frame | None:
        """Read until the camera yields a real frame, or the warm-up expires."""
        deadline = time.perf_counter() + self.warmup
        while True:
            ok, image = cap.read()
            if ok and image is not None and getattr(image, "size", 0):
                return Frame(image=image, captured_at=time.perf_counter(), index=1)
            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.03)

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
        last_good = time.perf_counter()
        while self._running and cap is not None:
            ok, image = cap.read()
            if not ok or image is None:
                # A dropped read is normal: USB cameras glitch, and a laptop
                # waking from sleep can miss several in a row. Only a sustained
                # outage means the feed is actually gone, so give it until
                # read_timeout to recover rather than tearing down on the first
                # failure.
                if time.perf_counter() - last_good >= self.read_timeout:
                    with self._new_frame:
                        self._error = (
                            f"lost the camera feed (no frames for "
                            f"{self.read_timeout:.0f}s). It may have been unplugged "
                            f"or taken by another application."
                        )
                        self._running = False
                        self._new_frame.notify_all()
                    return
                time.sleep(0.01)
                continue

            now = last_good = time.perf_counter()
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
