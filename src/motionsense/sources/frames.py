"""Sources backed by your own frames."""

from __future__ import annotations

import time
from typing import Callable, Iterable, Iterator

import numpy as np

from .base import Frame, VideoSource

__all__ = ["CallableSource", "IterableSource"]


class IterableSource(VideoSource):
    """Wraps any iterable of BGR arrays (or ``(image, timestamp)`` pairs).

    The adapter for a pipeline that already has frames -- a GStreamer bridge, a
    ROS topic, a Qt capture widget, a list of images in a test. Nothing else in
    the SDK needs to know where the pixels came from.
    """

    is_live = False

    def __init__(self, frames: Iterable, *, drop_stale: bool = False, fps: float = 30.0):
        self._frames = frames
        self._iter: Iterator | None = None
        self.drop_stale = drop_stale
        self._fps = fps
        self._counter = 0

    def open(self) -> None:
        self._iter = iter(self._frames)
        self._counter = 0

    def close(self) -> None:
        self._iter = None

    def read(self, timeout: float = 1.0) -> Frame | None:
        if self._iter is None:
            return None
        try:
            item = next(self._iter)
        except StopIteration:
            return None

        if isinstance(item, tuple):
            image, stamp = item
        else:
            image, stamp = item, self._counter / self._fps
        self._counter += 1
        return Frame(image=np.asarray(image), captured_at=float(stamp), index=self._counter)


class CallableSource(VideoSource):
    """Pulls frames from a function.

    Use when frames arrive from a system that hands them out on demand. Return
    ``None`` from the callable to end the stream.
    """

    def __init__(self, fn: Callable[[], np.ndarray | None], *, drop_stale: bool = True):
        self._fn = fn
        self.drop_stale = drop_stale
        self._counter = 0

    def open(self) -> None:
        self._counter = 0

    def close(self) -> None:
        pass

    def read(self, timeout: float = 1.0) -> Frame | None:
        image = self._fn()
        if image is None:
            return None
        self._counter += 1
        return Frame(image=np.asarray(image), captured_at=time.perf_counter(), index=self._counter)
