"""Capture behaviour, with a fake camera so it is deterministic."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from motionsense.sources import CallableSource, IterableSource, SourceError
from motionsense.sources.camera import CameraSource

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeCapture:
    """Stands in for ``cv2.VideoCapture``, producing frames on demand."""

    def __init__(self, *, interval=0.002, fail_after=None, opens=True):
        self.interval = interval
        self.fail_after = fail_after
        self._opens = opens
        self.count = 0
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self._opens

    def read(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            return False, None
        time.sleep(self.interval)
        self.count += 1
        frame = FRAME.copy()
        frame[0, 0, 0] = self.count % 256
        return True, frame

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    """Patch cv2.VideoCapture; returns a holder for the instance created."""
    import cv2

    holder = {}

    def factory(**kwargs):
        def make(index, api=None):
            capture = FakeCapture(**kwargs)
            holder["capture"] = capture
            return capture

        monkeypatch.setattr(cv2, "VideoCapture", make)
        return holder

    return factory


def test_camera_delivers_frames(fake_cv2):
    holder = fake_cv2()
    source = CameraSource(0)
    source.open()
    try:
        frame = source.read(timeout=2.0)
        assert frame is not None
        assert frame.image.shape == FRAME.shape
    finally:
        source.close()
    assert holder["capture"].released is True


def test_camera_hands_over_the_newest_frame_not_a_backlog(fake_cv2):
    """The latency fix: ``read()`` on a real capture returns the *oldest* queued
    frame, so a consumer slower than the camera falls further behind every
    frame. Dropping the backlog keeps the delay bounded."""
    fake_cv2(interval=0.001)
    source = CameraSource(0)
    source.open()
    try:
        first = source.read(timeout=2.0)
        assert first is not None
        time.sleep(0.15)  # let the grab thread run well ahead

        second = source.read(timeout=2.0)
        assert second is not None
        # Many frames were captured during the sleep; the next read must skip
        # them rather than replay them one at a time.
        assert second.index - first.index > 5
        assert source.dropped > 0
    finally:
        source.close()


def test_camera_read_is_not_stale_after_a_slow_consumer(fake_cv2):
    fake_cv2(interval=0.001)
    source = CameraSource(0)
    source.open()
    try:
        source.read(timeout=2.0)
        time.sleep(0.1)
        frame = source.read(timeout=2.0)
        age = time.perf_counter() - frame.captured_at
        assert age < 0.05, f"frame was {age * 1000:.0f} ms old"
    finally:
        source.close()


def test_camera_reports_a_lost_feed(fake_cv2):
    fake_cv2(interval=0.001, fail_after=3)
    source = CameraSource(0)
    source.open()
    try:
        deadline = time.perf_counter() + 2.0
        with pytest.raises(SourceError):
            while time.perf_counter() < deadline:
                source.read(timeout=0.2)
    finally:
        source.close()


def test_camera_raises_when_it_cannot_open(fake_cv2):
    fake_cv2(opens=False)
    source = CameraSource(0)
    with pytest.raises(SourceError, match="could not open camera"):
        source.open()


def test_camera_close_is_idempotent(fake_cv2):
    fake_cv2()
    source = CameraSource(0)
    source.open()
    source.close()
    source.close()


def test_iterable_source_is_finite_and_ordered():
    source = IterableSource([FRAME, FRAME, FRAME], fps=10.0)
    source.open()
    stamps = []
    while True:
        frame = source.read()
        if frame is None:
            break
        stamps.append(frame.captured_at)
    source.close()
    assert stamps == [0.0, 0.1, 0.2]
    assert source.is_live is False


def test_iterable_source_accepts_explicit_timestamps():
    source = IterableSource([(FRAME, 5.0), (FRAME, 5.5)])
    source.open()
    assert source.read().captured_at == 5.0
    assert source.read().captured_at == 5.5
    assert source.read() is None
    source.close()


def test_callable_source_ends_on_none():
    remaining = [FRAME, FRAME, None]
    source = CallableSource(lambda: remaining.pop(0))
    source.open()
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None
    source.close()


def test_source_context_manager_opens_and_closes(fake_cv2):
    holder = fake_cv2()
    with CameraSource(0) as source:
        assert source.read(timeout=2.0) is not None
    assert holder["capture"].released is True


def test_capture_thread_does_not_outlive_close(fake_cv2):
    fake_cv2(interval=0.001)
    before = threading.active_count()
    source = CameraSource(0)
    source.open()
    source.read(timeout=2.0)
    source.close()
    time.sleep(0.05)
    assert threading.active_count() <= before
