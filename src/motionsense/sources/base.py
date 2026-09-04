"""Video source protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["Frame", "SourceError", "VideoSource"]


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or loses its feed."""


@dataclass(slots=True)
class Frame:
    """One captured image."""

    #: BGR uint8, ``(h, w, 3)``.
    image: np.ndarray
    #: ``time.perf_counter()`` at capture.
    captured_at: float
    index: int


class VideoSource(ABC):
    """Anything that can produce frames.

    Implement this to feed the engine from a source it does not ship with -- an
    RTSP stream, a depth camera's RGB channel, a frame grabber, a test harness
    replaying a recorded session.
    """

    #: ``True`` for live sources, where skipping a backlog is correct because
    #: only the newest frame reflects reality. ``False`` for recordings, where
    #: every frame must be processed or the timeline breaks.
    drop_stale: bool = True

    #: ``True`` when the stream is open-ended, so a read that returns nothing is
    #: a lull rather than the end. ``False`` for finite sources, where the engine
    #: should shut down once frames run out.
    is_live: bool = True

    @abstractmethod
    def open(self) -> None:
        """Acquire the device or file. Raise :class:`SourceError` on failure."""

    @abstractmethod
    def read(self, timeout: float = 1.0) -> Frame | None:
        """Next frame, or ``None`` at end of stream. May raise :class:`SourceError`."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Must be safe to call more than once."""

    @property
    def dropped(self) -> int:
        """Frames captured but never delivered, because a newer one arrived first."""
        return 0

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
