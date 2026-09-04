"""Where frames come from."""

from .base import Frame, SourceError, VideoSource
from .camera import CameraSource
from .frames import CallableSource, IterableSource
from .video import VideoFileSource

__all__ = [
    "CallableSource",
    "CameraSource",
    "Frame",
    "IterableSource",
    "SourceError",
    "VideoFileSource",
    "VideoSource",
]
