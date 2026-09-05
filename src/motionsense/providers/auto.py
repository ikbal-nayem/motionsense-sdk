"""Picks a MediaPipe backend based on what is installed."""

from __future__ import annotations

import logging

from ..config import EngineConfig
from .base import LandmarkProvider

__all__ = ["create_provider", "available_backend"]

log = logging.getLogger("motionsense")


def available_backend() -> str:
    """``"solutions"``, ``"tasks"``, or ``"none"``.

    MediaPipe reorganised between 0.10 and 1.0: ``mediapipe.solutions`` was
    removed and the Tasks API became the only entry point. Both are supported so
    that installing this package does not force a MediaPipe version on the host
    application -- which matters when it is being added to a project that
    already depends on one.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return "none"
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return "solutions"
    try:
        import importlib

        importlib.import_module("mediapipe.tasks.python.vision")
    except ImportError:
        return "none"
    return "tasks"


def create_provider(config: EngineConfig) -> LandmarkProvider:
    """Build the default landmark provider for this environment."""
    requested = config.backend
    backend = available_backend() if requested == "auto" else requested

    if backend == "solutions":
        from .mediapipe_solutions import MediaPipeSolutionsProvider

        return MediaPipeSolutionsProvider(config)
    if backend == "tasks":
        from .mediapipe_tasks import MediaPipeTasksProvider

        return MediaPipeTasksProvider(config)

    raise ImportError(
        "no MediaPipe backend is available. Install it with `pip install mediapipe`, "
        "or pass your own LandmarkProvider to MotionEngine(provider=...)."
    )
