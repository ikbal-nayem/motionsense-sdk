"""Landmark detection backends.

:func:`create_provider` picks between the two MediaPipe APIs based on what is
installed. Supply your own :class:`LandmarkProvider` to ``MotionEngine`` to use
a different pose model entirely -- produce the 33-point BlazePose layout (see
:mod:`motionsense.landmarks`) and every recognizer works unchanged.
"""

from .auto import available_backend, create_provider
from .base import Landmarks, LandmarkProvider

__all__ = [
    "LandmarkProvider",
    "Landmarks",
    "MediaPipeSolutionsProvider",
    "MediaPipeTasksProvider",
    "available_backend",
    "create_provider",
]


def __getattr__(name: str):
    # Imported lazily: `import motionsense` should not pull in MediaPipe, which
    # takes seconds, for callers that supply their own provider.
    if name == "MediaPipeSolutionsProvider":
        from .mediapipe_solutions import MediaPipeSolutionsProvider

        return MediaPipeSolutionsProvider
    if name == "MediaPipeTasksProvider":
        from .mediapipe_tasks import MediaPipeTasksProvider

        return MediaPipeTasksProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
