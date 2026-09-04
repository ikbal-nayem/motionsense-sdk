"""motionsense -- real-time human activity events from any video source.

Feed it video, subscribe to activities, drive anything::

    from motionsense import MotionEngine

    engine = MotionEngine()

    @engine.on("swipe_right")
    def next_slide(event):
        deck.advance()

    engine.start()

The SDK detects *what the body is doing* and reports it. What that means -- a
keystroke, a function call, an MQTT publish, a servo angle -- is the
application's decision, and nothing in the detection path knows or cares.

Two kinds of activity come out:

* **Poses** (``Trigger.LEVEL``) are true for as long as they hold, and produce
  ``START`` / ``END`` events. Natural fit for a held control.
* **Gestures** (``Trigger.EDGE``) fire once when recognised, producing
  ``TRIGGER``. Natural fit for a command.

See ``catalog.all_activities()`` for what ships built in, and
``MotionEngine.define`` to add your own.
"""

from .calibration import Calibration
from .config import PRESETS, EngineConfig, Tuning
from .engine import MotionEngine
from .features import BodyFeatures, HandFeatures
from .landmarks import Hand, Pose
from .sources import (
    CallableSource,
    CameraSource,
    Frame,
    IterableSource,
    SourceError,
    VideoFileSource,
    VideoSource,
)
from .types import (
    ActivityDef,
    Event,
    FrameResult,
    HandSample,
    Phase,
    Signal,
    Snapshot,
    Trigger,
)

__version__ = "0.1.0"

__all__ = [
    "ActivityDef",
    "BodyFeatures",
    "Calibration",
    "CallableSource",
    "CameraSource",
    "EngineConfig",
    "Event",
    "Frame",
    "FrameResult",
    "Hand",
    "HandFeatures",
    "HandSample",
    "IterableSource",
    "MotionEngine",
    "PRESETS",
    "Phase",
    "Pose",
    "Signal",
    "Snapshot",
    "SourceError",
    "Trigger",
    "Tuning",
    "VideoFileSource",
    "VideoSource",
    "__version__",
    "catalog",
]


def __getattr__(name: str):
    # `catalog`, `recognizers`, `bindings` and `draw` are reachable as
    # attributes without being imported eagerly.
    if name in ("catalog", "recognizers", "bindings", "draw", "mathx", "providers"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
