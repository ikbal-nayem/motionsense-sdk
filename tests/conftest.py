from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from motionsense import EngineConfig, Event, MotionEngine  # noqa: E402
from synthetic import ScriptedProvider, frames_from  # noqa: E402

DUMMY_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


class Harness:
    """Runs a scripted pose sequence through the real engine."""

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig(preset="fast")
        self.events: list[Event] = []
        self.engine: MotionEngine | None = None

    def run(self, poses, *, fps: float = 30.0, hands=None) -> list[Event]:
        provider = ScriptedProvider(frames_from(poses, hands))
        self.engine = MotionEngine(self.config, provider=provider)
        self.engine.on_any(self.events.append)
        for i in range(len(poses)):
            self.engine.submit(DUMMY_IMAGE, timestamp=i / fps)
        return self.events

    # -- assertions helpers ----------------------------------------------------
    def triggered(self) -> list[str]:
        from motionsense import Phase

        return [e.activity for e in self.events if e.phase is Phase.TRIGGER]

    def started(self) -> list[str]:
        from motionsense import Phase

        return [e.activity for e in self.events if e.phase is Phase.START]

    def held(self) -> set[str]:
        return set(self.engine.active) if self.engine else set()


@pytest.fixture
def harness():
    return Harness()


@pytest.fixture
def make_harness():
    return Harness
