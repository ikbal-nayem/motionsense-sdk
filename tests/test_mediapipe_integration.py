"""Exercises the real MediaPipe providers.

The recognizer tests use a scripted provider so they stay deterministic and
offline. These check what a scripted provider cannot: that a model actually
loads, that detector output converts to the array layout everything downstream
assumes, and that backend selection works on whichever MediaPipe is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from motionsense import EngineConfig, IterableSource, MotionEngine
from motionsense.landmarks import Pose
from motionsense.providers import available_backend, create_provider

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

BACKEND = available_backend()
needs_backend = pytest.mark.skipif(BACKEND == "none", reason="no MediaPipe backend installed")


@pytest.fixture(scope="module")
def blank():
    return np.full((480, 640, 3), 32, dtype=np.uint8)


def test_a_backend_is_detected():
    assert BACKEND in ("solutions", "tasks")


@needs_backend
def test_provider_loads_and_returns_nothing_for_an_empty_scene(blank):
    provider = create_provider(EngineConfig(preset="fast"))
    try:
        result = provider.process(blank, 0.0, 0)
        assert result.pose is None
        assert result.hands == ()
    finally:
        provider.close()


@needs_backend
def test_provider_downscales_large_frames():
    """The model resizes internally anyway, so feeding it full resolution only
    pays for a bigger resize -- twice, when the hand model is on."""
    provider = create_provider(EngineConfig(preset="fast", inference_width=320))
    try:
        provider.process(np.full((1080, 1920, 3), 32, dtype=np.uint8), 0.0, 0)
        assert provider._rgb.shape[1] == 320
        assert provider._rgb.shape[0] == 180  # aspect preserved
    finally:
        provider.close()


@needs_backend
def test_hand_model_is_only_created_when_enabled():
    provider = create_provider(EngineConfig(preset="fast"))
    try:
        assert provider._hands is None
        provider.set_hands_enabled(True)
        assert provider._hands is not None
        provider.set_hands_enabled(False)
        assert provider._hands is None
    finally:
        provider.close()


@needs_backend
def test_engine_runs_the_real_pipeline_over_frames(blank):
    """Full stack with a real model: no exceptions, no events from an empty
    scene, and a clean shutdown."""
    engine = MotionEngine(EngineConfig(preset="fast"))
    events = []
    engine.on_any(events.append)

    engine.run(IterableSource([blank] * 5, fps=30.0))

    assert events == []
    assert engine.snapshot().frames == 5
    assert engine.running is False


@needs_backend
def test_repeated_timestamps_do_not_break_video_mode(blank):
    """Tasks' VIDEO mode rejects a non-increasing timestamp. Frames can share a
    millisecond when a recording is replayed faster than real time."""
    provider = create_provider(EngineConfig(preset="fast"))
    try:
        for _ in range(5):
            provider.process(blank, 0.0, 0)
    finally:
        provider.close()


@pytest.mark.skipif(BACKEND != "solutions", reason="legacy API not installed")
def test_solutions_landmark_conversion_layout():
    """Guards the contract every recognizer relies on: 33 rows of
    (x, y, z, visibility)."""
    from motionsense.providers.mediapipe_solutions import MediaPipeSolutionsProvider

    class FakeLandmark:
        def __init__(self, i):
            self.x, self.y, self.z, self.visibility = i * 0.01, i * 0.02, i * 0.03, 0.9

    class FakeList:
        landmark = [FakeLandmark(i) for i in range(Pose.COUNT)]

    array = MediaPipeSolutionsProvider._convert_pose(FakeList(), Pose.COUNT, 4)
    assert array.shape == (Pose.COUNT, 4)
    assert array.dtype == np.float32
    assert array[5, 0] == pytest.approx(0.05)
    assert array[5, 3] == pytest.approx(0.9)


@pytest.mark.skipif(BACKEND != "tasks", reason="Tasks API not installed")
def test_tasks_landmark_conversion_layout():
    from motionsense.providers.mediapipe_tasks import MediaPipeTasksProvider

    class FakeLandmark:
        def __init__(self, i):
            self.x, self.y, self.z, self.visibility = i * 0.01, i * 0.02, i * 0.03, 0.9

    landmarks = [[FakeLandmark(i) for i in range(Pose.COUNT)]]

    image = MediaPipeTasksProvider._first(landmarks, Pose.COUNT, world=False)
    assert image.shape == (Pose.COUNT, 4)
    assert image[5, 0] == pytest.approx(0.05)
    assert image[5, 3] == pytest.approx(0.9)

    world = MediaPipeTasksProvider._first(landmarks, Pose.COUNT, world=True)
    assert world.shape == (Pose.COUNT, 3)


@pytest.mark.skipif(BACKEND != "tasks", reason="Tasks API not installed")
def test_missing_visibility_defaults_to_present():
    """Some Tasks builds leave the optional visibility field unset. Reading that
    as zero would make every landmark invisible and nothing would ever fire."""
    from motionsense.providers.mediapipe_tasks import _visibility

    class Bare:
        x = y = z = 0.0

    assert _visibility(Bare()) == 1.0
