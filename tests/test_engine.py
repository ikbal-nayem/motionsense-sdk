"""The public API: subscriptions, lifecycle, extension, bindings."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import synthetic as S
from motionsense import EngineConfig, Event, IterableSource, MotionEngine, Phase, Signal, Trigger
from motionsense.bindings import ActionRouter
from motionsense.landmarks import Pose
from motionsense.providers.base import Landmarks, LandmarkProvider
from motionsense.recognizers import Recognizer

STANDING = dict(left_arm=(-80.0, 0.0), right_arm=(-80.0, 0.0))
IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def standing_frames(n: int):
    image, world = S.project(S.body(**STANDING))
    return [Landmarks(pose=image, pose_world=world, hands=()) for _ in range(n)]


def raised_frames(n: int):
    image, world = S.project(S.body(left_arm=(85, 0), right_arm=(-80, 0)))
    return [Landmarks(pose=image, pose_world=world, hands=()) for _ in range(n)]


class Replay(LandmarkProvider):
    def __init__(self, frames, *, loop: bool = False):
        self.frames = list(frames)
        self.loop = loop
        self.index = 0
        self.hands_enabled: bool | None = None

    def process(self, image, t, frame_index):
        if self.index >= len(self.frames):
            if not self.loop or not self.frames:
                return Landmarks(None, None, ())
            self.index = 0
        result = self.frames[self.index]
        self.index += 1
        return result

    def set_hands_enabled(self, enabled):
        self.hands_enabled = enabled

    def close(self):
        pass


def drive(engine: MotionEngine, count: int, fps: float = 30.0):
    for i in range(count):
        engine.submit(IMAGE, timestamp=i / fps)


# =====================================================================
# Subscription API
# =====================================================================
def test_decorator_and_direct_registration_both_work():
    engine = MotionEngine(provider=Replay(standing_frames(10) + raised_frames(10)))
    seen = []

    @engine.on("left_hand_up")
    def _raised(event: Event):
        seen.append(("decorator", event.phase))

    engine.on("left_hand_up", lambda e: seen.append(("direct", e.phase)))
    drive(engine, 20)

    assert ("decorator", Phase.START) in seen
    assert ("direct", Phase.START) in seen


def test_unknown_activity_is_rejected_at_subscription_time():
    """A typo in a mapping would otherwise be silent: the listener just never
    fires, and there is nothing to notice."""
    engine = MotionEngine(provider=Replay([]))
    with pytest.raises(KeyError, match="left_hand_upp"):
        engine.on("left_hand_upp", lambda e: None)


def test_unknown_activity_error_suggests_a_correction():
    engine = MotionEngine(provider=Replay([]))
    with pytest.raises(KeyError, match="Did you mean"):
        engine.on("jmup", lambda e: None)


def test_unsubscribe_stops_delivery():
    engine = MotionEngine(provider=Replay(standing_frames(6) + raised_frames(20)))
    seen = []
    sub = engine.on("left_hand_up", seen.append)
    sub.cancel()
    drive(engine, 26)
    assert seen == []


def test_start_and_end_phases_with_duration():
    frames = standing_frames(6) + raised_frames(15) + standing_frames(15)
    engine = MotionEngine(provider=Replay(frames))
    events = []
    engine.on_start("left_hand_up", events.append)
    engine.on_end("left_hand_up", events.append)
    drive(engine, len(frames))

    assert [e.phase for e in events] == [Phase.START, Phase.END]
    assert events[1].duration == pytest.approx(15 / 30.0, abs=0.1)


def test_on_any_sees_every_phase():
    frames = standing_frames(6) + raised_frames(15) + standing_frames(15)
    engine = MotionEngine(provider=Replay(frames))
    events = []
    engine.on_any(events.append)
    drive(engine, len(frames))

    phases = {(e.activity, e.phase) for e in events}
    assert ("left_hand_up", Phase.START) in phases
    assert ("left_hand_up", Phase.END) in phases
    assert ("hands_down", Phase.END) in phases


def test_ends_are_delivered_before_starts_in_the_same_frame():
    """So a binding releases the key it holds before pressing the next one."""
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    order = []
    engine.on_any(lambda e: order.append((e.activity, e.phase)))
    drive(engine, len(frames))

    end_index = order.index(("hands_down", Phase.END))
    start_index = order.index(("left_hand_up", Phase.START))
    assert end_index < start_index


def test_a_raising_listener_does_not_break_the_engine():
    frames = standing_frames(6) + raised_frames(15)
    engine = MotionEngine(provider=Replay(frames))
    survivors = []
    engine.on("left_hand_up", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    engine.on("left_hand_up", survivors.append)
    drive(engine, len(frames))
    assert len(survivors) == 1


def test_updates_are_off_by_default_and_can_be_enabled():
    frames = standing_frames(6) + raised_frames(20)

    quiet = MotionEngine(provider=Replay(list(frames)))
    seen = []
    quiet.on("left_hand_up", seen.append, phase=Phase.UPDATE)
    drive(quiet, len(frames))
    assert seen == []

    loud = MotionEngine(EngineConfig(emit_updates=True), provider=Replay(list(frames)))
    seen = []
    loud.on("left_hand_up", seen.append, phase=Phase.UPDATE)
    drive(loud, len(frames))
    assert len(seen) > 5


# =====================================================================
# State and lifecycle
# =====================================================================
def test_submit_returns_the_frame_result():
    engine = MotionEngine(provider=Replay(standing_frames(5)))
    result = engine.submit(IMAGE, timestamp=0.0)
    assert result.pose is not None
    assert result.index == 1
    assert result.image is None  # deliver_frames is off


def test_frame_result_exposes_poses_that_did_not_fire():
    """The diagnostic surface. A pose that fails because its geometry was
    rejected and one that fails because a landmark is invisible look identical
    from outside otherwise, and they need opposite fixes."""
    engine = MotionEngine(provider=Replay(standing_frames(5)))
    result = engine.submit(IMAGE, timestamp=0.0)

    assert "arms_crossed" in result.levels
    active, _confidence, data = result.levels["arms_crossed"]
    assert active is False
    # Measured and rejected -> a real negative number, not NaN.
    assert data["score"] < 0
    assert not np.isnan(data["score"])


def test_frame_result_marks_unmeasurable_poses_as_nan():
    image, world = S.project(S.body(**STANDING))
    image[[Pose.LEFT_WRIST, Pose.RIGHT_WRIST], 3] = 0.0  # wrists gone entirely
    engine = MotionEngine(provider=Replay([Landmarks(pose=image, pose_world=world, hands=())]))

    result = engine.submit(IMAGE, timestamp=0.0)
    assert np.isnan(result.levels["arms_crossed"][2]["score"])


def test_deliver_frames_attaches_the_image():
    engine = MotionEngine(EngineConfig(deliver_frames=True), provider=Replay(standing_frames(3)))
    result = engine.submit(IMAGE, timestamp=0.0)
    assert result.image is IMAGE


def test_preview_interval_attaches_the_image_every_nth_frame():
    """Drawing the preview costs real time on the detection thread, so it can be
    run at a fraction of the detection rate. The phase starts at the first frame
    so a preview is not blank until the interval comes round."""
    engine = MotionEngine(
        EngineConfig(deliver_frames=True, preview_interval=3),
        provider=Replay(standing_frames(9)),
    )
    delivered = [
        engine.submit(IMAGE, timestamp=i / 30.0).image is not None for i in range(9)
    ]
    assert delivered == [True, False, False, True, False, False, True, False, False]


def test_preview_interval_does_not_gate_anything_but_the_image():
    """Only the preview is thinned. Activity state has to stay per-frame, or a
    mapped key would wait on a frame that happens to be drawing."""
    engine = MotionEngine(
        EngineConfig(deliver_frames=True, preview_interval=4),
        provider=Replay(raised_frames(12)),
    )
    results = [engine.submit(IMAGE, timestamp=i / 30.0) for i in range(12)]

    assert sum(r.image is not None for r in results) == 3
    # The pose is held throughout, on every frame, image or no image.
    assert all("left_hand_up" in r.active for r in results[3:])


def test_preview_interval_must_be_positive():
    with pytest.raises(ValueError, match="preview_interval"):
        EngineConfig(preview_interval=0)


def test_losing_the_body_releases_held_activities():
    frames = raised_frames(20) + [Landmarks(None, None, ())] * 30
    engine = MotionEngine(provider=Replay(frames))
    events = []
    engine.on_any(events.append)
    drive(engine, len(frames))

    assert engine.active == frozenset()
    assert engine.tracking is False
    assert ("left_hand_up", Phase.END) in {(e.activity, e.phase) for e in events}


def test_a_single_dropped_detection_does_not_interrupt_a_pose():
    """Detection misses the odd frame with a person standing perfectly still.
    Ending the pose on the first miss would make every hold stutter."""
    frames = raised_frames(15) + [Landmarks(None, None, ())] + raised_frames(15)
    engine = MotionEngine(provider=Replay(frames))
    ends = []
    engine.on_end("left_hand_up", ends.append)
    drive(engine, len(frames))

    assert ends == []
    assert "left_hand_up" in engine.active


def test_run_over_an_iterable_source_and_release_on_shutdown():
    frames = raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    events = []
    engine.on_any(events.append)

    engine.run(IterableSource([IMAGE] * len(frames), fps=30.0))

    # The stream ended while the pose was held; shutdown must still emit the END
    # or a bound key would stay pressed.
    assert ("left_hand_up", Phase.END) in {(e.activity, e.phase) for e in events}
    assert engine.active == frozenset()
    assert engine.running is False


def test_snapshot_reports_progress():
    engine = MotionEngine(provider=Replay(standing_frames(20)))
    drive(engine, 20)
    snap = engine.snapshot()
    assert snap.frames == 20
    assert snap.tracking is True
    assert snap.fps > 0


def test_submit_is_rejected_while_running():
    """The detector is stateful; driving it from two threads would corrupt it."""

    def endless():
        while True:
            time.sleep(0.002)
            yield IMAGE

    running = threading.Event()
    engine = MotionEngine(provider=Replay(standing_frames(4), loop=True))
    engine.on_frame(lambda result: running.set())
    engine.start(IterableSource(endless(), fps=30.0))
    try:
        assert running.wait(5.0), "engine never processed a frame"
        with pytest.raises(RuntimeError, match="cannot submit"):
            engine.submit(IMAGE)
    finally:
        engine.stop()
    assert engine.running is False


def test_running_is_true_during_a_blocking_run():
    """`run()` owns no thread handle, so `running` has to come from the loop
    itself. A host embedding the engine in its own thread polls this."""
    observed = []

    engine = MotionEngine(provider=Replay(standing_frames(6)))
    engine.on_frame(lambda result: observed.append(engine.running))
    assert engine.running is False

    engine.run(IterableSource([IMAGE] * 6, fps=30.0))

    assert observed and all(observed)
    assert engine.running is False


def test_stop_releases_held_activities():
    """A binding holding a key needs its END event, or the key stays down after
    the engine is gone."""

    def endless():
        while True:
            time.sleep(0.002)
            yield IMAGE

    held = threading.Event()
    engine = MotionEngine(provider=Replay(raised_frames(4), loop=True))
    events = []
    engine.on_any(events.append)
    engine.on_start("left_hand_up", lambda e: held.set())
    engine.start(IterableSource(endless(), fps=30.0))
    assert held.wait(5.0), "pose never registered"
    engine.stop()

    assert ("left_hand_up", Phase.END) in {(e.activity, e.phase) for e in events}
    assert engine.active == frozenset()


# =====================================================================
# Model gating
# =====================================================================
def test_hand_model_stays_off_when_nothing_needs_it():
    """The hand model costs about as much as the pose model. Running it for a
    body-gesture setup that never asks for a finger state is pure waste."""
    provider = Replay(standing_frames(3))
    engine = MotionEngine(EngineConfig(enable_hands="auto"), provider=provider)
    engine.on("jump", lambda e: None)
    engine.submit(IMAGE, timestamp=0.0)
    assert provider.hands_enabled is False


def test_hand_model_turns_on_when_a_finger_activity_is_subscribed():
    provider = Replay(standing_frames(3))
    engine = MotionEngine(EngineConfig(enable_hands="auto"), provider=provider)
    engine.on("left_pinch", lambda e: None)
    engine.submit(IMAGE, timestamp=0.0)
    assert provider.hands_enabled is True


def test_wildcard_listener_turns_the_hand_model_on():
    provider = Replay(standing_frames(3))
    engine = MotionEngine(EngineConfig(enable_hands="auto"), provider=provider)
    engine.on_any(lambda e: None)
    engine.submit(IMAGE, timestamp=0.0)
    assert provider.hands_enabled is True


def test_explicit_false_wins_over_a_subscription():
    provider = Replay(standing_frames(3))
    engine = MotionEngine(EngineConfig(enable_hands=False), provider=provider)
    engine.on("left_pinch", lambda e: None)
    engine.submit(IMAGE, timestamp=0.0)
    assert provider.hands_enabled is False


def test_restricting_activities_skips_the_rest():
    frames = standing_frames(6) + raised_frames(15)
    config = EngineConfig(activities={"left_hand_up"})
    engine = MotionEngine(config, provider=Replay(frames))
    events = []
    engine.on_any(events.append)
    drive(engine, len(frames))

    assert {e.activity for e in events} == {"left_hand_up"}


# =====================================================================
# Extension
# =====================================================================
def test_define_a_custom_level_activity():
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    engine.define(
        "left_wrist_above_hips",
        lambda f: float(f.P[Pose.LEFT_WRIST][1]),
        trigger="level",
        enter=0.5,
        exit=0.3,
    )
    seen = []
    engine.on("left_wrist_above_hips", seen.append)
    drive(engine, len(frames))

    assert len(seen) == 1
    assert seen[0].phase is Phase.START


def test_define_a_custom_gesture():
    frames = standing_frames(6) + raised_frames(20) + standing_frames(10)
    engine = MotionEngine(provider=Replay(frames))
    engine.define(
        "salute",
        lambda f: float(f.left_hand_rise),
        trigger=Trigger.EDGE,
        enter=0.06,
        exit=0.0,
    )
    seen = []
    engine.on("salute", seen.append)
    drive(engine, len(frames))

    assert [e.phase for e in seen] == [Phase.TRIGGER]


def test_custom_activity_appears_in_the_catalog():
    from motionsense import catalog

    engine = MotionEngine(provider=Replay([]))
    engine.define("wiggle", lambda f: False, name="Wiggle", category="Custom")
    definition = catalog.get("wiggle")
    assert definition is not None and definition.name == "Wiggle"
    catalog.unregister("wiggle")


def test_a_nan_predicate_releases_rather_than_latching():
    frames = raised_frames(12) + standing_frames(12)
    engine = MotionEngine(provider=Replay(frames))
    values = iter([1.0] * 12 + [float("nan")] * 12)
    engine.define("flaky", lambda f: next(values), trigger="level", enter=0.5, exit=0.4)
    drive(engine, len(frames))
    assert "flaky" not in engine.active


def test_custom_recognizer_declares_its_signals():
    from motionsense.types import ActivityDef

    class Custom(Recognizer):
        activities = (
            ActivityDef("finger_thing", "Finger Thing", "Custom", Trigger.LEVEL, "", frozenset({Signal.HANDS})),
        )

        def update(self, f, out):
            out.level("finger_thing", False)

    provider = Replay(standing_frames(3))
    engine = MotionEngine(EngineConfig(enable_hands="auto"), provider=provider)
    engine.add_recognizer(Custom())
    engine.on("finger_thing", lambda e: None)
    engine.submit(IMAGE, timestamp=0.0)

    assert provider.hands_enabled is True
    from motionsense import catalog

    catalog.unregister("finger_thing")


# =====================================================================
# Bindings
# =====================================================================
def test_action_router_maps_activities_to_functions():
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    calls = []

    ActionRouter(engine, {"left_hand_up": lambda: calls.append("raised")})
    drive(engine, len(frames))
    assert calls == ["raised"]


def test_action_router_accepts_zero_and_one_argument_actions():
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    calls = []

    ActionRouter(
        engine,
        {
            "left_hand_up": {
                "start": lambda: calls.append("bare"),
                "end": lambda event: calls.append(f"with-event:{event.phase.value}"),
            }
        },
    )
    drive(engine, len(frames))
    assert calls == ["bare"]


def test_action_router_can_be_paused_without_unbinding():
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    calls = []

    router = ActionRouter(engine, {"left_hand_up": lambda: calls.append(1)})
    router.enabled = False
    drive(engine, len(frames))
    assert calls == []
    assert router.bound == ("left_hand_up",)


def test_action_router_load_replaces_the_whole_mapping():
    frames = standing_frames(6) + raised_frames(20)
    engine = MotionEngine(provider=Replay(frames))
    first, second = [], []

    router = ActionRouter(engine, {"left_hand_up": lambda: first.append(1)})
    router.load({"left_hand_up": lambda: second.append(1)})
    drive(engine, len(frames))

    assert first == []
    assert second == [1]
