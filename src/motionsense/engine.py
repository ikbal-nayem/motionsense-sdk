"""The engine: video in, activity events out."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable

import numpy as np

from .calibration import Calibration, CalibrationCollector
from .catalog import register, require
from .config import EngineConfig
from .dispatcher import Dispatcher, Subscription
from .features import FeatureExtractor
from .providers.base import LandmarkProvider
from .recognizers import Recognizer, Sink, default_recognizers
from .recognizers.base import PredicateRecognizer
from .sources.base import SourceError, VideoSource
from .types import ActivityDef, Event, EventHandler, FrameResult, Phase, Signal, Snapshot, Trigger

__all__ = ["MotionEngine"]

log = logging.getLogger("motionsense")

#: How long a body may go undetected before held activities are released. Below
#: this the previous state is carried forward, so one missed detection in a
#: 30 FPS stream does not interrupt a held pose.
_TRACKING_GRACE = 0.35


class MotionEngine:
    """Detects human activities in a video stream and reports them to listeners.

    The engine owns the pipeline -- capture, inference, feature extraction,
    recognition, dispatch -- and exposes it as a subscription API::

        engine = MotionEngine()

        @engine.on("jump")
        def jumped(event):
            fire_cannon()

        @engine.on_start("left_hand_up")
        def raised(event):
            thrusters.on()

        @engine.on_end("left_hand_up")
        def lowered(event):
            thrusters.off()

        engine.start()          # background threads; returns immediately
        ...
        engine.stop()

    What a listener does is entirely up to the caller: press a key, call a
    function, publish MQTT, move a robot arm. The SDK has no opinion about it,
    and nothing in the detection path knows what a keyboard is.

    Threading: capture runs on its own thread so inference always sees the newest
    frame; recognition runs on a second thread. Listeners are called on the
    recognition thread by default, which is the lowest-latency option. If your
    listeners do I/O, set ``dispatch="thread"`` so they cannot stall detection.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        provider: LandmarkProvider | None = None,
        recognizers: Iterable[Recognizer] | None = None,
    ):
        self.config = config or EngineConfig()
        self._provider = provider
        self._owns_provider = provider is None
        self._recognizers: list[Recognizer] = list(
            recognizers if recognizers is not None else default_recognizers()
        )
        self._extractor = FeatureExtractor(self.config)
        self._dispatcher = Dispatcher(
            emit_updates=self.config.emit_updates,
            threaded=self.config.dispatch == "thread",
        )
        self._dispatcher.set_change_hook(self._refresh_signals)

        self._frame_listeners: list[Callable[[FrameResult], None]] = []
        self._error_listeners: list[Callable[[Exception], None]] = []

        self._thread: threading.Thread | None = None
        # Set by the loop itself rather than by start(), so `running` is also
        # true during a blocking run() -- which owns no thread handle.
        self._loop_active = False
        self._stop = threading.Event()
        self._source: VideoSource | None = None
        self._lock = threading.RLock()

        self._t0: float | None = None
        self._last_tracked: float | None = None
        self._released = True
        self._tracking = False
        self._configured = False

        self._collector: CalibrationCollector | None = None
        self._calibrated = threading.Event()

        self._frames = 0
        self._fps = 0.0
        self._latency = 0.0
        self._last_frame_t: float | None = None

    # =====================================================================
    # Listener registration
    # =====================================================================
    def on(
        self,
        activity: str | Iterable[str] | None = None,
        callback: EventHandler | None = None,
        *,
        phase: Phase | Iterable[Phase] | None = None,
    ):
        """Subscribe to an activity's natural onset.

        For a gesture that is ``Phase.TRIGGER``; for a pose it is ``Phase.START``.
        Pass ``phase=`` to choose explicitly.

        Works as a decorator or a direct call::

            @engine.on("wave_right_hand")
            def waved(event): ...

            sub = engine.on("jump", lambda e: print(e))
            sub.cancel()

        Unknown activity ids raise immediately rather than silently never firing.
        """
        resolved = self._resolve_phase(activity, phase)

        def subscribe(fn: EventHandler):
            self._validate(activity)
            sub = self._dispatcher.add(activity, resolved, fn)
            try:
                fn.subscription = sub  # type: ignore[attr-defined]
            except AttributeError:
                pass
            return fn

        if callback is not None:
            self._validate(activity)
            return self._dispatcher.add(activity, resolved, callback)
        return subscribe

    def on_start(self, activity=None, callback=None):
        """Fires when a pose begins."""
        return self.on(activity, callback, phase=Phase.START)

    def on_end(self, activity=None, callback=None):
        """Fires when a pose is released. ``event.duration`` says how long it was held."""
        return self.on(activity, callback, phase=Phase.END)

    def on_update(self, activity=None, callback=None):
        """Fires every frame a pose is held. Requires ``emit_updates=True``."""
        if not self.config.emit_updates:
            log.warning("on_update listener added but EngineConfig.emit_updates is False")
        return self.on(activity, callback, phase=Phase.UPDATE)

    def on_trigger(self, activity=None, callback=None):
        """Fires once per recognised gesture."""
        return self.on(activity, callback, phase=Phase.TRIGGER)

    def on_any(self, callback: EventHandler) -> Subscription:
        """Receive every event of every kind. Useful for logging and UIs."""
        return self._dispatcher.add(None, None, callback)

    def on_frame(self, callback: Callable[[FrameResult], None]) -> Subscription:
        """Receive each processed frame, for preview rendering or custom analysis.

        Set ``EngineConfig.deliver_frames=True`` to have the image attached.
        """
        self._frame_listeners.append(callback)
        return Subscription(lambda: _discard(self._frame_listeners, callback))

    def on_error(self, callback: Callable[[Exception], None]) -> Subscription:
        """Receive failures from the capture/inference loop. The loop stops after one."""
        self._error_listeners.append(callback)
        return Subscription(lambda: _discard(self._error_listeners, callback))

    # =====================================================================
    # Lifecycle
    # =====================================================================
    def start(self, source: VideoSource | int | str | None = None) -> "MotionEngine":
        """Begin processing on background threads and return immediately.

        ``source`` accepts a :class:`~motionsense.sources.base.VideoSource`, a
        camera index, a file path, or ``None`` for the default camera.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("engine is already running")
            resolved = _coerce_source(source)
            self._stop.clear()
            self._source = resolved
            self._thread = threading.Thread(
                target=self._run_loop, args=(resolved,), name="motionsense-engine", daemon=True
            )
            if self.config.dispatch == "thread":
                self._dispatcher.start_worker()
            self._thread.start()
        return self

    def run(self, source: VideoSource | int | str | None = None) -> None:
        """Process on the calling thread until the source ends or :meth:`stop` is called."""
        resolved = _coerce_source(source)
        self._stop.clear()
        self._source = resolved
        if self.config.dispatch == "thread":
            self._dispatcher.start_worker()
        self._run_loop(resolved)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop processing and release every held activity.

        The release matters: a binding holding a key down needs its END event, or
        the key stays pressed after the engine is gone. Safe to call from inside
        a listener -- it will not try to join its own thread.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        if self._loop_active:
            return True
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "MotionEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # =====================================================================
    # Push mode
    # =====================================================================
    def submit(self, image: np.ndarray, timestamp: float | None = None) -> FrameResult:
        """Process one frame synchronously and return what it produced.

        For applications that already have a video pipeline and want the engine
        as a pure function of frames -- no threads, no capture, fully
        deterministic. Do not mix with :meth:`start`; the detector is stateful and
        not safe to drive from two threads.

        ``timestamp`` should be a monotonically increasing time in seconds. It
        defaults to the wall clock, which is right for live frames but wrong for
        faster-than-real-time playback -- pass the media timeline there, or every
        velocity will be scaled by the playback rate.
        """
        if self.running:
            raise RuntimeError("cannot submit() while the engine is running; use stop() first")
        self._prepare()
        stamp = time.perf_counter() if timestamp is None else float(timestamp)
        self._frames += 1
        return self._process(image, stamp, self._frames)

    # =====================================================================
    # State
    # =====================================================================
    @property
    def active(self) -> frozenset[str]:
        """Activity ids currently held."""
        return self._dispatcher.active

    @property
    def tracking(self) -> bool:
        """Whether a body is currently detected."""
        return self._tracking

    def snapshot(self) -> Snapshot:
        """Thread-safe view of engine state, cheap enough to poll from a UI."""
        source = self._source
        return Snapshot(
            running=self.running,
            active=self._dispatcher.active,
            fps=self._fps,
            latency=self._latency,
            frames=self._frames,
            dropped=source.dropped if source is not None else 0,
            tracking=self._tracking,
        )

    # =====================================================================
    # Calibration
    # =====================================================================
    @property
    def calibration(self) -> Calibration:
        return self._extractor.calibration

    @calibration.setter
    def calibration(self, value: Calibration) -> None:
        self._extractor.calibration = value

    def calibrate(self, seconds: float = 2.0, *, wait: bool = False, timeout: float = 15.0) -> Calibration:
        """Record the subject's neutral standing pose as a personal reference.

        Have the subject stand still, facing the camera, arms down. Thresholds
        that depend on body proportion or camera angle then key off this instead
        of a population average. Everything works without it; this makes it fit.

        With ``wait=True`` this blocks until collection finishes, so call it from
        a thread other than the one running the engine.
        """
        self._calibrated.clear()
        self._collector = CalibrationCollector(seconds)
        if wait:
            if not self._calibrated.wait(timeout):
                self._collector = None
                raise TimeoutError("calibration timed out; is a body visible to the camera?")
        return self._extractor.calibration

    # =====================================================================
    # Extension
    # =====================================================================
    def add_recognizer(self, recognizer: Recognizer) -> Recognizer:
        """Attach a custom recognizer and register the activities it declares."""
        for activity in recognizer.activities:
            try:
                register(activity)
            except ValueError:
                pass
        recognizer.configure(self.config.tuning)
        self._recognizers.append(recognizer)
        self._refresh_signals()
        return recognizer

    @property
    def recognizers(self) -> tuple[Recognizer, ...]:
        return tuple(self._recognizers)

    def define(
        self,
        activity_id: str,
        fn: Callable,
        *,
        trigger: Trigger | str = Trigger.LEVEL,
        name: str | None = None,
        category: str = "Custom",
        description: str = "",
        requires: Iterable[Signal] = (Signal.POSE,),
        enter: float = 0.5,
        exit: float = 0.4999,
        min_on: float | None = None,
        min_off: float | None = None,
        cooldown: float = 0.4,
    ) -> Recognizer:
        """Define a new activity from a function of the body features.

        The quickest way to teach the SDK something it does not know::

            engine.define(
                "hand_over_heart",
                lambda f: -abs(float(f.P[Pose.LEFT_WRIST][0]) - 0.15),
                trigger="level", enter=-0.12, exit=-0.20,
            )

        ``fn`` receives :class:`~motionsense.features.BodyFeatures` -- already
        normalised, filtered and visibility-tagged -- and returns a number or a
        bool. Return ``float('nan')`` when the frame cannot answer. The result
        gets the same hysteresis, debouncing and event semantics as a built-in.
        """
        definition = ActivityDef(
            id=activity_id,
            name=name or activity_id.replace("_", " ").title(),
            category=category,
            trigger=Trigger(trigger) if isinstance(trigger, str) else trigger,
            description=description,
            requires=frozenset(requires),
        )
        try:
            register(definition)
        except ValueError:
            register(definition, replace=True)
        return self.add_recognizer(
            PredicateRecognizer(
                definition, fn, enter=enter, exit=exit, min_on=min_on, min_off=min_off, cooldown=cooldown
            )
        )

    # =====================================================================
    # Internals
    # =====================================================================
    def _validate(self, activity) -> None:
        if activity is None:
            return
        for aid in (activity,) if isinstance(activity, str) else activity:
            require(aid)

    @staticmethod
    def _resolve_phase(activity, phase):
        if phase is not None:
            return phase
        if activity is None:
            return (Phase.START, Phase.TRIGGER)
        ids = (activity,) if isinstance(activity, str) else tuple(activity)
        phases = set()
        for aid in ids:
            definition = require(aid)
            phases.add(Phase.TRIGGER if definition.is_edge else Phase.START)
        return tuple(phases)

    def _prepare(self) -> None:
        """Create the provider and configure recognizers, once per run."""
        if self._provider is None:
            from .providers.auto import create_provider

            self._provider = create_provider(self.config)
        if not self._configured:
            for recognizer in self._recognizers:
                recognizer.configure(self.config.tuning)
            self._configured = True
        self._refresh_signals()

    def _refresh_signals(self) -> None:
        """Decide whether the hand model is worth running.

        The hand model costs about as much per frame as the pose model. In
        ``"auto"`` mode it only runs when something is actually listening for a
        finger state, which for a typical body-gesture setup is never -- so the
        default configuration is roughly twice as fast as one that always runs
        both.
        """
        provider = self._provider
        if provider is None:
            return
        setting = self.config.enable_hands
        if setting is True:
            provider.set_hands_enabled(True)
            return
        if setting is False:
            provider.set_hands_enabled(False)
            return

        hand_ids = [
            a.id
            for recognizer in self._recognizers
            for a in recognizer.activities
            if Signal.HANDS in a.requires and self.config.wants(a.id)
        ]
        if not hand_ids:
            provider.set_hands_enabled(False)
            return
        if self._dispatcher.has_wildcard():
            provider.set_hands_enabled(True)
            return
        provider.set_hands_enabled(any(self._dispatcher.has_listener(i) for i in hand_ids))

    def _run_loop(self, source: VideoSource) -> None:
        self._loop_active = True
        try:
            source.open()
        except Exception as exc:
            self._loop_active = False
            self._emit_error(exc)
            return

        try:
            self._prepare()
            self._t0 = None
            self._last_tracked = None
            self._released = True

            while not self._stop.is_set():
                try:
                    frame = source.read(timeout=0.4)
                except SourceError as exc:
                    self._emit_error(exc)
                    break
                except Exception as exc:  # pragma: no cover - source bug
                    self._emit_error(exc)
                    break

                if frame is None:
                    if source.is_live:
                        continue
                    break

                self._frames += 1
                try:
                    self._process(frame.image, frame.captured_at, frame.index)
                except Exception as exc:  # pragma: no cover - recognizer bug
                    log.exception("frame processing failed")
                    self._emit_error(exc)
                    break
        finally:
            self._shutdown(source)

    def _shutdown(self, source: VideoSource) -> None:
        try:
            events = self._dispatcher.release_all(self._now(), time.time())
            if events:
                self._dispatcher.deliver(events)
        finally:
            self._dispatcher.stop_worker()
            try:
                source.close()
            except Exception:
                log.exception("source close failed")
            if self._owns_provider and self._provider is not None:
                self._provider.close()
                self._provider = None
                self._configured = False
            for recognizer in self._recognizers:
                recognizer.reset()
            self._extractor.reset()
            self._tracking = False
            self._loop_active = False
            self._stop.set()

    def _process(self, image: np.ndarray, captured_at: float, index: int) -> FrameResult:
        started = time.perf_counter()
        if self._t0 is None:
            self._t0 = captured_at
        # Engine-relative time: small magnitudes keep the least-squares
        # regressions well conditioned, and it starts at zero for both a live
        # camera and a file.
        t = captured_at - self._t0
        wall = time.time()

        landmarks = self._provider.process(image, t, index)
        height, width = image.shape[0], image.shape[1]
        features = self._extractor.update(
            landmarks.pose, landmarks.pose_world, landmarks.hands, t, width / max(height, 1)
        )

        events: tuple[Event, ...] = ()
        levels: dict = {}
        if features is None:
            events = self._handle_tracking_loss(t, wall, index)
        else:
            self._tracking = True
            self._last_tracked = t
            self._released = False
            self._collect_calibration(features, t)

            sink = Sink(self.config.activities)
            for recognizer in self._recognizers:
                recognizer.update(features, sink)
            levels = sink.levels
            events = self._dispatcher.build(sink, t, wall, index)
            if events:
                self._dispatcher.deliver(events)

        self._update_stats(t, started, captured_at)

        result = FrameResult(
            index=index,
            timestamp=t,
            wall_time=wall,
            pose=landmarks.pose,
            hands=landmarks.hands,
            active=self._dispatcher.active,
            events=events,
            image=image if self.config.deliver_frames else None,
            latency=time.perf_counter() - started,
            levels=levels,
        )
        for listener in tuple(self._frame_listeners):
            try:
                listener(result)
            except Exception:
                log.exception("frame listener raised")
        return result

    def _handle_tracking_loss(self, t: float, wall: float, index: int) -> tuple[Event, ...]:
        """Hold state briefly, then release everything.

        Detection drops the odd frame even with a person standing still. Ending
        every held activity on the first miss would make poses stutter; never
        ending them would leave a key held after the user walks away. The grace
        window is the compromise.
        """
        self._tracking = False
        if self._released:
            return ()
        if self._last_tracked is not None and t - self._last_tracked <= _TRACKING_GRACE:
            return ()

        self._released = True
        events = self._dispatcher.build(Sink(self.config.activities), t, wall, index)
        if events:
            self._dispatcher.deliver(events)
        for recognizer in self._recognizers:
            recognizer.reset()
        self._extractor.reset()
        return events

    def _collect_calibration(self, features, t: float) -> None:
        collector = self._collector
        if collector is None:
            return
        if not collector.started:
            collector.begin(t)
        collector.add(features, t)
        if collector.done(t):
            self._extractor.calibration = collector.finish()
            self._collector = None
            self._calibrated.set()

    def _update_stats(self, t: float, started: float, captured_at: float) -> None:
        if self._last_frame_t is not None:
            dt = t - self._last_frame_t
            if dt > 1e-6:
                instant = 1.0 / dt
                # ~1 second time constant, so the reading is steady enough to
                # display but still reacts to a real slowdown.
                self._fps = instant if self._fps == 0.0 else self._fps + 0.15 * (instant - self._fps)
        self._last_frame_t = t

        source = self._source
        now = time.perf_counter()
        # For a live source, latency is measured from the moment the frame was
        # grabbed, so it includes any queueing. For a file there is no such
        # clock, so it is just processing time.
        sample = (now - captured_at) if (source is not None and source.is_live) else (now - started)
        self._latency = sample if self._latency == 0.0 else self._latency + 0.15 * (sample - self._latency)

    def _now(self) -> float:
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    def _emit_error(self, exc: Exception) -> None:
        log.error("engine error: %s", exc)
        for listener in tuple(self._error_listeners):
            try:
                listener(exc)
            except Exception:
                log.exception("error listener raised")


def _coerce_source(source) -> VideoSource:
    from .sources.camera import CameraSource
    from .sources.video import VideoFileSource

    if source is None:
        return CameraSource(0)
    if isinstance(source, VideoSource):
        return source
    if isinstance(source, int):
        return CameraSource(source)
    if isinstance(source, str):
        return VideoFileSource(source)
    raise TypeError(f"cannot use {type(source).__name__} as a video source")


def _discard(bucket: list, item) -> None:
    try:
        bucket.remove(item)
    except ValueError:
        pass
