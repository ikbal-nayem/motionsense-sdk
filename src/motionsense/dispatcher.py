"""Listener registry and the level/edge state machine that produces events."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Iterable

from .types import Event, EventHandler, Phase

__all__ = ["Dispatcher", "Subscription"]

log = logging.getLogger("motionsense")

#: Ends are delivered before starts within a frame, so a binding releases the
#: key it was holding before pressing the next one. The reverse order briefly
#: holds both, which reads as a chord to whatever is listening.
_PHASE_ORDER = {Phase.END: 0, Phase.START: 1, Phase.UPDATE: 2, Phase.TRIGGER: 3}


class Subscription:
    """Handle returned by every ``on_*`` call. Call it, or :meth:`cancel`, to unsubscribe."""

    __slots__ = ("_cancel", "_active")

    def __init__(self, cancel):
        self._cancel = cancel
        self._active = True

    def cancel(self) -> None:
        if self._active:
            self._active = False
            self._cancel()

    __call__ = cancel

    @property
    def active(self) -> bool:
        return self._active


class Dispatcher:
    """Converts recognizer output into events and delivers them to listeners."""

    def __init__(self, *, emit_updates: bool = False, threaded: bool = False):
        self.emit_updates = emit_updates
        self._by_activity: dict[str, dict[Phase, list[EventHandler]]] = {}
        self._by_phase: dict[Phase, list[EventHandler]] = {}
        self._any: list[EventHandler] = []
        self._active: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._on_change = None

        self._threaded = threaded
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        if threaded:
            self._queue = queue.Queue(maxsize=512)

    # -- registration ----------------------------------------------------------
    def set_change_hook(self, fn) -> None:
        """Called whenever the listener set changes, so the engine can revisit
        which models are worth running."""
        self._on_change = fn

    def add(
        self,
        activities: Iterable[str] | str | None,
        phases: Iterable[Phase] | Phase | None,
        callback: EventHandler,
    ) -> Subscription:
        if isinstance(activities, str):
            activities = (activities,)
        if isinstance(phases, Phase):
            phases = (phases,)

        with self._lock:
            if activities is None and phases is None:
                self._any.append(callback)
                removers = [lambda: _discard(self._any, callback)]
            elif activities is None:
                removers = []
                for phase in phases:
                    bucket = self._by_phase.setdefault(phase, [])
                    bucket.append(callback)
                    removers.append(lambda b=bucket: _discard(b, callback))
            else:
                removers = []
                phase_list = tuple(phases) if phases is not None else tuple(Phase)
                for activity in activities:
                    table = self._by_activity.setdefault(activity, {})
                    for phase in phase_list:
                        bucket = table.setdefault(phase, [])
                        bucket.append(callback)
                        removers.append(lambda b=bucket: _discard(b, callback))

        self._notify_change()

        def cancel():
            with self._lock:
                for remove in removers:
                    remove()
            self._notify_change()

        return Subscription(cancel)

    def clear(self) -> None:
        with self._lock:
            self._by_activity.clear()
            self._by_phase.clear()
            self._any.clear()
        self._notify_change()

    def has_listener(self, activity_id: str) -> bool:
        with self._lock:
            table = self._by_activity.get(activity_id)
            return bool(table and any(table.values()))

    def has_wildcard(self) -> bool:
        with self._lock:
            return bool(self._any) or any(self._by_phase.values())

    def _notify_change(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:  # pragma: no cover - hook must never break registration
                log.exception("listener-change hook failed")

    # -- event production --------------------------------------------------------
    @property
    def active(self) -> frozenset[str]:
        return frozenset(self._active)

    def build(self, sink, t: float, wall: float, frame_index: int) -> tuple[Event, ...]:
        """Diff this frame's levels against the held set and emit the events."""
        events: list[Event] = []
        levels = sink.levels
        now_active = {aid for aid, (state, _, _) in levels.items() if state}

        for aid in tuple(self._active):
            if aid not in now_active:
                started, _ = self._active.pop(aid)
                events.append(
                    Event(aid, Phase.END, t, wall, 1.0, duration=t - started, frame_index=frame_index)
                )

        want_updates = self.emit_updates
        for aid in now_active:
            _, confidence, data = levels[aid]
            existing = self._active.get(aid)
            if existing is None:
                self._active[aid] = (t, confidence)
                events.append(Event(aid, Phase.START, t, wall, confidence, 0.0, data, frame_index))
            elif want_updates:
                events.append(
                    Event(aid, Phase.UPDATE, t, wall, confidence, t - existing[0], data, frame_index)
                )

        for aid, confidence, data in sink.edges:
            events.append(Event(aid, Phase.TRIGGER, t, wall, confidence, 0.0, data, frame_index))

        events.sort(key=lambda e: _PHASE_ORDER[e.phase])
        return tuple(events)

    def release_all(self, t: float, wall: float) -> tuple[Event, ...]:
        """Emit END for everything still held. Used when the engine stops, so no
        binding is left with a key down."""
        events = [
            Event(aid, Phase.END, t, wall, 1.0, duration=t - started)
            for aid, (started, _) in self._active.items()
        ]
        self._active.clear()
        return tuple(events)

    # -- delivery ------------------------------------------------------------------
    def deliver(self, events: Iterable[Event]) -> None:
        if self._threaded:
            for event in events:
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    # Better to lose an event than to stall detection behind a
                    # listener that cannot keep up.
                    log.warning("event queue full; dropping %s", event.activity)
            return
        for event in events:
            self._invoke(event)

    def _invoke(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._any)
            handlers += self._by_phase.get(event.phase, ())
            table = self._by_activity.get(event.activity)
            if table:
                handlers += table.get(event.phase, ())
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # One bad listener must not take down the detection loop.
                log.exception("listener for %s.%s raised", event.activity, event.phase.value)

    # -- threaded mode ---------------------------------------------------------------
    def start_worker(self) -> None:
        if not self._threaded or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._drain, name="motionsense-dispatch", daemon=True)
        self._worker.start()

    def stop_worker(self, timeout: float = 2.0) -> None:
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        self._worker = None

    def _drain(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            self._invoke(event)


def _discard(bucket: list, item) -> None:
    try:
        bucket.remove(item)
    except ValueError:
        pass
