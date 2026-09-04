"""Keyboard and mouse output.

Optional: ``pip install motionsense[keys]``. Nothing in the detection path
depends on this module -- it is one possible consumer of events, and the SDK
works identically without it.
"""

from __future__ import annotations

import logging
import threading
from typing import Mapping

from ..catalog import require
from ..types import Event, Phase, Trigger

__all__ = ["KeyBindings"]

log = logging.getLogger("motionsense")


def _keyboard():
    try:
        import keyboard
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "keyboard output needs the `keyboard` package: pip install motionsense[keys]"
        ) from exc
    return keyboard


class KeyBindings:
    """Drives real key presses from activities.

    Poses are *held* (key down on start, up on end) and gestures are *tapped*, so
    a mapping is usually just ``{activity: key}``::

        keys = KeyBindings(engine, {"left_hand_up": "a", "jump": "space"})

    Two details this gets right that a naive implementation does not:

    **Reference-counted holds.** Several activities may map to the same key. The
    key goes down when the first one starts and comes up only when the last one
    ends, so overlapping poses cannot release a key another pose is still
    holding.

    **Non-blocking taps.** A tap needs a gap between press and release or the
    target application coalesces it away. Sleeping for that gap on the calling
    thread would stall the detection loop for the duration -- with inline
    dispatch that is a dropped frame per tap. The release is scheduled on a timer
    instead, so the detector never waits.
    """

    def __init__(
        self,
        engine,
        mapping: Mapping[str, str] | None = None,
        *,
        tap_duration: float = 0.045,
        autostart: bool = True,
    ):
        self._engine = engine
        self._tap_duration = tap_duration
        self._holders: dict[str, set[str]] = {}
        self._keys: dict[str, str] = {}
        self._subs: list = []
        self._timers: set[threading.Timer] = set()
        self._lock = threading.RLock()
        self._enabled = True
        self._kb = _keyboard()

        if mapping:
            self.load(mapping, autostart=autostart)

    # -- state -------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value == self._enabled:
            return
        self._enabled = value
        if not value:
            # Never leave a key physically down when output is switched off.
            self.release_all()

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._keys)

    # -- binding -------------------------------------------------------------------
    def load(self, mapping: Mapping[str, str], *, autostart: bool = True) -> "KeyBindings":
        """Replace all bindings. Chooses hold or tap from each activity's trigger."""
        self.clear()
        for activity, key in mapping.items():
            definition = require(activity)
            if definition.trigger is Trigger.EDGE:
                self.tap(activity, key)
            else:
                self.hold(activity, key)
        if autostart and not self._engine.running:
            log.debug("key bindings loaded; engine not started yet")
        return self

    def hold(self, activity: str, key: str) -> "KeyBindings":
        """Press ``key`` while the activity is held, release when it ends."""
        require(activity)
        self._keys[activity] = key
        self._subs.append(self._engine.on(activity, self._make_press(activity, key), phase=Phase.START))
        self._subs.append(self._engine.on(activity, self._make_release(activity, key), phase=Phase.END))
        return self

    def tap(self, activity: str, key: str) -> "KeyBindings":
        """Press and release ``key`` once per occurrence."""
        require(activity)
        self._keys[activity] = key
        phase = Phase.TRIGGER if require(activity).is_edge else Phase.START
        self._subs.append(self._engine.on(activity, self._make_tap(key), phase=phase))
        return self

    def clear(self) -> None:
        for sub in self._subs:
            sub.cancel()
        self._subs.clear()
        self._keys.clear()
        self.release_all()

    def release_all(self) -> None:
        """Lift every key this object is holding."""
        with self._lock:
            keys = [key for key, holders in self._holders.items() if holders]
            self._holders.clear()
            timers, self._timers = list(self._timers), set()
        for timer in timers:
            timer.cancel()
        for key in keys:
            self._send(self._kb.release, key)

    # -- handlers --------------------------------------------------------------------
    def _make_press(self, activity: str, key: str):
        def handler(event: Event) -> None:
            if not self._enabled:
                return
            with self._lock:
                holders = self._holders.setdefault(key, set())
                first = not holders
                holders.add(activity)
            if first:
                self._send(self._kb.press, key)

        return handler

    def _make_release(self, activity: str, key: str):
        def handler(event: Event) -> None:
            with self._lock:
                holders = self._holders.get(key)
                if not holders:
                    return
                holders.discard(activity)
                if holders:
                    return
            self._send(self._kb.release, key)

        return handler

    def _make_tap(self, key: str):
        def handler(event: Event) -> None:
            if not self._enabled:
                return
            self._send(self._kb.press, key)
            timer = threading.Timer(self._tap_duration, self._finish_tap, args=(key,))
            timer.daemon = True
            with self._lock:
                self._timers.add(timer)
            timer.start()

        return handler

    def _finish_tap(self, key: str) -> None:
        with self._lock:
            self._timers = {t for t in self._timers if t.is_alive()}
        self._send(self._kb.release, key)

    def _send(self, fn, key: str) -> None:
        try:
            fn(key)
        except Exception:
            # A bad key name or a permissions failure must not kill the engine.
            log.exception("could not send key %r", key)

    # -- lifecycle ---------------------------------------------------------------------
    def __enter__(self) -> "KeyBindings":
        return self

    def __exit__(self, *exc) -> None:
        self.clear()
