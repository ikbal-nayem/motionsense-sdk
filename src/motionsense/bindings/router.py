"""Bulk, swappable mapping from activities to your own functions."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

from ..dispatcher import Subscription
from ..types import Event, Phase

__all__ = ["ActionRouter"]

log = logging.getLogger("motionsense")

Action = Callable[[Event], None] | Callable[[], None]


class ActionRouter:
    """Routes activities to callables, as one swappable set.

    ``engine.on`` is enough for a handful of listeners written in code. This is
    for the case where the mapping is *data* -- loaded from a config file, edited
    in a UI, switched when the user changes context -- which is the shape most
    control applications end up needing::

        router = ActionRouter(engine)
        router.load({
            "swipe_left":  library.previous_track,
            "swipe_right": library.next_track,
            "left_pinch":  {"start": grab, "end": release},
            "jump":        lambda e: log(f"jumped at {e.confidence:.2f}"),
        })

        router.load(DRIVING_PROFILE)   # atomically replaces the whole mapping
        router.enabled = False         # pause without tearing anything down

    Actions may take the :class:`~motionsense.types.Event` or no arguments; both
    are detected and called correctly.
    """

    def __init__(self, engine, mapping: Mapping[str, Action | Mapping[str, Action]] | None = None):
        self._engine = engine
        self._subs: dict[str, list[Subscription]] = {}
        self._enabled = True
        if mapping:
            self.load(mapping)

    # -- state ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Master switch. Subscriptions stay in place; actions just stop running.

        Cheaper and less error-prone than unbinding, and it cannot lose a
        pending release: a held activity's END still arrives, it is simply
        ignored.
        """
        self._enabled = bool(value)

    @property
    def bound(self) -> tuple[str, ...]:
        return tuple(self._subs)

    # -- binding ------------------------------------------------------------------
    def load(self, mapping: Mapping[str, Action | Mapping[str, Action]]) -> "ActionRouter":
        """Replace every binding with ``mapping``."""
        self.clear()
        for activity, spec in mapping.items():
            if isinstance(spec, Mapping):
                for phase_name, action in spec.items():
                    self.bind(activity, action, phase=Phase(phase_name))
            else:
                self.bind(activity, spec)
        return self

    def bind(self, activity: str, action: Action, *, phase: Phase | None = None) -> "ActionRouter":
        """Bind one activity. Defaults to its natural onset phase."""
        handler = self._wrap(action)
        sub = self._engine.on(activity, handler, phase=phase)
        self._subs.setdefault(activity, []).append(sub)
        return self

    def unbind(self, activity: str) -> None:
        for sub in self._subs.pop(activity, ()):
            sub.cancel()

    def clear(self) -> None:
        for subs in self._subs.values():
            for sub in subs:
                sub.cancel()
        self._subs.clear()

    # -- internals -------------------------------------------------------------------
    def _wrap(self, action: Action) -> Callable[[Event], None]:
        takes_event = _accepts_argument(action)

        def handler(event: Event) -> None:
            if not self._enabled:
                return
            try:
                action(event) if takes_event else action()
            except Exception:
                log.exception("action for %s raised", event.activity)

        return handler


def _accepts_argument(fn: Callable) -> bool:
    """Whether ``fn`` can be called with one positional argument.

    Lets a mapping mix ``lambda e: ...`` with a bare ``player.pause``, which is
    what a hand-written config actually looks like. Falls back to passing the
    event when the signature cannot be read (builtins, C extensions).
    """
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif parameter.kind is parameter.VAR_POSITIONAL:
            return True
    return positional >= 1
