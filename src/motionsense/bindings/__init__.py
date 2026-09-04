"""Ways to turn events into effects.

Both are optional and entirely separate from detection: the engine emits events
and knows nothing about what consumes them.

* :class:`~motionsense.bindings.router.ActionRouter` -- activities to your own
  functions, as a swappable mapping.
* :class:`~motionsense.bindings.keys.KeyBindings` -- activities to real key
  presses. Needs ``pip install motionsense[keys]``.
"""

from .router import ActionRouter

__all__ = ["ActionRouter", "KeyBindings"]


def __getattr__(name: str):
    if name == "KeyBindings":
        from .keys import KeyBindings

        return KeyBindings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
