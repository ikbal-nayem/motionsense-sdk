"""The activity catalog: what the SDK can recognise, and how to extend it.

Ids are stable and match the original MotionKey activity set one-for-one, so an
existing key-mapping configuration keeps working unchanged. Ids added by this
SDK are listed under "extensions" below.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .types import ActivityDef, Signal, Trigger

__all__ = [
    "BUILTIN",
    "all_activities",
    "categories",
    "get",
    "ids",
    "register",
    "require",
    "unregister",
]

_POSE = frozenset({Signal.POSE})
_HANDS = frozenset({Signal.HANDS})


def _level(id_: str, name: str, category: str, description: str, requires=_POSE) -> ActivityDef:
    return ActivityDef(id_, name, category, Trigger.LEVEL, description, requires)


def _edge(id_: str, name: str, category: str, description: str, requires=_POSE) -> ActivityDef:
    return ActivityDef(id_, name, category, Trigger.EDGE, description, requires)


BUILTIN: tuple[ActivityDef, ...] = (
    # -- Arms and hands (pose-level) ------------------------------------------
    _level("both_hands_up", "Both Hands Up", "Arms", "Both wrists raised above head height."),
    _level("left_hand_up", "Left Hand Up", "Arms", "Left wrist raised above head height."),
    _level("right_hand_up", "Right Hand Up", "Arms", "Right wrist raised above head height."),
    _level("hands_down", "Hands Down", "Arms", "Both hands resting below head height."),
    _level("t_pose", "T-Pose", "Arms", "Both arms straight and horizontal, out to the sides."),
    _level("arms_crossed", "Arms Crossed", "Arms", "Wrists crossed over the midline in front of the chest."),
    # -- Posture ---------------------------------------------------------------
    _level("lean_left", "Lean Left", "Posture", "Torso tilted toward the subject's left."),
    _level("lean_right", "Lean Right", "Posture", "Torso tilted toward the subject's right."),
    _level("squat", "Squat / Crouch", "Posture", "Knees bent, hips lowered."),
    _level("head_down", "Head Down", "Posture", "Head lowered or bowed toward the chest."),
    # -- Motion (gestures) -----------------------------------------------------
    _edge("jump", "Jump", "Motion", "A ballistic upward launch followed by a landing."),
    _edge("sit_down", "Sit Down", "Motion", "Transition from standing into a lowered posture."),
    _edge("stand_up", "Stand Up", "Motion", "Transition from a lowered posture back to standing."),
    _edge("wave_left_hand", "Wave Left Hand", "Motion", "Left hand raised and swung side to side."),
    _edge("wave_right_hand", "Wave Right Hand", "Motion", "Right hand raised and swung side to side."),
    # -- Motion extensions -----------------------------------------------------
    _edge("swipe_left", "Swipe Left", "Motion", "A single fast hand sweep toward the subject's left."),
    _edge("swipe_right", "Swipe Right", "Motion", "A single fast hand sweep toward the subject's right."),
    _edge("swipe_up", "Swipe Up", "Motion", "A single fast upward hand sweep."),
    _edge("swipe_down", "Swipe Down", "Motion", "A single fast downward hand sweep."),
    # -- Finger tracking (needs the hand model) --------------------------------
    _level("left_fist_closed", "Left Fist Closed", "Hands", "Left hand closed into a fist.", _HANDS),
    _level("left_fist_open", "Left Fist Open", "Hands", "Left hand open with fingers extended.", _HANDS),
    _level("right_fist_closed", "Right Fist Closed", "Hands", "Right hand closed into a fist.", _HANDS),
    _level("right_fist_open", "Right Fist Open", "Hands", "Right hand open with fingers extended.", _HANDS),
    _level("left_pinch", "Left Pinch", "Hands", "Left thumb and index fingertip touching.", _HANDS),
    _level("right_pinch", "Right Pinch", "Hands", "Right thumb and index fingertip touching.", _HANDS),
)

_registry: dict[str, ActivityDef] = {a.id: a for a in BUILTIN}


def register(activity: ActivityDef, *, replace: bool = False) -> ActivityDef:
    """Add a custom activity so it shows up in listings and event metadata.

    Recognizers registered through :meth:`~motionsense.engine.MotionEngine.add_recognizer`
    do this automatically for the activities they declare.
    """
    if activity.id in _registry and not replace:
        raise ValueError(
            f"activity {activity.id!r} is already registered; pass replace=True to override"
        )
    _registry[activity.id] = activity
    return activity


def unregister(activity_id: str) -> None:
    _registry.pop(activity_id, None)


def get(activity_id: str) -> ActivityDef | None:
    return _registry.get(activity_id)


def require(activity_id: str) -> ActivityDef:
    """Like :func:`get`, but raises with a helpful message for unknown ids.

    Catching typos at bind time rather than never firing is the whole point --
    a misspelled activity in a mapping is otherwise silent.
    """
    try:
        return _registry[activity_id]
    except KeyError:
        from difflib import get_close_matches

        hint = get_close_matches(activity_id, _registry, n=3)
        suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise KeyError(f"unknown activity {activity_id!r}.{suffix}") from None


def all_activities() -> tuple[ActivityDef, ...]:
    return tuple(_registry.values())


def ids() -> tuple[str, ...]:
    return tuple(_registry)


def categories() -> tuple[str, ...]:
    return tuple(sorted({a.category for a in _registry.values()}))


def signals_for(activity_ids: Iterable[str]) -> frozenset[Signal]:
    """Union of the inputs needed to recognise the given activities."""
    needed: set[Signal] = set()
    for aid in activity_ids:
        definition = _registry.get(aid)
        if definition is not None:
            needed |= definition.requires
    return frozenset(needed)


def __iter__() -> Iterator[ActivityDef]:  # pragma: no cover - convenience
    return iter(_registry.values())
