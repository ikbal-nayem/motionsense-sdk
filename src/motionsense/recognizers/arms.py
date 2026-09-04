"""Arm and upper-body poses: hands up/down, T-pose, arms crossed."""

from __future__ import annotations

from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures
from ..landmarks import Pose
from ..mathx import SchmittGate
from .base import (
    Recognizer,
    Sink,
    combine,
    margin,
    margin_at_least,
    min_margin,
    weakest_available,
)

__all__ = ["ArmsRecognizer"]

_IDS = ("both_hands_up", "left_hand_up", "right_hand_up", "hands_down", "t_pose", "arms_crossed")


class ArmsRecognizer(Recognizer):
    """Static arm configurations.

    ``hands_down`` is deliberately the complement of the other arm states rather
    than an independent test, so exactly one of {both up, left up, right up,
    down} holds at any time. Bindings can then map all four without worrying
    about two keys being held at once.
    """

    activities = tuple(d for d in (get(i) for i in _IDS) if d is not None)

    def __init__(self):
        self._left_up: SchmittGate | None = None
        self._right_up: SchmittGate | None = None
        self._t_pose: SchmittGate | None = None
        self._crossed: SchmittGate | None = None
        self._tuning = Tuning()

    def configure(self, tuning: Tuning) -> None:
        self._tuning = tuning
        on, off = tuning.level_min_on, tuning.level_min_off
        self._left_up = SchmittGate(tuning.hand_up_enter, tuning.hand_up_exit, on, off)
        self._right_up = SchmittGate(tuning.hand_up_enter, tuning.hand_up_exit, on, off)
        self._t_pose = SchmittGate(0.0, tuning.pose_margin_exit, on, off)
        self._crossed = SchmittGate(0.0, tuning.pose_margin_exit, on, off)

    def reset(self) -> None:
        for gate in (self._left_up, self._right_up, self._t_pose, self._crossed):
            if gate is not None:
                gate.reset()

    def update(self, f: BodyFeatures, out: Sink) -> None:
        if self._left_up is None:
            self.configure(self._tuning)
        t = f.t

        left_up = self._left_up.update(f.left_hand_rise, t)
        right_up = self._right_up.update(f.right_hand_rise, t)
        left_conf = self._left_up.confidence
        right_conf = self._right_up.confidence

        t_pose_score = self._t_pose_score(f)
        t_pose = self._t_pose.update(t_pose_score, t)

        crossed_score = self._crossed_score(f)
        crossed = self._crossed.update(crossed_score, t)

        both = left_up and right_up
        out.level("both_hands_up", both, min(left_conf, right_conf))
        out.level("left_hand_up", left_up and not both, left_conf, rise=_r(f.left_hand_rise))
        out.level("right_hand_up", right_up and not both, right_conf, rise=_r(f.right_hand_rise))
        # A complement of the other three: without a body none of them is
        # actually known to be false, so this must not default to true either.
        out.level("hands_down", f.has_body and not left_up and not right_up and not t_pose, 1.0)
        # Scores go into the event data so a pose that is not firing can be
        # diagnosed from outside: a negative score says the geometry was
        # measured and rejected, NaN says it could not be measured at all.
        out.level("t_pose", t_pose, self._t_pose.confidence, score=_r(t_pose_score))
        out.level("arms_crossed", crossed, self._crossed.confidence, score=_r(crossed_score))

    # -- scores ----------------------------------------------------------------
    def _t_pose_score(self, f: BodyFeatures) -> float:
        """One margin for "both arms straight, level and out to the sides".

        Straightness comes from the elbow angle rather than from
        ``|wrist - shoulder|`` compared to a fixed length, so it holds for any
        arm length; and the elbow angle is measured in 3D, so an arm angled
        slightly toward the camera still reads as straight.
        """
        tn = self._tuning
        if not f.visible(Pose.LEFT_WRIST, Pose.RIGHT_WRIST, Pose.LEFT_ELBOW, Pose.RIGHT_ELBOW):
            return float("nan")

        lw = f.P[Pose.LEFT_WRIST]
        rw = f.P[Pose.RIGHT_WRIST]
        return min_margin(
            margin(abs(f.left_arm_tilt), tn.t_pose_arm_angle, tn.t_pose_arm_angle),
            margin(abs(f.right_arm_tilt), tn.t_pose_arm_angle, tn.t_pose_arm_angle),
            margin_at_least(f.left_elbow_angle, tn.t_pose_elbow_min, 30.0),
            margin_at_least(f.right_elbow_angle, tn.t_pose_elbow_min, 30.0),
            # Subject's left arm reaches toward -x, right arm toward +x.
            margin(float(lw[0]), -tn.t_pose_spread, 0.30),
            margin_at_least(float(rw[0]), tn.t_pose_spread, 0.30),
        )

    def _crossed_score(self, f: BodyFeatures) -> float:
        """Wrists swapped sides and held at chest height.

        This pose hides the landmarks that identify it: folded arms tuck each
        hand under the opposite arm, so wrist visibility drops and any feature
        derived from a wrist -- the elbow angles included -- becomes
        unmeasurable. Testing it the obvious way therefore fails on exactly the
        people holding the pose properly. Three things follow from that:

        * wrists are accepted at a lower visibility than the global gate;
        * the essential test uses only wrist *positions*, which the detector
          still estimates when confidence is low;
        * elbow flexion is supporting evidence that cannot veto.

        The crossing itself is measured as ``left.x - right.x`` rather than each
        wrist against the body midline. A relative test needs no assumption that
        the subject is centred and does not care whether the fold is
        symmetric -- and it separates far better: at rest it reads about -0.93,
        against +0.21 for even a tight fold.
        """
        tn = self._tuning
        if not all(
            f.vis[i] >= tn.arms_crossed_min_visibility
            for i in (Pose.LEFT_WRIST, Pose.RIGHT_WRIST)
        ):
            return float("nan")

        lw = f.P[Pose.LEFT_WRIST]
        rw = f.P[Pose.RIGHT_WRIST]
        top = float(f.shoulder_center[1]) + 0.15

        essential = min_margin(
            margin_at_least(float(lw[0]) - float(rw[0]), tn.arms_crossed_separation, 0.18),
            margin_at_least(float(lw[1]), 0.15, 0.15),
            margin_at_least(float(rw[1]), 0.15, 0.15),
            margin(float(lw[1]), top, 0.20),
            margin(float(rw[1]), top, 0.20),
        )
        supporting = weakest_available(
            margin(f.left_elbow_angle, tn.arms_crossed_elbow_max, 30.0),
            margin(f.right_elbow_angle, tn.arms_crossed_elbow_max, 30.0),
        )
        return combine(essential, supporting)


def _r(value: float) -> float:
    """Round for event data, leaving NaN intact so 'unmeasurable' stays visible."""
    return value if value != value else round(float(value), 3)
