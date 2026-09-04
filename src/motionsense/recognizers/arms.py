"""Arm and upper-body poses: hands up/down, T-pose, arms crossed."""

from __future__ import annotations

from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures
from ..landmarks import Pose
from ..mathx import SchmittGate
from .base import Recognizer, Sink, margin, margin_at_least, min_margin

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
        out.level("left_hand_up", left_up and not both, left_conf)
        out.level("right_hand_up", right_up and not both, right_conf)
        out.level("hands_down", not left_up and not right_up and not t_pose, 1.0)
        out.level("t_pose", t_pose, self._t_pose.confidence)
        out.level("arms_crossed", crossed, self._crossed.confidence)

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
        """Both wrists past the midline, held at chest height, elbows bent."""
        tn = self._tuning
        if not f.visible(Pose.LEFT_WRIST, Pose.RIGHT_WRIST):
            return float("nan")

        lw = f.P[Pose.LEFT_WRIST]
        rw = f.P[Pose.RIGHT_WRIST]
        top = float(f.shoulder_center[1]) + 0.12
        return min_margin(
            margin_at_least(float(lw[0]), tn.arms_crossed_midline, 0.15),
            margin(float(rw[0]), -tn.arms_crossed_midline, 0.15),
            margin_at_least(float(lw[1]), 0.15, 0.15),
            margin_at_least(float(rw[1]), 0.15, 0.15),
            margin(float(lw[1]), top, 0.20),
            margin(float(rw[1]), top, 0.20),
            margin(f.left_elbow_angle, tn.arms_crossed_elbow_max, 30.0),
            margin(f.right_elbow_angle, tn.arms_crossed_elbow_max, 30.0),
        )
