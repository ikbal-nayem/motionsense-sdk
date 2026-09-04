"""Whole-body posture: lean, head down, squat, and the sit/stand transitions."""

from __future__ import annotations

import math

from ..calibration import Calibration
from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures
from ..mathx import SchmittGate
from .base import Recognizer, Sink, nanmax, nanmean

__all__ = ["PostureRecognizer"]

_IDS = ("lean_left", "lean_right", "squat", "head_down", "sit_down", "stand_up")


class PostureRecognizer(Recognizer):
    """Posture states plus the edges between them.

    Runs before the motion recognizer so that ``squat`` is already decided when
    jump detection asks about it.
    """

    activities = tuple(d for d in (get(i) for i in _IDS) if d is not None)

    def __init__(self):
        self._lean_left: SchmittGate | None = None
        self._lean_right: SchmittGate | None = None
        self._head_down: SchmittGate | None = None
        self._squat: SchmittGate | None = None
        self._tuning = Tuning()
        self._calibration: Calibration | None = None
        self._was_squatting = False

    def configure(self, tuning: Tuning) -> None:
        self._tuning = tuning
        on, off = tuning.level_min_on, tuning.level_min_off
        self._lean_left = SchmittGate(tuning.lean_enter_deg, tuning.lean_exit_deg, on, off)
        self._lean_right = SchmittGate(tuning.lean_enter_deg, tuning.lean_exit_deg, on, off)
        self._head_down = SchmittGate(tuning.head_down_enter, tuning.head_down_exit, on, off)
        # Thresholds 1.0 / 0.0 because the squat score below is built so that 1.0
        # is exactly the entry condition and 0.0 exactly the exit condition,
        # whichever cue produced it.
        self._squat = SchmittGate(1.0, 0.0, tuning.posture_min_on, tuning.posture_min_off)
        self._calibration = None

    def reset(self) -> None:
        for gate in (self._lean_left, self._lean_right, self._head_down, self._squat):
            if gate is not None:
                gate.reset()
        self._was_squatting = False

    def update(self, f: BodyFeatures, out: Sink) -> None:
        if self._squat is None:
            self.configure(self._tuning)
        self._sync_calibration(f.calibration)
        t = f.t

        lean = f.torso_lean
        right = self._lean_right.update(lean, t)
        # Mirror the signal rather than the gate, so both directions share one
        # threshold pair and cannot disagree about where the dead band is.
        left = self._lean_left.update(lean if math.isnan(lean) else -lean, t)
        out.level("lean_right", right and not left, self._lean_right.confidence)
        out.level("lean_left", left and not right, self._lean_left.confidence)

        head_down = self._head_down.update(f.nose_above_shoulders, t)
        out.level("head_down", head_down, self._head_down.confidence)

        squatting = self._squat.update(self._squat_score(f), t)
        out.level("squat", squatting, self._squat.confidence)

        # Edges come from the gate's own transition, so they inherit its
        # debouncing instead of needing a separate cooldown.
        if squatting != self._was_squatting and self._squat.changed_at == t:
            out.edge("sit_down" if squatting else "stand_up", 1.0)
        self._was_squatting = squatting

    # -- squat scoring ---------------------------------------------------------
    def _squat_score(self, f: BodyFeatures) -> float:
        """Squat depth as a unitless score: >= 1.0 enters, <= 0.0 exits.

        Two independent cues, fused with ``max`` because each one goes quiet
        rather than wrong in the view where it cannot see:

        * **Knee flexion**, measured in 3D from world landmarks. The reliable cue
          whenever the ankles are in frame, and it is view-independent -- it does
          not care whether the subject faces the camera or stands sideways.
        * **Thigh verticality**, the cosine of the thigh's angle from vertical,
          needing only hips and knees. This is the fallback for the very common
          webcam framing that cuts off the feet, where a knee angle cannot be
          computed at all.

        The original implementation used ``(knee_y - hip_y) / (ankle_y - hip_y)``,
        which requires visible ankles and reads the projected image only; both
        cues here degrade more gracefully than that.
        """
        tn = self._tuning

        knee = nanmean(f.left_knee_angle, f.right_knee_angle)
        knee_span = tn.squat_knee_exit_deg - tn.squat_knee_enter_deg
        s_knee = (tn.squat_knee_exit_deg - knee) / knee_span if knee_span > 0 else float("nan")

        thigh = nanmean(f.left_thigh_vertical, f.right_thigh_vertical)
        thigh_span = tn.squat_exit - tn.squat_enter
        s_thigh = (tn.squat_exit - thigh) / thigh_span if thigh_span > 0 else float("nan")

        return nanmax(s_knee, s_thigh)

    def _sync_calibration(self, calibration: Calibration) -> None:
        """Re-derive calibration-dependent thresholds when a new one is captured."""
        if calibration is self._calibration:
            return
        self._calibration = calibration
        tn = self._tuning
        self._head_down.on_threshold = calibration.scaled(
            "nose_above_shoulders", tn.head_down_enter_ratio, tn.head_down_enter
        )
        self._head_down.off_threshold = calibration.scaled(
            "nose_above_shoulders", tn.head_down_exit_ratio, tn.head_down_exit
        )
