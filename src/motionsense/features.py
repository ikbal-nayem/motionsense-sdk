"""Turns raw landmarks into a scale- and position-invariant body description.

Everything downstream reads :class:`BodyFeatures`, never raw landmarks. The
point of this layer is that a threshold written against it means one fixed
physical thing, no matter where the subject stands or what shape the frame is.
Three corrections get it there:

**Aspect correction.** Detector output is normalised to ``[0, 1]`` on each axis
independently, so on a 16:9 frame one x-unit is 1.78 times longer than one
y-unit. Any distance, angle or ratio mixing the two is therefore wrong -- a
shoulder width measured horizontally comes out 78% too large relative to a torso
measured vertically, and it changes again on a 4:3 camera. Multiplying x by the
aspect ratio puts both axes in units of frame height.

**Scale normalisation.** Dividing by a body scale derived from the torso removes
the subject's distance from the camera. A threshold of 0.06 body units means the
same gesture at two feet and at fifteen.

**Origin shift.** Translating to the hip centre removes where in the frame the
subject is standing, so walking across the room does not change any feature.

Together these are a similarity normalisation (translate + uniform scale).
Rotation is deliberately *not* normalised: the image vertical is the gravity
reference that makes "up", "lean" and "head down" meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .calibration import EMPTY as _EMPTY_CALIBRATION, Calibration
from .config import EngineConfig
from .landmarks import Hand, Pose, TORSO
from .mathx import ExpSmoother, OneEuroFilter, angle_at, angle_at3, procrustes_scale, signed_angle_from_up
from .types import HandSample

__all__ = ["BodyFeatures", "FeatureExtractor", "HandFeatures"]

NAN = float("nan")
_DEG = 180.0 / math.pi

#: Body scale is smoothed over this many seconds. A person's size does not
#: change, so heavy smoothing here is free accuracy: it removes scale jitter from
#: every normalised feature at once.
_SCALE_TAU = 0.9
#: Bound on how fast the smoothed scale may follow a single frame. Caps the
#: damage from one badly-fitted torso without blocking genuine approach/retreat.
_SCALE_CLAMP = (0.65, 1.55)
#: Tracking gap after which filters are reset instead of interpolated across.
_TRACKING_GAP = 0.4


@dataclass(slots=True)
class HandFeatures:
    """Finger geometry for one hand, in ratios free of scale, rotation and view.

    The ratios are measured in three dimensions, not on the image plane: a hand
    angled toward or away from the camera foreshortens, and a gap between two
    fingertips that lies mostly along the view direction all but disappears in
    projection while being entirely real.
    """

    side: str
    score: float
    #: Mean over the four non-thumb fingers of ``|tip - wrist| / |pip - wrist|``.
    #: Below ~1.0 the fingertips have curled inside their own knuckles.
    curl: float
    #: Thumb-to-index-tip 3D distance over hand size. Below ~0.35 is a pinch.
    pinch: float
    #: Palm centre in aspect-corrected image coordinates.
    center: np.ndarray
    points: np.ndarray


@dataclass(slots=True)
class BodyFeatures:
    """One frame of normalised body state.

    Unavailable measurements are ``NaN`` rather than a guessed value, and NaN
    propagates through :class:`~motionsense.mathx.gates.SchmittGate` as "no
    evidence". A recognizer therefore degrades to silence when the landmarks it
    needs are out of frame, instead of firing on extrapolated positions.
    """

    t: float
    #: Whether a torso was located this frame. ``False`` for a hand seen without
    #: a body -- every body-derived field below is then NaN, exactly like an
    #: occluded landmark. NaN is not enough on its own for a recognizer whose
    #: output is the logical complement of other pose states (``hands_down`` is
    #: the built-in example): "no evidence" must not collapse to "false" there,
    #: or the complement wrongly reads as true. Check this field instead.
    has_body: bool
    #: Body scale in frame-height units; the divisor that produced ``P``.
    scale: float
    aspect: float
    #: ``(33, 2)`` normalised: origin at hip centre, +x subject's right, +y up.
    P: np.ndarray
    #: ``(33, 3)`` world landmarks in metres (+y up), or ``None``. Use for joint
    #: angles: they are view-independent, which 2D angles are not.
    W: np.ndarray | None
    #: ``(33,)`` per-landmark visibility.
    vis: np.ndarray
    min_visibility: float
    hands: tuple[HandFeatures, ...]
    calibration: Calibration

    # -- derived scalars (NaN when the inputs are not visible) ----------------
    shoulder_center: np.ndarray
    torso_center: np.ndarray
    #: Torso tilt from vertical in degrees; positive leans to the subject's right.
    torso_lean: float
    #: Height of the nose above the shoulder line, in body units.
    nose_above_shoulders: float
    shoulder_center_height: float
    shoulder_width: float
    #: Torso centre height above the bottom of the frame, in body units. This is
    #: the only *absolute* vertical feature -- everything else is hip-relative,
    #: which would cancel out exactly the whole-body motion a jump consists of.
    elevation: float
    left_hand_rise: float
    right_hand_rise: float
    left_elbow_angle: float
    right_elbow_angle: float
    #: Shoulder-to-wrist direction, degrees from horizontal (+ is upward).
    left_arm_tilt: float
    right_arm_tilt: float
    arm_length: float
    left_knee_angle: float
    right_knee_angle: float
    left_thigh_vertical: float
    right_thigh_vertical: float
    thigh_length: float

    # -- helpers ---------------------------------------------------------------
    def visible(self, *indices: int) -> bool:
        """True when every named landmark clears the visibility threshold."""
        return all(self.vis[i] >= self.min_visibility for i in indices)

    def point(self, index: int) -> np.ndarray:
        """Normalised position of one landmark."""
        return self.P[index]

    def hand(self, side: str) -> HandFeatures | None:
        for h in self.hands:
            if h.side == side:
                return h
        return None


class FeatureExtractor:
    """Stateful per-stream converter from landmarks to :class:`BodyFeatures`."""

    __slots__ = ("config", "_filter", "_world_filter", "_scale", "_last_t", "_calibration")

    def __init__(self, config: EngineConfig):
        self.config = config
        self._filter = OneEuroFilter(config.filter_min_cutoff, config.filter_beta)
        # World landmarks carry more depth noise than the image ones, so they get
        # a lower cutoff. They only feed joint angles, which change slowly.
        self._world_filter = OneEuroFilter(max(config.filter_min_cutoff * 0.7, 0.4), config.filter_beta * 0.6)
        self._scale = ExpSmoother(_SCALE_TAU)
        self._last_t: float | None = None
        self._calibration: Calibration = _EMPTY_CALIBRATION

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @calibration.setter
    def calibration(self, value: Calibration) -> None:
        self._calibration = value or _EMPTY_CALIBRATION

    def reset(self) -> None:
        self._filter.reset()
        self._world_filter.reset()
        self._scale.reset()
        self._last_t = None

    def update(
        self,
        pose: np.ndarray | None,
        pose_world: np.ndarray | None,
        hands: tuple[HandSample, ...],
        t: float,
        aspect: float,
    ) -> BodyFeatures | None:
        """Build features for one frame, or ``None`` if nothing can be reported."""
        hand_features = tuple(
            f for f in (self._hand_features(h, aspect) for h in hands) if f is not None
        )

        if pose is None:
            # A long gap means the next detection is a fresh acquisition; carrying
            # filter state across it would drag the new pose toward the old one.
            if self._last_t is not None and t - self._last_t > _TRACKING_GAP:
                self.reset()
            return self._hands_only(hand_features, t, aspect) if hand_features else None

        if self._last_t is not None and t - self._last_t > _TRACKING_GAP:
            self.reset()
        self._last_t = t

        vis = pose[:, 3]
        if not all(vis[i] >= self.config.tuning.min_visibility for i in TORSO):
            # Without a torso there is no body frame to normalise against -- but
            # finger geometry needs no body frame, so a visible hand still reports.
            return self._hands_only(hand_features, t, aspect) if hand_features else None

        # Aspect-correct, then filter. Filtering after correction keeps the
        # 1-Euro speed term isotropic -- otherwise horizontal motion would look
        # 1.78x faster than vertical and get a different cutoff.
        xy = pose[:, :2].astype(np.float64, copy=True)
        xy[:, 0] *= aspect
        if self.config.filter_landmarks:
            xy = self._filter(xy, t)

        torso_pts = xy[list(TORSO)]
        raw_scale = 2.0 * procrustes_scale(torso_pts)
        if raw_scale < 1e-5:
            return None
        previous = self._scale.value
        if previous is not None:
            lo, hi = _SCALE_CLAMP
            raw_scale = min(max(raw_scale, previous * lo), previous * hi)
        scale = self._scale.update(raw_scale, t)
        if scale < 1e-5:
            return None

        hip_center = 0.5 * (xy[Pose.LEFT_HIP] + xy[Pose.RIGHT_HIP])
        # Translate to the hip centre and flip y so +y is up. One vectorised pass
        # over all 33 points; every feature below is then plain scalar arithmetic.
        P = (xy - hip_center) / scale
        P[:, 1] *= -1.0

        W = None
        if pose_world is not None:
            w = pose_world.astype(np.float64, copy=True)
            if self.config.filter_landmarks:
                w = self._world_filter(w, t)
            w[:, 1] *= -1.0
            W = w

        return self._assemble(P, W, vis, xy, hip_center, scale, aspect, hand_features, t)

    def _hands_only(self, hand_features: tuple[HandFeatures, ...], t: float, aspect: float) -> BodyFeatures:
        """Features for a frame where a hand is seen but no body frame can be built.

        Most commonly a close desk framing that cuts the torso out of the shot.
        Finger geometry is self-normalised (wrist-to-knuckle span), so it needs
        no body frame. Every body-derived field is set the same way an occluded
        landmark already is elsewhere: NaN, which the recognizers already treat
        as "no evidence" rather than "false" -- so body-based poses silently
        stay quiet instead of tripping on placeholder geometry, while hand
        recognizers, the only ones that read ``hands``, work exactly as usual.
        """
        empty = np.full((Pose.COUNT, 2), NAN)
        return BodyFeatures(
            t=t,
            has_body=False,
            scale=NAN,
            aspect=aspect,
            P=empty,
            W=None,
            vis=np.zeros(Pose.COUNT),
            min_visibility=self.config.tuning.min_visibility,
            hands=hand_features,
            calibration=self._calibration,
            shoulder_center=np.array([NAN, NAN]),
            torso_center=np.array([NAN, NAN]),
            torso_lean=NAN,
            nose_above_shoulders=NAN,
            shoulder_center_height=NAN,
            shoulder_width=NAN,
            elevation=NAN,
            left_hand_rise=NAN,
            right_hand_rise=NAN,
            left_elbow_angle=NAN,
            right_elbow_angle=NAN,
            left_arm_tilt=NAN,
            right_arm_tilt=NAN,
            arm_length=NAN,
            left_knee_angle=NAN,
            right_knee_angle=NAN,
            left_thigh_vertical=NAN,
            right_thigh_vertical=NAN,
            thigh_length=NAN,
        )

    # -- feature assembly ------------------------------------------------------
    def _assemble(self, P, W, vis, xy, hip_center, scale, aspect, hand_features, t) -> BodyFeatures:
        cfg = self.config.tuning
        min_vis = cfg.min_visibility

        def seen(*idx: int) -> bool:
            return all(vis[i] >= min_vis for i in idx)

        l_sh, r_sh = P[Pose.LEFT_SHOULDER], P[Pose.RIGHT_SHOULDER]
        shoulder_center = 0.5 * (l_sh + r_sh)
        shoulder_width = float(np.hypot(*(l_sh - r_sh)))
        torso_center = 0.25 * (P[Pose.LEFT_SHOULDER] + P[Pose.RIGHT_SHOULDER] + P[Pose.LEFT_HIP] + P[Pose.RIGHT_HIP])

        # Lean: the torso vector's tilt from vertical. Superior to the common
        # "horizontal shoulder/hip offset over shoulder width" because it stays
        # correct when the subject turns (which shrinks shoulder width and would
        # otherwise inflate the ratio).
        torso_lean = signed_angle_from_up(shoulder_center) * _DEG

        nose_above = NAN
        if seen(Pose.NOSE):
            nose_above = float(P[Pose.NOSE][1] - shoulder_center[1])

        # Absolute height in the frame, needed for jump. xy[:,1] is still the raw
        # normalised y, so (1 - y) is the distance up from the frame's bottom edge.
        torso_y_img = 0.25 * (
            xy[Pose.LEFT_SHOULDER][1] + xy[Pose.RIGHT_SHOULDER][1] + xy[Pose.LEFT_HIP][1] + xy[Pose.RIGHT_HIP][1]
        )
        elevation = float((1.0 - torso_y_img) / scale)

        left_rise = right_rise = NAN
        if seen(Pose.NOSE, Pose.LEFT_WRIST):
            left_rise = float(P[Pose.LEFT_WRIST][1] - P[Pose.NOSE][1])
        if seen(Pose.NOSE, Pose.RIGHT_WRIST):
            right_rise = float(P[Pose.RIGHT_WRIST][1] - P[Pose.NOSE][1])

        l_elbow = self._joint_angle(P, W, vis, min_vis, Pose.LEFT_SHOULDER, Pose.LEFT_ELBOW, Pose.LEFT_WRIST)
        r_elbow = self._joint_angle(P, W, vis, min_vis, Pose.RIGHT_SHOULDER, Pose.RIGHT_ELBOW, Pose.RIGHT_WRIST)
        l_knee = self._joint_angle(P, W, vis, min_vis, Pose.LEFT_HIP, Pose.LEFT_KNEE, Pose.LEFT_ANKLE)
        r_knee = self._joint_angle(P, W, vis, min_vis, Pose.RIGHT_HIP, Pose.RIGHT_KNEE, Pose.RIGHT_ANKLE)

        l_tilt = self._arm_tilt(P, vis, min_vis, Pose.LEFT_SHOULDER, Pose.LEFT_WRIST)
        r_tilt = self._arm_tilt(P, vis, min_vis, Pose.RIGHT_SHOULDER, Pose.RIGHT_WRIST)

        arm_length = self._segment_sum(
            P, vis, min_vis,
            (Pose.LEFT_SHOULDER, Pose.LEFT_ELBOW, Pose.LEFT_WRIST),
            (Pose.RIGHT_SHOULDER, Pose.RIGHT_ELBOW, Pose.RIGHT_WRIST),
        )

        l_thigh_v, l_thigh_len = self._thigh(P, vis, min_vis, Pose.LEFT_HIP, Pose.LEFT_KNEE)
        r_thigh_v, r_thigh_len = self._thigh(P, vis, min_vis, Pose.RIGHT_HIP, Pose.RIGHT_KNEE)
        thigh_length = _nanmean2(l_thigh_len, r_thigh_len)

        return BodyFeatures(
            t=t,
            has_body=True,
            scale=scale,
            aspect=aspect,
            P=P,
            W=W,
            vis=vis,
            min_visibility=min_vis,
            hands=hand_features,
            calibration=self._calibration,
            shoulder_center=shoulder_center,
            torso_center=torso_center,
            torso_lean=torso_lean,
            nose_above_shoulders=nose_above,
            shoulder_center_height=float(shoulder_center[1]),
            shoulder_width=shoulder_width,
            elevation=elevation,
            left_hand_rise=left_rise,
            right_hand_rise=right_rise,
            left_elbow_angle=l_elbow,
            right_elbow_angle=r_elbow,
            left_arm_tilt=l_tilt,
            right_arm_tilt=r_tilt,
            arm_length=arm_length,
            left_knee_angle=l_knee,
            right_knee_angle=r_knee,
            left_thigh_vertical=l_thigh_v,
            right_thigh_vertical=r_thigh_v,
            thigh_length=thigh_length,
        )

    @staticmethod
    def _joint_angle(P, W, vis, min_vis, a: int, b: int, c: int) -> float:
        """Joint angle in degrees, preferring the view-independent 3D estimate."""
        if not all(vis[i] >= min_vis for i in (a, b, c)):
            return NAN
        if W is not None:
            return angle_at3(W[a], W[b], W[c]) * _DEG
        return angle_at(P[a], P[b], P[c]) * _DEG

    @staticmethod
    def _arm_tilt(P, vis, min_vis, shoulder: int, wrist: int) -> float:
        if not all(vis[i] >= min_vis for i in (shoulder, wrist)):
            return NAN
        v = P[wrist] - P[shoulder]
        return math.degrees(math.atan2(float(v[1]), abs(float(v[0])) + 1e-9))

    @staticmethod
    def _segment_sum(P, vis, min_vis, left: tuple[int, ...], right: tuple[int, ...]) -> float:
        """Mean limb length over whichever side(s) are visible.

        Self-measuring the limb makes 'arm straight' and 'arm extended' tests
        independent of body proportion: the reference is this subject's own arm,
        taken from the current frame, not a population average.
        """
        totals = []
        for chain in (left, right):
            if not all(vis[i] >= min_vis for i in chain):
                continue
            total = 0.0
            for a, b in zip(chain, chain[1:]):
                total += float(np.hypot(*(P[b] - P[a])))
            totals.append(total)
        return float(sum(totals) / len(totals)) if totals else NAN

    @staticmethod
    def _thigh(P, vis, min_vis, hip: int, knee: int) -> tuple[float, float]:
        """(cosine of thigh angle from vertical, thigh length in body units)."""
        if not all(vis[i] >= min_vis for i in (hip, knee)):
            return NAN, NAN
        v = P[hip] - P[knee]
        length = float(np.hypot(v[0], v[1]))
        if length < 1e-6:
            return NAN, NAN
        return float(v[1] / length), length

    def _hand_features(self, sample: HandSample, aspect: float) -> HandFeatures | None:
        if sample.score < self.config.tuning.hand_min_score:
            return None
        # Ratios are measured in 3D; only the display copy below is flattened.
        # A thumb held in front of or behind the index is genuinely far from it,
        # but projects to almost no gap at all -- measuring the pinch on the
        # image plane alone reports that as a pinch. Depth costs nothing here:
        # the detector already returns z for every landmark, on roughly the same
        # scale as x. Aspect correction applies to x only; z is not an image
        # axis and must not be stretched with the frame.
        space = sample.points.astype(np.float64, copy=True)
        space[:, 0] *= aspect
        pts = space[:, :2]

        wrist = space[Hand.WRIST]
        # Hand size from the wrist-to-middle-knuckle span: rigid, always visible,
        # and unaffected by whether the fingers are curled.
        hand_scale = float(np.linalg.norm(space[Hand.MIDDLE_MCP] - wrist))
        if hand_scale < 1e-6:
            return None

        # Curl as a radial ratio about the wrist. The obvious test -- fingertip
        # lower on screen than its own knuckle -- only works for an upright hand
        # and inverts entirely when the hand points downward. Distance from the
        # wrist does not care which way the hand is rotated.
        ratios = []
        for tip, pip, _mcp in Hand.FINGERS:
            d_pip = float(np.linalg.norm(space[pip] - wrist))
            if d_pip < 1e-6:
                continue
            ratios.append(float(np.linalg.norm(space[tip] - wrist)) / d_pip)
        curl = float(sum(ratios) / len(ratios)) if ratios else NAN

        pinch = float(np.linalg.norm(space[Hand.THUMB_TIP] - space[Hand.INDEX_TIP])) / hand_scale

        return HandFeatures(
            side=sample.side,
            score=sample.score,
            curl=curl,
            pinch=pinch,
            center=pts.mean(axis=0),
            points=pts,
        )


def _nanmean2(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return 0.5 * (a + b)
