"""A parametric 3D stick figure, projected to landmarks.

Tests drive the real pipeline -- features, recognizers, dispatcher -- from an
articulated body model instead of a camera, so behaviour is exactly
reproducible and every parameter (distance, frame rate, joint angle) can be
varied independently.

The model is genuinely 3D and orthographically projected, which matters: a
frontal squat foreshortens the thigh rather than rotating it in the image, so a
2D-only test would not distinguish a correct squat detector from a broken one.
"""

from __future__ import annotations

import math

import numpy as np

from motionsense.landmarks import Hand, Pose
from motionsense.providers.base import Landmarks, LandmarkProvider
from motionsense.types import HandSample

# Segment lengths in metres, roughly adult proportions.
TORSO = 0.50
SHOULDER_HALF = 0.17
HIP_HALF = 0.09
NOSE_ABOVE_SHOULDER = 0.18
UPPER_ARM = 0.28
FOREARM = 0.26
THIGH = 0.45
SHANK = 0.45

#: Normalised frame-height units per metre. Puts a standing figure at about 85%
#: of the frame.
PX = 0.538


def body(
    *,
    squat: float = 0.0,
    left_arm: tuple[float, float] = (-80.0, 0.0),
    right_arm: tuple[float, float] = (-80.0, 0.0),
    lean: float = 0.0,
    head_pitch: float = 0.0,
) -> dict[int, np.ndarray]:
    """3D joint positions in metres, hip-centred, y pointing down.

    ``left_arm`` / ``right_arm`` are ``(elevation, forward)`` in degrees:
    elevation is measured from the +x axis in the frontal plane, so -90 hangs
    straight down and 0 is straight out sideways. ``squat`` runs 0 (standing) to
    1 (deep). ``lean`` tilts the torso in degrees, positive to the subject's
    right.
    """
    joints: dict[int, np.ndarray] = {}

    joints[Pose.LEFT_HIP] = np.array([-HIP_HALF, 0.0, 0.0])
    joints[Pose.RIGHT_HIP] = np.array([HIP_HALF, 0.0, 0.0])

    tilt = math.radians(lean)
    up = np.array([math.sin(tilt), -math.cos(tilt), 0.0])
    shoulder_mid = up * TORSO
    side = np.array([math.cos(tilt), math.sin(tilt), 0.0])
    joints[Pose.LEFT_SHOULDER] = shoulder_mid - side * SHOULDER_HALF
    joints[Pose.RIGHT_SHOULDER] = shoulder_mid + side * SHOULDER_HALF

    head = shoulder_mid + up * NOSE_ABOVE_SHOULDER
    nose = head + np.array([0.0, NOSE_ABOVE_SHOULDER * head_pitch, 0.05])
    joints[Pose.NOSE] = nose
    for index, offset in (
        (Pose.LEFT_EYE_INNER, -0.02), (Pose.LEFT_EYE, -0.03), (Pose.LEFT_EYE_OUTER, -0.04),
        (Pose.RIGHT_EYE_INNER, 0.02), (Pose.RIGHT_EYE, 0.03), (Pose.RIGHT_EYE_OUTER, 0.04),
    ):
        joints[index] = nose + np.array([offset, -0.02, 0.0])
    joints[Pose.LEFT_EAR] = nose + np.array([-0.07, -0.01, -0.04])
    joints[Pose.RIGHT_EAR] = nose + np.array([0.07, -0.01, -0.04])
    joints[Pose.MOUTH_LEFT] = nose + np.array([-0.03, 0.04, 0.0])
    joints[Pose.MOUTH_RIGHT] = nose + np.array([0.03, 0.04, 0.0])

    _arm(joints, joints[Pose.LEFT_SHOULDER], left_arm, -1,
         Pose.LEFT_ELBOW, Pose.LEFT_WRIST, Pose.LEFT_PINKY, Pose.LEFT_INDEX, Pose.LEFT_THUMB)
    _arm(joints, joints[Pose.RIGHT_SHOULDER], right_arm, 1,
         Pose.RIGHT_ELBOW, Pose.RIGHT_WRIST, Pose.RIGHT_PINKY, Pose.RIGHT_INDEX, Pose.RIGHT_THUMB)

    # Sagittal-plane leg bend: the thigh swings forward (+z) and the shank back,
    # which is what a squat does to a camera looking at the subject head on.
    thigh_angle = math.radians(80.0 * squat)
    shank_angle = math.radians(40.0 * squat)
    knee_offset = np.array([0.0, THIGH * math.cos(thigh_angle), THIGH * math.sin(thigh_angle)])
    ankle_offset = knee_offset + np.array([0.0, SHANK * math.cos(shank_angle), -SHANK * math.sin(shank_angle)])

    for hip, knee, ankle, heel, toe, sign in (
        (Pose.LEFT_HIP, Pose.LEFT_KNEE, Pose.LEFT_ANKLE, Pose.LEFT_HEEL, Pose.LEFT_FOOT_INDEX, -1),
        (Pose.RIGHT_HIP, Pose.RIGHT_KNEE, Pose.RIGHT_ANKLE, Pose.RIGHT_HEEL, Pose.RIGHT_FOOT_INDEX, 1),
    ):
        base = joints[hip]
        joints[knee] = base + knee_offset + np.array([sign * 0.01, 0.0, 0.0])
        joints[ankle] = base + ankle_offset
        joints[heel] = joints[ankle] + np.array([0.0, 0.02, -0.05])
        joints[toe] = joints[ankle] + np.array([0.0, 0.02, 0.12])

    return joints


def _arm(joints, shoulder, angles, sign, elbow_i, wrist_i, pinky_i, index_i, thumb_i):
    elevation, forward = angles
    a = math.radians(elevation)
    f = math.radians(forward)
    direction = np.array([sign * math.cos(a) * math.cos(f), -math.sin(a), math.sin(f)])
    direction /= np.linalg.norm(direction)
    elbow = shoulder + direction * UPPER_ARM
    wrist = elbow + direction * FOREARM
    joints[elbow_i] = elbow
    joints[wrist_i] = wrist
    joints[pinky_i] = wrist + direction * 0.07 + np.array([sign * 0.02, 0.0, 0.0])
    joints[index_i] = wrist + direction * 0.08
    joints[thumb_i] = wrist + direction * 0.04 + np.array([-sign * 0.02, 0.0, 0.0])


def project(
    joints: dict[int, np.ndarray],
    *,
    aspect: float = 4 / 3,
    ground: float = 0.97,
    lift: float = 0.0,
    distance: float = 1.0,
    shift_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic projection to (image landmarks, world landmarks).

    ``distance`` scales the figure -- 2.0 is twice as far away and therefore half
    the size -- and ``shift_x`` slides it across the frame. Feature normalisation
    is supposed to make both invisible to the recognizers, which is exactly what
    the invariance tests check.
    """
    px = PX / distance
    ankle_y = max(joints[Pose.LEFT_ANKLE][1], joints[Pose.RIGHT_ANKLE][1])
    hip_img_y = ground - ankle_y * px - lift * px

    image = np.zeros((Pose.COUNT, 4), dtype=np.float32)
    world = np.zeros((Pose.COUNT, 3), dtype=np.float32)
    for index in range(Pose.COUNT):
        point = joints[index]
        image[index, 0] = 0.5 + shift_x + point[0] * px / aspect
        image[index, 1] = hip_img_y + point[1] * px
        image[index, 2] = point[2] * px
        image[index, 3] = 1.0
        world[index] = point
    return image, world


def hand(
    side: str, *, curl: float = 0.0, pinch: float = 1.0, center=(0.5, 0.5), thumb: str | None = None
) -> HandSample:
    """A 21-point hand. ``curl`` 0 is flat open, 1 is a closed fist; ``pinch`` 1
    is wide apart, 0 is thumb touching index.

    ``thumb`` overrides the thumb, which otherwise trails the index tip at a
    distance set by ``pinch``. That default cannot pose a thumb independently of
    the fingers, so it can express neither of the two gestures that are *about*
    the thumb:

    ``"folded"``
        across the curled fingers, where a real fist puts it. The tip lands
        beside the index tip, so this is the hand that makes a bare
        thumb-to-index gap indistinguishable from a pinch.
    ``"up"``
        extended and vertical, clear of the fingers -- a thumbs-up.
    """
    points = np.zeros((Hand.COUNT, 3), dtype=np.float32)
    scale = 0.10
    cx, cy = center
    points[Hand.WRIST] = (cx, cy + scale, 0.0)

    knuckles = {Hand.INDEX_MCP: -0.3, Hand.MIDDLE_MCP: -0.1, Hand.RING_MCP: 0.1, Hand.PINKY_MCP: 0.3}
    for mcp, offset in knuckles.items():
        points[mcp] = (cx + offset * scale, cy + 0.35 * scale, 0.0)

    # Curling folds each finger in the plane containing the finger and the palm
    # normal: the tip swings toward the camera and then back down into the palm.
    # A palm-facing camera therefore sees the tip come *inside* its own knuckle,
    # which is the geometry the curl ratio is built to measure.
    proximal, distal = 0.45 * scale, 0.55 * scale
    for tip, pip, mcp in Hand.FINGERS:
        base = points[mcp]
        a1 = math.radians(70.0 * curl)
        a2 = a1 + math.radians(160.0 * curl)
        points[pip] = (
            base[0],
            base[1] - proximal * math.cos(a1),
            base[2] + proximal * math.sin(a1),
        )
        points[tip] = (
            points[pip][0],
            points[pip][1] - distal * math.cos(a2),
            points[pip][2] + distal * math.sin(a2),
        )

    points[Hand.THUMB_MCP] = (cx - 0.35 * scale, cy + 0.7 * scale, 0.0)
    points[Hand.THUMB_CMC] = (cx - 0.30 * scale, cy + 0.9 * scale, 0.0)

    if thumb == "up":
        # Straight up out of the fist: past its own joint, and well clear of the
        # fingers so it cannot read as reaching for the index.
        points[Hand.THUMB_IP] = (cx - 0.38 * scale, cy + 0.20 * scale, 0.0)
        points[Hand.THUMB_TIP] = (cx - 0.40 * scale, cy - 0.30 * scale, 0.0)
    elif thumb == "folded":
        # Bent over the front of the curled fingers, tip resting beside the
        # index tip and no further from the wrist than its own joint.
        points[Hand.THUMB_IP] = (cx - 0.30 * scale, cy + 0.48 * scale, 0.04 * scale)
        points[Hand.THUMB_TIP] = (cx - 0.20 * scale, cy + 0.50 * scale, 0.08 * scale)
    else:
        index_tip = points[Hand.INDEX_TIP]
        thumb_tip = (index_tip[0] - 0.55 * scale * pinch, index_tip[1] + 0.25 * scale * pinch, 0.0)
        points[Hand.THUMB_TIP] = thumb_tip
        points[Hand.THUMB_IP] = (thumb_tip[0] - 0.1 * scale, thumb_tip[1] + 0.15 * scale, 0.0)

    return HandSample(side=side, points=points, score=1.0)


class ScriptedProvider(LandmarkProvider):
    """Replays a prepared list of :class:`Landmarks`, one per frame."""

    def __init__(self, frames: list[Landmarks]):
        self.frames = frames
        self.calls = 0

    def process(self, image, t, frame_index):
        if self.calls >= len(self.frames):
            return Landmarks(None, None, ())
        result = self.frames[self.calls]
        self.calls += 1
        return result

    def close(self):
        pass


def frames_from(poses, hands_per_frame=None) -> list[Landmarks]:
    """Bundle (image, world) pairs into provider output."""
    out = []
    for i, (image, world) in enumerate(poses):
        hands = ()
        if hands_per_frame is not None:
            hands = hands_per_frame[i]
        out.append(Landmarks(pose=image, pose_world=world, hands=hands))
    return out
