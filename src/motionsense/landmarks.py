"""Landmark index constants and skeleton topology.

These indices match the BlazePose (33 point) and MediaPipe Hands (21 point)
layouts, but they are declared here as plain integers rather than imported from
MediaPipe. That keeps the feature and recognizer layers importable -- and
testable -- without the model runtime present, and it means a different pose
backend can be plugged in as long as it produces the same point ordering.

Left/right refer to the *subject's* anatomy, not to sides of the image.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "HAND_CONNECTIONS",
    "Hand",
    "POSE_CONNECTIONS",
    "Pose",
    "TORSO",
]


class Pose:
    """Indices into a ``(33, 4)`` pose landmark array."""

    NOSE: Final = 0
    LEFT_EYE_INNER: Final = 1
    LEFT_EYE: Final = 2
    LEFT_EYE_OUTER: Final = 3
    RIGHT_EYE_INNER: Final = 4
    RIGHT_EYE: Final = 5
    RIGHT_EYE_OUTER: Final = 6
    LEFT_EAR: Final = 7
    RIGHT_EAR: Final = 8
    MOUTH_LEFT: Final = 9
    MOUTH_RIGHT: Final = 10
    LEFT_SHOULDER: Final = 11
    RIGHT_SHOULDER: Final = 12
    LEFT_ELBOW: Final = 13
    RIGHT_ELBOW: Final = 14
    LEFT_WRIST: Final = 15
    RIGHT_WRIST: Final = 16
    LEFT_PINKY: Final = 17
    RIGHT_PINKY: Final = 18
    LEFT_INDEX: Final = 19
    RIGHT_INDEX: Final = 20
    LEFT_THUMB: Final = 21
    RIGHT_THUMB: Final = 22
    LEFT_HIP: Final = 23
    RIGHT_HIP: Final = 24
    LEFT_KNEE: Final = 25
    RIGHT_KNEE: Final = 26
    LEFT_ANKLE: Final = 27
    RIGHT_ANKLE: Final = 28
    LEFT_HEEL: Final = 29
    RIGHT_HEEL: Final = 30
    LEFT_FOOT_INDEX: Final = 31
    RIGHT_FOOT_INDEX: Final = 32

    COUNT: Final = 33


class Hand:
    """Indices into a ``(21, 3)`` hand landmark array."""

    WRIST: Final = 0
    THUMB_CMC: Final = 1
    THUMB_MCP: Final = 2
    THUMB_IP: Final = 3
    THUMB_TIP: Final = 4
    INDEX_MCP: Final = 5
    INDEX_PIP: Final = 6
    INDEX_DIP: Final = 7
    INDEX_TIP: Final = 8
    MIDDLE_MCP: Final = 9
    MIDDLE_PIP: Final = 10
    MIDDLE_DIP: Final = 11
    MIDDLE_TIP: Final = 12
    RING_MCP: Final = 13
    RING_PIP: Final = 14
    RING_DIP: Final = 15
    RING_TIP: Final = 16
    PINKY_MCP: Final = 17
    PINKY_PIP: Final = 18
    PINKY_DIP: Final = 19
    PINKY_TIP: Final = 20

    COUNT: Final = 21

    #: (tip, pip, mcp) per non-thumb finger, used for curl scoring.
    FINGERS: Final = (
        (8, 6, 5),
        (12, 10, 9),
        (16, 14, 13),
        (20, 18, 17),
    )


#: The four points that define the body frame. Chosen because they are the most
#: reliably visible and the most rigid relative to one another.
TORSO: Final = (Pose.LEFT_SHOULDER, Pose.RIGHT_SHOULDER, Pose.LEFT_HIP, Pose.RIGHT_HIP)


POSE_CONNECTIONS: Final = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

HAND_CONNECTIONS: Final = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
