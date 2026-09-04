"""Optional preview rendering.

Purely a debugging and UI convenience -- it draws from the numpy landmark arrays
the engine already produced, so it needs OpenCV but not MediaPipe, and it costs
nothing when unused.
"""

from __future__ import annotations

import numpy as np

from .landmarks import HAND_CONNECTIONS, POSE_CONNECTIONS
from .types import FrameResult, Snapshot

__all__ = ["draw_hands", "draw_hud", "draw_pose", "render"]

_SKELETON = (120, 220, 120)
_JOINT = (250, 250, 250)
_HAND = (90, 200, 250)
_TEXT = (245, 245, 245)
_SHADOW = (20, 20, 20)


def render(
    result: FrameResult,
    *,
    mirror: bool = True,
    hud: bool = True,
    snapshot: Snapshot | None = None,
    labels: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Build a display image from a frame result.

    ``mirror=True`` produces the selfie view users expect from a gesture
    interface: the subject's right hand appears on the right of the picture.
    Because the engine already works in that canonical orientation, mirroring
    the image for display makes the landmarks line up with no extra maths --
    it is the *un*-mirrored view that needs the coordinates flipped back.

    Requires ``EngineConfig.deliver_frames=True``; returns ``None`` otherwise.
    """
    if result.image is None:
        return None
    import cv2

    canvas = cv2.flip(result.image, 1) if mirror else result.image.copy()
    draw_pose(canvas, result.pose, unmirror=not mirror)
    draw_hands(canvas, result.hands, unmirror=not mirror)
    if hud:
        names = [(labels or {}).get(a, a) for a in sorted(result.active)]
        draw_hud(canvas, names, snapshot)
    return canvas


def draw_pose(image: np.ndarray, pose: np.ndarray | None, *, unmirror: bool = False, min_visibility: float = 0.5) -> None:
    """Draw the skeleton onto a BGR image in place."""
    if pose is None:
        return
    import cv2

    height, width = image.shape[:2]
    xs = pose[:, 0]
    if unmirror:
        xs = 1.0 - xs
    points = np.stack((xs * width, pose[:, 1] * height), axis=1).astype(np.int32)
    visible = pose[:, 3] >= min_visibility

    for a, b in POSE_CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(image, tuple(points[a]), tuple(points[b]), _SKELETON, 2, cv2.LINE_AA)
    for i, point in enumerate(points):
        if visible[i]:
            cv2.circle(image, tuple(point), 3, _JOINT, -1, cv2.LINE_AA)


def draw_hands(image: np.ndarray, hands, *, unmirror: bool = False) -> None:
    """Draw hand skeletons onto a BGR image in place."""
    if not hands:
        return
    import cv2

    height, width = image.shape[:2]
    for hand in hands:
        xs = hand.points[:, 0]
        if unmirror:
            xs = 1.0 - xs
        points = np.stack((xs * width, hand.points[:, 1] * height), axis=1).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, tuple(points[a]), tuple(points[b]), _HAND, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, tuple(point), 2, _JOINT, -1, cv2.LINE_AA)


def draw_hud(image: np.ndarray, active: list[str], snapshot: Snapshot | None = None) -> None:
    """Overlay the active activity list and, if given, engine stats."""
    import cv2

    lines = []
    if snapshot is not None:
        lines.append(f"{snapshot.fps:5.1f} fps   {snapshot.latency * 1000:5.1f} ms")
        if not snapshot.tracking:
            lines.append("no body detected")
    lines.extend(active or (["-"] if snapshot is not None else []))

    y = 26
    for line in lines:
        # Drawn twice with an offset: a cheap outline that stays readable over
        # both a bright and a dark background.
        cv2.putText(image, line, (13, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.58, _SHADOW, 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, _TEXT, 1, cv2.LINE_AA)
        y += 24
