"""Landmark provider for the legacy ``mediapipe.solutions`` API.

Used automatically on MediaPipe 0.10.x, where it is the zero-setup option: the
models ship inside the wheel, so nothing has to be downloaded. MediaPipe 1.0
removed this API, and installs on that version fall back to
:mod:`motionsense.providers.mediapipe_tasks`.

This is where the per-frame cost lives -- inference dominates everything else by
two orders of magnitude -- so the work here is about not doing more of it than
necessary:

* **Downscale before inference.** The model resizes to its own fixed input size
  internally. Handing it a 1080p frame buys nothing and pays for a larger
  resize, twice (once for pose, once for hands).
* **Mirror coordinates, not pixels.** Flipping a 720p frame touches ~2.7 million
  bytes. Flipping the landmark x column touches 33 floats and produces exactly
  the same geometry.
* **Run the hand model only when something needs it.** It costs about as much as
  the pose model, and most control schemes never use a finger state.
* **Reuse the colour-conversion buffer.** One allocation per resolution instead
  of one per frame.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import EngineConfig
from ..landmarks import Hand, Pose
from ..types import HandSample
from .base import Landmarks, LandmarkProvider

__all__ = ["MediaPipeSolutionsProvider"]

_POSE_FIELDS = 4
_HAND_FIELDS = 3


def _import_mediapipe():
    # MediaPipe logs a wall of C++ warnings on first use. Quiet it unless the
    # host application has already expressed a preference.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mediapipe is required for the default landmark provider. "
            "Install it with `pip install mediapipe`, or pass your own "
            "LandmarkProvider to MotionEngine."
        ) from exc
    return mp


class MediaPipeSolutionsProvider(LandmarkProvider):
    """Pose (and optionally hand) landmarks from MediaPipe Solutions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._mp = None
        self._pose = None
        self._hands = None
        self._hands_enabled = False
        self._rgb: np.ndarray | None = None
        self._resized: np.ndarray | None = None
        self._last_hands: tuple[HandSample, ...] = ()
        self._frames_since_hands = 0

    # -- lifecycle -------------------------------------------------------------
    def _ensure_pose(self):
        if self._pose is not None:
            return self._pose
        mp = self._mp or _import_mediapipe()
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.model_smoothing,
            enable_segmentation=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return self._pose

    def set_hands_enabled(self, enabled: bool) -> None:
        if enabled == self._hands_enabled:
            return
        self._hands_enabled = enabled
        if not enabled:
            if self._hands is not None:
                self._hands.close()
                self._hands = None
            self._last_hands = ()
            return

        mp = self._mp or _import_mediapipe()
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            # The hand model only offers lite (0) and full (1).
            model_complexity=min(self.config.model_complexity, 1),
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._hands_enabled = False
        self._last_hands = ()

    # -- inference -------------------------------------------------------------
    def process(self, image: np.ndarray, t: float, frame_index: int) -> Landmarks:
        import cv2

        rgb = self._prepare(image, cv2)
        pose_model = self._ensure_pose()

        # MediaPipe skips an internal copy for read-only arrays.
        rgb.flags.writeable = False
        pose_results = pose_model.process(rgb)

        hands = self._last_hands
        if self._hands is not None:
            self._frames_since_hands += 1
            if self._frames_since_hands >= self.config.hands_interval:
                self._frames_since_hands = 0
                hand_results = self._hands.process(rgb)
                hands = self._convert_hands(hand_results)
                self._last_hands = hands
        rgb.flags.writeable = True

        pose = self._convert_pose(getattr(pose_results, "pose_landmarks", None), Pose.COUNT, _POSE_FIELDS)
        world = self._convert_pose(
            getattr(pose_results, "pose_world_landmarks", None), Pose.COUNT, _POSE_FIELDS
        )

        if pose is None:
            # Hand landmarks without a body cannot be normalised, and stale ones
            # would keep finger activities latched after the user leaves.
            self._last_hands = ()
            return Landmarks(None, None, ())

        if not self.config.mirrored_input:
            pose[:, 0] = 1.0 - pose[:, 0]
            if world is not None:
                world[:, 0] = -world[:, 0]

        return Landmarks(pose=pose, pose_world=world[:, :3] if world is not None else None, hands=hands)

    # -- helpers ---------------------------------------------------------------
    def _prepare(self, image: np.ndarray, cv2) -> np.ndarray:
        """Downscale if requested and convert BGR to RGB into a reused buffer."""
        target = self.config.inference_width
        if target and image.shape[1] > target:
            height = max(1, int(round(image.shape[0] * target / image.shape[1])))
            size = (target, height)
            if self._resized is None or self._resized.shape[:2] != (height, target):
                self._resized = np.empty((height, target, 3), dtype=image.dtype)
            # INTER_AREA is the correct kernel for downscaling: it averages the
            # source pixels that map into each destination pixel, so it does not
            # alias. INTER_LINEAR would undersample and add landmark jitter.
            cv2.resize(image, size, dst=self._resized, interpolation=cv2.INTER_AREA)
            image = self._resized

        shape = (image.shape[0], image.shape[1], 3)
        if self._rgb is None or self._rgb.shape != shape:
            self._rgb = np.empty(shape, dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self._rgb)
        return self._rgb

    @staticmethod
    def _convert_pose(landmark_list, count: int, fields: int) -> np.ndarray | None:
        """Protobuf landmarks to a flat float32 array in one pass.

        ``np.fromiter`` with an explicit ``count`` sizes the output up front, so
        there is no intermediate Python list and no reallocation.
        """
        if landmark_list is None:
            return None
        landmarks = landmark_list.landmark
        if len(landmarks) < count:
            return None
        flat = np.fromiter(
            (value for lm in landmarks for value in (lm.x, lm.y, lm.z, lm.visibility)),
            dtype=np.float32,
            count=count * fields,
        )
        return flat.reshape(count, fields)

    def _convert_hands(self, results) -> tuple[HandSample, ...]:
        multi = getattr(results, "multi_hand_landmarks", None)
        if not multi:
            return ()
        handedness = getattr(results, "multi_handedness", None) or []

        samples = []
        for i, hand_landmarks in enumerate(multi):
            points = np.fromiter(
                (v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)),
                dtype=np.float32,
                count=Hand.COUNT * _HAND_FIELDS,
            ).reshape(Hand.COUNT, _HAND_FIELDS)

            label, score = "right", 1.0
            if i < len(handedness):
                classification = handedness[i].classification[0]
                label = classification.label.lower()
                score = float(classification.score)

            # MediaPipe assigns handedness on the assumption that the image is
            # already mirrored (selfie view). A raw camera feed is not, so the
            # label refers to the opposite hand and must be swapped. Getting this
            # wrong silently maps every left-hand gesture to the right hand.
            if not self.config.mirrored_input:
                label = "right" if label == "left" else "left"
                points[:, 0] = 1.0 - points[:, 0]

            samples.append(HandSample(side=label, points=points, score=score))
        return tuple(samples)
