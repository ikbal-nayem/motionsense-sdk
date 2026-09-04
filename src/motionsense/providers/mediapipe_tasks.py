"""MediaPipe Tasks landmark provider.

The Tasks API is what ``mediapipe.solutions`` became; from MediaPipe 1.0 it is
the only one. Same models underneath, so detection quality matches the legacy
path, with two differences that matter here: model bundles are external files
(see :mod:`motionsense.providers.models`), and inference runs in an explicit
VIDEO mode that is given a timestamp, which lets the tracker reason about time
instead of assuming a fixed frame interval.

The per-frame optimisations are the same as the legacy provider's: downscale
before inference, mirror coordinates rather than pixels, run the hand model only
when something needs it, and reuse the colour-conversion buffer.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import EngineConfig
from ..landmarks import Hand, Pose
from ..types import HandSample
from .base import Landmarks, LandmarkProvider
from .models import POSE_BY_COMPLEXITY, resolve_model

__all__ = ["MediaPipeTasksProvider"]


def _import_mediapipe():
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    return mp, mp_python, vision


class MediaPipeTasksProvider(LandmarkProvider):
    """Pose (and optionally hand) landmarks via the MediaPipe Tasks API."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._mp = None
        self._vision = None
        self._python = None
        self._pose = None
        self._hands = None
        self._hands_enabled = False
        self._rgb: np.ndarray | None = None
        self._resized: np.ndarray | None = None
        self._last_hands: tuple[HandSample, ...] = ()
        self._frames_since_hands = 0
        self._last_stamp_ms = -1

    # -- lifecycle -------------------------------------------------------------
    def _ensure_imports(self):
        if self._mp is None:
            self._mp, self._python, self._vision = _import_mediapipe()
        return self._mp, self._python, self._vision

    def _ensure_pose(self):
        if self._pose is not None:
            return self._pose
        _, mp_python, vision = self._ensure_imports()
        name = POSE_BY_COMPLEXITY.get(self.config.model_complexity, "pose_full")
        path = resolve_model(
            name,
            search_dir=self.config.model_dir,
            allow_download=self.config.allow_model_download,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)
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

        _, mp_python, vision = self._ensure_imports()
        path = resolve_model(
            "hand",
            search_dir=self.config.model_dir,
            allow_download=self.config.allow_model_download,
        )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._hands_enabled = False
        self._last_hands = ()
        self._last_stamp_ms = -1

    # -- inference -------------------------------------------------------------
    def process(self, image: np.ndarray, t: float, frame_index: int) -> Landmarks:
        import cv2

        mp, _, _ = self._ensure_imports()
        rgb = self._prepare(image, cv2)
        pose_model = self._ensure_pose()

        # VIDEO mode requires strictly increasing integer milliseconds. Two frames
        # can land in the same millisecond above 1000 FPS, and a file source
        # replayed faster than real time can repeat a timestamp, so nudge rather
        # than let the graph reject the frame.
        stamp = int(t * 1000.0)
        if stamp <= self._last_stamp_ms:
            stamp = self._last_stamp_ms + 1
        self._last_stamp_ms = stamp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = pose_model.detect_for_video(mp_image, stamp)

        hands = self._last_hands
        if self._hands is not None:
            self._frames_since_hands += 1
            if self._frames_since_hands >= self.config.hands_interval:
                self._frames_since_hands = 0
                hands = self._convert_hands(self._hands.detect_for_video(mp_image, stamp))
                self._last_hands = hands

        pose = self._first(getattr(pose_result, "pose_landmarks", None), Pose.COUNT, world=False)
        if pose is None:
            self._last_hands = ()
            return Landmarks(None, None, ())

        world = self._first(getattr(pose_result, "pose_world_landmarks", None), Pose.COUNT, world=True)

        if not self.config.mirrored_input:
            pose[:, 0] = 1.0 - pose[:, 0]
            if world is not None:
                world[:, 0] = -world[:, 0]

        return Landmarks(pose=pose, pose_world=world, hands=hands)

    # -- helpers ---------------------------------------------------------------
    def _prepare(self, image: np.ndarray, cv2) -> np.ndarray:
        target = self.config.inference_width
        if target and image.shape[1] > target:
            height = max(1, int(round(image.shape[0] * target / image.shape[1])))
            if self._resized is None or self._resized.shape[:2] != (height, target):
                self._resized = np.empty((height, target, 3), dtype=image.dtype)
            cv2.resize(image, (target, height), dst=self._resized, interpolation=cv2.INTER_AREA)
            image = self._resized

        shape = (image.shape[0], image.shape[1], 3)
        if self._rgb is None or self._rgb.shape != shape:
            self._rgb = np.empty(shape, dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self._rgb)
        # mp.Image wraps the buffer without copying, so it must stay contiguous
        # and alive for the duration of the call -- both hold here.
        return self._rgb

    @staticmethod
    def _first(landmark_lists, count: int, *, world: bool) -> np.ndarray | None:
        """First detected body as an array, or None.

        Tasks returns a list of bodies; the engine tracks one subject, so only
        the first is used. World landmarks carry no visibility, so it is taken
        from the image landmarks instead (see :meth:`process`).
        """
        if not landmark_lists:
            return None
        landmarks = landmark_lists[0]
        if len(landmarks) < count:
            return None
        if world:
            flat = np.fromiter(
                (v for lm in landmarks for v in (lm.x, lm.y, lm.z)),
                dtype=np.float32,
                count=count * 3,
            )
            return flat.reshape(count, 3)
        flat = np.fromiter(
            (
                v
                for lm in landmarks
                for v in (lm.x, lm.y, lm.z, _visibility(lm))
            ),
            dtype=np.float32,
            count=count * 4,
        )
        return flat.reshape(count, 4)

    def _convert_hands(self, result) -> tuple[HandSample, ...]:
        landmark_lists = getattr(result, "hand_landmarks", None)
        if not landmark_lists:
            return ()
        handedness = getattr(result, "handedness", None) or []

        samples = []
        for i, landmarks in enumerate(landmark_lists):
            points = np.fromiter(
                (v for lm in landmarks for v in (lm.x, lm.y, lm.z)),
                dtype=np.float32,
                count=Hand.COUNT * 3,
            ).reshape(Hand.COUNT, 3)

            label, score = "right", 1.0
            if i < len(handedness) and handedness[i]:
                category = handedness[i][0]
                label = (category.category_name or "right").lower()
                score = float(category.score)

            # Handedness assumes a mirrored (selfie) image, exactly as in the
            # legacy API. A raw camera feed is not mirrored, so the label refers
            # to the opposite hand.
            if not self.config.mirrored_input:
                label = "right" if label == "left" else "left"
                points[:, 0] = 1.0 - points[:, 0]

            samples.append(HandSample(side=label, points=points, score=score))
        return tuple(samples)


def _visibility(landmark) -> float:
    """Visibility, defaulting to present.

    Tasks marks the field optional and some builds leave it unset. Defaulting to
    1.0 keeps such a build usable -- the visibility gate simply stops filtering,
    which is the pre-existing behaviour of every 2D pose pipeline, rather than
    every landmark reading as invisible and nothing ever being detected.
    """
    value = getattr(landmark, "visibility", None)
    if value is None:
        return 1.0
    return float(value)
