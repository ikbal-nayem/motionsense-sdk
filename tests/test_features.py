"""The normalisation contract: features must not depend on framing."""

from __future__ import annotations

import numpy as np
import pytest

import synthetic as S
from motionsense import EngineConfig
from motionsense.features import FeatureExtractor
from motionsense.landmarks import Pose

CONFIG = EngineConfig(preset="fast")


def extract(joints, *, frames: int = 5, **projection):
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(joints, **projection)
    features = None
    for i in range(frames):
        features = extractor.update(image, world, (), i / 30.0, projection.get("aspect", 4 / 3))
    return features


def compare(a, b, *, tol=1e-3):
    for name in (
        "shoulder_center_height",
        "nose_above_shoulders",
        "torso_lean",
        "left_hand_rise",
        "right_hand_rise",
        "left_elbow_angle",
        "left_arm_tilt",
        "left_knee_angle",
        "left_thigh_vertical",
        "thigh_length",
        "arm_length",
    ):
        assert getattr(a, name) == pytest.approx(getattr(b, name), abs=tol), name


def test_features_are_invariant_to_distance():
    """Half the size in frame must produce the same numbers, or every threshold
    would silently mean something different at a different camera distance."""
    near = extract(S.body(), distance=1.0)
    far = extract(S.body(), distance=2.0)

    assert far.scale == pytest.approx(near.scale / 2, rel=0.02)
    compare(near, far)


def test_features_are_invariant_to_position_in_frame():
    centred = extract(S.body())
    offset = extract(S.body(), shift_x=0.18)
    compare(centred, offset)


def test_features_are_invariant_to_frame_aspect_ratio():
    """Detector coordinates are normalised per axis, so without aspect
    correction a 16:9 camera and a 4:3 camera disagree about every length."""
    four_three = extract(S.body(), aspect=4 / 3)
    sixteen_nine = extract(S.body(), aspect=16 / 9)
    compare(four_three, sixteen_nine, tol=5e-3)


def test_body_scale_matches_hand_computation():
    features = extract(S.body())
    assert features.scale == pytest.approx(0.306, abs=0.005)
    assert features.shoulder_center_height == pytest.approx(0.878, abs=0.01)
    assert features.nose_above_shoulders == pytest.approx(0.316, abs=0.01)


def test_lean_angle_is_reported_directly_in_degrees():
    for angle in (-25.0, -10.0, 0.0, 15.0, 30.0):
        features = extract(S.body(lean=angle))
        assert features.torso_lean == pytest.approx(angle, abs=0.5)


def test_low_visibility_landmarks_become_nan_not_guesses():
    """MediaPipe extrapolates points that are out of frame. Using them is the
    main source of phantom activations, so they must read as unknown."""
    joints = S.body()
    image, world = S.project(joints)
    image[Pose.LEFT_WRIST, 3] = 0.1
    image[Pose.LEFT_ANKLE, 3] = 0.1

    extractor = FeatureExtractor(CONFIG)
    features = None
    for i in range(4):
        features = extractor.update(image, world, (), i / 30.0, 4 / 3)

    assert np.isnan(features.left_hand_rise)
    assert np.isnan(features.left_knee_angle)
    # The visible side still reports, so a partly occluded body degrades rather
    # than going dark.
    assert not np.isnan(features.right_hand_rise)
    assert not np.isnan(features.right_knee_angle)


def test_missing_torso_yields_no_features():
    joints = S.body()
    image, world = S.project(joints)
    image[[Pose.LEFT_HIP, Pose.RIGHT_HIP], 3] = 0.0

    extractor = FeatureExtractor(CONFIG)
    assert extractor.update(image, world, (), 0.0, 4 / 3) is None


def test_hand_curl_and_pinch_are_scale_free():
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())

    small = S.hand("right", curl=0.0, pinch=1.0, center=(0.5, 0.5))
    big = S.hand("right", curl=0.0, pinch=1.0, center=(0.5, 0.5))
    big.points = big.points * 2.0  # twice as close to the camera

    a = extractor.update(image, world, (small,), 0.0, 4 / 3)
    extractor.reset()
    b = extractor.update(image, world, (big,), 0.0, 4 / 3)

    assert a.hands[0].curl == pytest.approx(b.hands[0].curl, rel=1e-3)
    assert a.hands[0].pinch == pytest.approx(b.hands[0].pinch, rel=1e-3)


def test_hand_curl_separates_open_from_closed():
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())

    open_hand = extractor.update(image, world, (S.hand("left", curl=0.0),), 0.0, 4 / 3).hands[0]
    extractor.reset()
    fist = extractor.update(image, world, (S.hand("left", curl=1.0),), 0.0, 4 / 3).hands[0]

    assert open_hand.curl > 1.06
    assert fist.curl < 0.92
