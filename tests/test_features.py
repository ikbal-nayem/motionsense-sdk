"""The normalisation contract: features must not depend on framing."""

from __future__ import annotations

import numpy as np
import pytest

import synthetic as S
from motionsense import EngineConfig
from motionsense.features import FeatureExtractor
from motionsense.landmarks import Hand, Pose

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


def test_hands_report_without_any_pose():
    """A close desk framing that cuts the torso out of the shot still has a
    hand model result; finger geometry needs no body frame to be usable."""
    extractor = FeatureExtractor(CONFIG)
    features = extractor.update(None, None, (S.hand("right", curl=1.0),), 0.0, 4 / 3)

    assert features is not None
    assert features.hands[0].side == "right"
    assert features.hands[0].curl < 0.92
    assert np.isnan(features.scale)
    assert np.isnan(features.torso_lean)
    assert np.isnan(features.left_elbow_angle)


def test_hands_report_when_torso_is_not_visible():
    joints = S.body()
    image, world = S.project(joints)
    image[[Pose.LEFT_HIP, Pose.RIGHT_HIP, Pose.LEFT_SHOULDER, Pose.RIGHT_SHOULDER], 3] = 0.0

    extractor = FeatureExtractor(CONFIG)
    features = extractor.update(image, world, (S.hand("left", curl=1.0),), 0.0, 4 / 3)

    assert features is not None
    assert features.hands[0].side == "left"
    assert features.hands[0].curl < 0.92
    assert np.isnan(features.shoulder_center_height)


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


def test_pinch_is_not_fooled_by_a_gap_along_the_view_direction():
    """The reported failure: fingers plainly apart, but reported as pinching.

    A hand angled toward the camera separates thumb from index mostly in depth,
    which projects to almost no gap at all. Measured on the image plane the two
    tips look touching; measured in 3D they are as far apart as they really are.
    """
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())

    hand = S.hand("right", curl=0.0, pinch=1.0)
    # Park the thumb tip right on top of the index tip in x/y, then pull it well
    # away from the camera. Nothing about the real gap has shrunk.
    hand.points[Hand.THUMB_TIP] = (
        hand.points[Hand.INDEX_TIP][0],
        hand.points[Hand.INDEX_TIP][1],
        hand.points[Hand.INDEX_TIP][2] + 0.10,
    )

    features = extractor.update(image, world, (hand,), 0.0, 4 / 3)
    assert features.hands[0].pinch > CONFIG.tuning.pinch_exit


def hand_features(**kwargs):
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())
    return extractor.update(image, world, (S.hand("right", **kwargs),), 0.0, 4 / 3).hands[0]


def test_outer_curl_ignores_the_index():
    """A pinch curls the index to meet the thumb, so the four-finger mean drops
    toward a fist. Excluding the index is what keeps the two distinguishable."""
    pinching = hand_features(curl=0.0, pinch=0.15)
    fist = hand_features(curl=1.0)

    assert pinching.outer_curl > 1.2  # middle, ring and pinky still extended
    assert fist.outer_curl < 0.92  # every finger in


def test_thumb_extension_separates_a_folded_thumb_from_an_extended_one():
    folded = hand_features(curl=1.0, thumb="folded")
    extended = hand_features(curl=1.0, thumb="up")

    assert folded.thumb_extension < 1.0  # tip no further out than its own joint
    assert extended.thumb_extension > 1.3


def test_thumb_direction_is_signed_against_image_up():
    """Hand landmarks keep the image's y-down convention, unlike the body frame.
    Getting the sign wrong here would invert the gesture entirely."""
    up = hand_features(curl=1.0, thumb="up")
    assert up.thumb_up == pytest.approx(1.0, abs=0.05)

    down = S.hand("right", curl=1.0, thumb="up")
    # Reflect the thumb below its own knuckle to point the same hand downward.
    mcp_y = down.points[Hand.THUMB_MCP][1]
    for index in (Hand.THUMB_IP, Hand.THUMB_TIP):
        down.points[index][1] = mcp_y + (mcp_y - down.points[index][1])
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())
    flipped = extractor.update(image, world, (down,), 0.0, 4 / 3).hands[0]
    assert flipped.thumb_up == pytest.approx(-1.0, abs=0.05)


def test_a_folded_thumb_looks_like_a_pinch_by_gap_alone():
    """The reason the gap needs corroborating: this hand is a fist, and its raw
    thumb-to-index gap sits below the pinch threshold anyway."""
    fist = hand_features(curl=1.0, thumb="folded")
    assert fist.pinch < CONFIG.tuning.pinch_enter
    assert fist.outer_curl < CONFIG.tuning.pinch_outer_curl_min  # what rules it out


def test_hand_curl_separates_open_from_closed():
    extractor = FeatureExtractor(CONFIG)
    image, world = S.project(S.body())

    open_hand = extractor.update(image, world, (S.hand("left", curl=0.0),), 0.0, 4 / 3).hands[0]
    extractor.reset()
    fist = extractor.update(image, world, (S.hand("left", curl=1.0),), 0.0, 4 / 3).hands[0]

    assert open_hand.curl > 1.06
    assert fist.curl < 0.92
