"""End-to-end recognition, driven through the real engine by a scripted body."""

from __future__ import annotations

import math

import numpy as np
import pytest

import synthetic as S
from motionsense import EngineConfig, Phase, Tuning
from motionsense.landmarks import Pose

STANDING = dict(left_arm=(-80.0, 0.0), right_arm=(-80.0, 0.0))


def hold(pose_kwargs, frames: int, **projection):
    """The same pose repeated, as (image, world) pairs."""
    return [S.project(S.body(**pose_kwargs), **projection) for _ in range(frames)]


def standing(frames: int, **projection):
    return hold(STANDING, frames, **projection)


# =====================================================================
# Static poses
# =====================================================================
def test_standing_reports_only_hands_down(harness):
    harness.run(standing(20))
    assert harness.held() == {"hands_down"}
    assert harness.triggered() == []


def test_hands_up(harness):
    poses = standing(10) + hold(dict(left_arm=(85, 0), right_arm=(85, 0)), 15)
    harness.run(poses)
    assert "both_hands_up" in harness.held()
    assert "hands_down" not in harness.held()


def test_single_hand_up_is_exclusive_with_both(harness):
    poses = standing(10) + hold(dict(left_arm=(85, 0), right_arm=(-80, 0)), 15)
    harness.run(poses)
    assert harness.held() == {"left_hand_up"}


def test_t_pose(harness):
    poses = standing(10) + hold(dict(left_arm=(0, 0), right_arm=(0, 0)), 15)
    harness.run(poses)
    assert "t_pose" in harness.held()
    assert "hands_down" not in harness.held()


def test_t_pose_rejects_bent_arms():
    """Straightness is judged by elbow angle, so a raised-but-folded arm is not
    a T-pose even though the wrists are at shoulder height."""
    from conftest import Harness

    joints = S.body(left_arm=(0, 0), right_arm=(0, 0))
    joints[Pose.LEFT_WRIST] = joints[Pose.LEFT_ELBOW] + np.array([0.10, -0.22, 0.0])
    joints[Pose.RIGHT_WRIST] = joints[Pose.RIGHT_ELBOW] + np.array([-0.10, -0.22, 0.0])

    harness = Harness()
    harness.run(standing(8) + [S.project(joints) for _ in range(15)])
    assert "t_pose" not in harness.held()


def folded(left_wrist, right_wrist, left_elbow, right_elbow, *, visibility=1.0):
    """An arms-crossed pose, optionally with the wrists partly occluded."""
    joints = S.body()
    joints[Pose.LEFT_ELBOW] = np.array(left_elbow)
    joints[Pose.RIGHT_ELBOW] = np.array(right_elbow)
    joints[Pose.LEFT_WRIST] = np.array(left_wrist)
    joints[Pose.RIGHT_WRIST] = np.array(right_wrist)
    image, world = S.project(joints)
    image[[Pose.LEFT_WRIST, Pose.RIGHT_WRIST], 3] = visibility
    return image, world


FOLDS = {
    # Forearms across the chest, hands out past the opposite side.
    "wide": dict(
        left_wrist=(0.15, -0.28, 0.10), right_wrist=(-0.15, -0.30, 0.12),
        left_elbow=(-0.22, -0.20, 0.05), right_elbow=(0.22, -0.20, 0.05),
    ),
    # Hands gripping the opposite upper arms.
    "gripping": dict(
        left_wrist=(0.22, -0.24, 0.10), right_wrist=(-0.22, -0.30, 0.12),
        left_elbow=(-0.21, -0.16, 0.02), right_elbow=(0.21, -0.18, 0.02),
    ),
    # A tight fold where the hands barely pass the midline.
    "tight": dict(
        left_wrist=(0.06, -0.26, 0.14), right_wrist=(-0.06, -0.32, 0.12),
        left_elbow=(-0.20, -0.18, 0.02), right_elbow=(0.20, -0.20, 0.02),
    ),
    # Folded low, at the waist rather than the chest.
    "low": dict(
        left_wrist=(0.10, -0.14, 0.14), right_wrist=(-0.10, -0.18, 0.12),
        left_elbow=(-0.19, -0.08, 0.02), right_elbow=(0.19, -0.10, 0.02),
    ),
}


@pytest.mark.parametrize("style", sorted(FOLDS))
def test_arms_crossed(make_harness, style):
    harness = make_harness()
    harness.run(standing(8) + [folded(**FOLDS[style]) for _ in range(15)])
    assert "arms_crossed" in harness.held()


@pytest.mark.parametrize("visibility", [0.45, 0.32])
def test_arms_crossed_survives_occluded_wrists(make_harness, visibility):
    """Folding your arms tucks your hands under the opposite arm, so this is
    exactly the pose that drives wrist visibility down. Applying the global
    visibility gate here made the activity undetectable by construction."""
    harness = make_harness()
    poses = standing(8) + [folded(**FOLDS["tight"], visibility=visibility) for _ in range(15)]
    harness.run(poses)
    assert "arms_crossed" in harness.held()


def test_arms_crossed_ignores_clasped_hands(make_harness):
    """Hands together in front are at the same height and the same place -- the
    wrists have not swapped sides, so there is no crossing."""
    harness = make_harness()
    poses = standing(8) + [
        folded(
            left_wrist=(0.02, -0.20, 0.20), right_wrist=(-0.02, -0.20, 0.20),
            left_elbow=(-0.18, -0.16, 0.06), right_elbow=(0.18, -0.16, 0.06),
        )
        for _ in range(15)
    ]
    harness.run(poses)
    assert "arms_crossed" not in harness.held()


def test_arms_crossed_ignores_one_arm_reaching_across(make_harness):
    harness = make_harness()
    poses = standing(8) + [
        folded(
            left_wrist=(0.20, -0.28, 0.12), right_wrist=(0.46, 0.0, 0.0),
            left_elbow=(-0.20, -0.18, 0.02), right_elbow=(0.30, -0.25, 0.0),
        )
        for _ in range(15)
    ]
    harness.run(poses)
    assert "arms_crossed" not in harness.held()


def test_arms_crossed_is_not_confused_by_an_off_centre_subject(make_harness):
    """The crossing is measured between the wrists, not against the frame
    midline, so standing to one side cannot fake it or mask it."""
    for shift in (-0.18, 0.0, 0.18):
        harness = make_harness()
        image_world = [
            S.project(_folded_joints(**FOLDS["tight"]), shift_x=shift) for _ in range(15)
        ]
        harness.run(standing(8, shift_x=shift) + image_world)
        assert "arms_crossed" in harness.held(), f"shift={shift}"


def _folded_joints(left_wrist, right_wrist, left_elbow, right_elbow):
    joints = S.body()
    joints[Pose.LEFT_ELBOW] = np.array(left_elbow)
    joints[Pose.RIGHT_ELBOW] = np.array(right_elbow)
    joints[Pose.LEFT_WRIST] = np.array(left_wrist)
    joints[Pose.RIGHT_WRIST] = np.array(right_wrist)
    return joints


def test_standing_is_not_arms_crossed(harness):
    harness.run(standing(20))
    assert "arms_crossed" not in harness.held()


@pytest.mark.parametrize(
    "angle,expected",
    [(25.0, "lean_right"), (-25.0, "lean_left"), (5.0, None)],
)
def test_lean(make_harness, angle, expected):
    harness = make_harness()
    harness.run(standing(8) + hold(dict(lean=angle, **STANDING), 15))
    leans = {a for a in harness.held() if a.startswith("lean_")}
    assert leans == ({expected} if expected else set())


def test_head_down(harness):
    harness.run(standing(8) + hold(dict(head_pitch=0.7, **STANDING), 15))
    assert "head_down" in harness.held()


def test_upright_head_is_not_head_down(harness):
    harness.run(standing(20))
    assert "head_down" not in harness.held()


# =====================================================================
# Squat: the case 2D geometry gets wrong
# =====================================================================
def test_squat_detected_from_a_frontal_view(harness):
    """The thigh barely rotates in the image here -- it foreshortens. Only the
    3D knee angle sees this squat, which is the point of using world landmarks."""
    harness.run(standing(10) + hold(dict(squat=0.6, **STANDING), 18))
    assert "squat" in harness.held()


def test_squat_emits_sit_and_stand_edges(harness):
    poses = standing(10) + hold(dict(squat=0.8, **STANDING), 18) + standing(18)
    harness.run(poses)
    assert harness.triggered() == ["sit_down", "stand_up"]
    assert "squat" not in harness.held()


def test_shallow_knee_bend_is_not_a_squat(harness):
    harness.run(standing(10) + hold(dict(squat=0.15, **STANDING), 18))
    assert "squat" not in harness.held()


def test_squat_survives_missing_ankles(make_harness):
    """The very common webcam framing that cuts off the feet. The knee-angle cue
    is unavailable; thigh verticality has to carry it."""
    harness = make_harness()
    poses = []
    for _ in range(10):
        image, world = S.project(S.body(**STANDING))
        image[[Pose.LEFT_ANKLE, Pose.RIGHT_ANKLE], 3] = 0.0
        poses.append((image, world))
    for _ in range(18):
        # A squat seen from an angle, where the thigh really does rotate.
        joints = S.body(**STANDING)
        joints[Pose.LEFT_KNEE] = joints[Pose.LEFT_HIP] + np.array([-0.30, 0.33, 0.0])
        joints[Pose.RIGHT_KNEE] = joints[Pose.RIGHT_HIP] + np.array([0.30, 0.33, 0.0])
        image, world = S.project(joints)
        image[[Pose.LEFT_ANKLE, Pose.RIGHT_ANKLE], 3] = 0.0
        poses.append((image, world))

    harness.run(poses)
    assert "squat" in harness.held()


# =====================================================================
# Jump
# =====================================================================
def jump_sequence(fps: float, *, v0: float = 1.9, before: float = 0.7, after: float = 0.7):
    """A ballistic hop: rise and fall under gravity, standing either side."""
    airtime = 2 * v0 / 9.81
    poses = standing(int(before * fps))
    for i in range(int(airtime * fps)):
        t = i / fps
        poses.append(S.project(S.body(**STANDING), lift=v0 * t - 0.5 * 9.81 * t * t))
    poses += standing(int(after * fps))
    return poses


@pytest.mark.parametrize("fps", [20.0, 30.0, 60.0])
def test_jump_is_detected_at_any_frame_rate(make_harness, fps):
    """Velocity comes from a least-squares fit in units per second, so one
    threshold holds across frame rates. A per-frame delta would need retuning."""
    harness = make_harness()
    harness.run(jump_sequence(fps), fps=fps)
    assert harness.triggered().count("jump") == 1


def test_jump_is_detected_at_any_distance(make_harness):
    for distance in (0.7, 1.0, 1.8):
        harness = make_harness()
        poses = []
        fps = 30.0
        poses += standing(20, distance=distance)
        airtime = 2 * 1.9 / 9.81
        for i in range(int(airtime * fps)):
            t = i / fps
            poses.append(
                S.project(S.body(**STANDING), lift=1.9 * t - 0.5 * 9.81 * t * t, distance=distance)
            )
        poses += standing(20, distance=distance)
        harness.run(poses, fps=fps)
        assert harness.triggered().count("jump") == 1, f"distance={distance}"


def test_standing_still_never_jumps(harness):
    harness.run(standing(90))
    assert "jump" not in harness.triggered()


def test_standing_up_from_a_squat_is_not_a_jump(harness):
    """The torso rises just as fast as in a hop. Posture context is what
    separates them."""
    poses = standing(10)
    for i in range(12):
        poses += hold(dict(squat=0.9 * (1 - i / 12), **STANDING), 1)
    poses += standing(25)
    harness.run(poses)
    assert "jump" not in harness.triggered()
    assert "stand_up" in harness.triggered()


def test_a_small_bob_is_not_a_jump(harness):
    fps = 30.0
    poses = standing(20)
    for i in range(10):
        poses.append(S.project(S.body(**STANDING), lift=0.02 * math.sin(math.pi * i / 10)))
    poses += standing(20)
    harness.run(poses, fps=fps)
    assert "jump" not in harness.triggered()


# =====================================================================
# Wave and swipe
# =====================================================================
def raise_left_arm(fps: float, seconds: float = 0.5):
    """A deliberate arm raise, not a teleport.

    Interpolates joint positions rather than sweeping the shoulder angle through
    its arc: rotating from hanging through horizontal to overhead would carry the
    wrist out to a full arm's length sideways and back, which really is a
    horizontal sweep. People raise a hand by lifting it, bending the elbow, so
    the wrist travels roughly straight up.
    """
    frames = int(seconds * fps)
    down = S.body(left_arm=(-80.0, 0.0), right_arm=(-80.0, 0.0))
    up = S.body(left_arm=(80.0, 0.0), right_arm=(-80.0, 0.0))
    poses = []
    for i in range(frames):
        k = (i + 1) / frames
        blended = {index: (1 - k) * down[index] + k * up[index] for index in down}
        poses.append(S.project(blended))
    return poses


def wave_left(fps: float, cycles: float = 5.0, hz: float = 2.5, amplitude: float = 18.0):
    frames = int(cycles / hz * fps)
    return [
        S.project(
            S.body(
                left_arm=(80.0 + amplitude * math.sin(2 * math.pi * hz * i / fps), 0.0),
                right_arm=(-80.0, 0.0),
            )
        )
        for i in range(frames)
    ]


def test_wave(harness):
    fps = 30.0
    harness.run(standing(6) + raise_left_arm(fps) + wave_left(fps), fps=fps)
    assert "wave_left_hand" in harness.triggered()
    assert "wave_right_hand" not in harness.triggered()


def test_holding_a_hand_up_is_not_a_wave(harness):
    harness.run(standing(6) + hold(dict(left_arm=(85, 0), right_arm=(-80, 0)), 60))
    assert "wave_left_hand" not in harness.triggered()
    assert "left_hand_up" in harness.held()


def reaching(x: float):
    """Body with the right hand held out at chest height, ``x`` metres sideways."""
    joints = S.body(left_arm=(-80.0, 0.0), right_arm=(-30.0, 20.0))
    wrist = np.array([x, -0.30, 0.22])
    shoulder = joints[Pose.RIGHT_SHOULDER]
    joints[Pose.RIGHT_WRIST] = wrist
    joints[Pose.RIGHT_ELBOW] = 0.5 * (shoulder + wrist) + np.array([0.06, 0.07, 0.04])
    return joints


def test_swipe(make_harness):
    """A flat horizontal sweep at chest height, the way people actually swipe."""
    fps = 60.0
    harness = make_harness()
    poses = [S.project(reaching(0.45)) for _ in range(24)]
    steps = 15
    for i in range(steps):
        poses.append(S.project(reaching(0.45 - 0.80 * (i + 1) / steps)))
    poses += [S.project(reaching(-0.35)) for _ in range(30)]

    harness.run(poses, fps=fps)
    assert "swipe_left" in harness.triggered()
    assert "swipe_right" not in harness.triggered()


def test_swipe_the_other_way(make_harness):
    fps = 60.0
    harness = make_harness()
    poses = [S.project(reaching(-0.35)) for _ in range(24)]
    steps = 15
    for i in range(steps):
        poses.append(S.project(reaching(-0.35 + 0.80 * (i + 1) / steps)))
    poses += [S.project(reaching(0.45)) for _ in range(30)]

    harness.run(poses, fps=fps)
    assert "swipe_right" in harness.triggered()


def test_a_wave_does_not_register_as_a_swipe(harness):
    """Both are fast and wide, so speed and travel cannot separate them. Only
    the longer-window directness check does."""
    fps = 60.0
    harness.run(standing(6) + raise_left_arm(fps) + wave_left(fps, cycles=8), fps=fps)

    assert "wave_left_hand" in harness.triggered()
    swipes = [a for a in harness.triggered() if a.startswith("swipe_")]
    assert swipes == []


def test_vertical_swipes_are_off_by_default(make_harness):
    """An arm raise and an upward swipe are the same motion to one camera, so
    the ambiguous pair is opt-in rather than heuristically guessed at."""
    fps = 60.0
    poses = standing(10) + raise_left_arm(fps, seconds=0.35) + hold(
        dict(left_arm=(85, 0), right_arm=(-80, 0)), 20
    )

    default = make_harness()
    default.run(poses, fps=fps)
    assert "swipe_up" not in default.triggered()
    assert "left_hand_up" in default.held()

    opted_in = make_harness(EngineConfig(preset="fast", tuning=Tuning(vertical_swipes=True)))
    opted_in.run(poses, fps=fps)
    assert "swipe_up" in opted_in.triggered()


# =====================================================================
# Hands
# =====================================================================
def test_fist_closed_and_open(make_harness):
    config = EngineConfig(preset="fast", enable_hands=True)

    harness = make_harness(config)
    poses = standing(24)
    hands = [(S.hand("right", curl=0.0),)] * 12 + [(S.hand("right", curl=1.0),)] * 12
    harness.run(poses, hands=hands)

    assert "right_fist_open" in harness.started()
    assert "right_fist_closed" in harness.held()
    assert "right_fist_open" not in harness.held()


def test_pinch(make_harness):
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    poses = standing(24)
    hands = [(S.hand("left", curl=0.0, pinch=1.0),)] * 12 + [
        (S.hand("left", curl=0.0, pinch=0.15),)
    ] * 12
    harness.run(poses, hands=hands)
    assert "left_pinch" in harness.held()


def test_a_real_fist_does_not_report_a_pinch(make_harness):
    """A fist folds the thumb across the curled index, so the two tips end up as
    close together as a deliberate pinch puts them -- this hand's raw gap reads
    0.245 against a 0.34 threshold. The gap alone therefore cannot tell the
    gestures apart; the fingers a pinch does not use are what can."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    hands = [(S.hand("right", curl=1.0, thumb="folded"),)] * 24
    harness.run(standing(24), hands=hands)

    assert "right_fist_closed" in harness.held()
    assert "right_pinch" not in harness.held()


def test_thumbs_up(make_harness):
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    hands = [(S.hand("right", curl=1.0, thumb="folded"),)] * 12 + [
        (S.hand("right", curl=1.0, thumb="up"),)
    ] * 12
    harness.run(standing(24), hands=hands)

    assert "right_thumbs_up" in harness.held()


def test_thumbs_up_does_not_also_report_a_fist(make_harness):
    """A thumbs-up curls exactly the four fingers the fist gate watches, so both
    would hold at once -- and a user who has mapped both is pressing two keys
    with one gesture."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    hands = [(S.hand("left", curl=1.0, thumb="up"),)] * 24
    harness.run(standing(24), hands=hands)

    assert harness.held() & {"left_thumbs_up", "left_fist_closed"} == {"left_thumbs_up"}


def test_a_fist_is_not_a_thumbs_up(make_harness):
    """The thumb is folded rather than extended, which is the whole difference."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    hands = [(S.hand("right", curl=1.0, thumb="folded"),)] * 24
    harness.run(standing(24), hands=hands)

    assert "right_thumbs_up" not in harness.held()


def test_an_open_hand_is_not_a_thumbs_up(make_harness):
    """The thumb points up on an open hand too -- the fingers are what rule it out."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    hands = [(S.hand("right", curl=0.0),)] * 24
    harness.run(standing(24), hands=hands)

    assert "right_thumbs_up" not in harness.held()


def test_fist_closed_without_a_visible_body(make_harness):
    """A desk framing that only shows a hand still has no pose landmarks at
    all -- finger geometry does not need a body frame to be measured."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    poses = [(None, None)] * 24
    hands = [(S.hand("right", curl=1.0),)] * 24
    harness.run(poses, hands=hands)

    assert "right_fist_closed" in harness.held()


def test_hands_down_does_not_fire_without_a_body(make_harness):
    """``hands_down`` is the complement of the other three arm states, which
    makes it a trap for a hands-only frame: no evidence for any of the other
    three must not silently read as 'false', or the complement wrongly reports
    the hands as down when the body was never even seen."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    poses = [(None, None)] * 24
    hands = [(S.hand("right", curl=1.0),)] * 24
    harness.run(poses, hands=hands)

    assert "hands_down" not in harness.held()


def test_fist_closed_when_the_torso_is_out_of_frame(make_harness):
    """Pose landmarks exist but are mostly extrapolated -- e.g. a webcam framed
    tight on the desk. The torso gate must not also silence the hand."""
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)

    def cropped():
        image, world = S.project(S.body(**STANDING))
        image[[Pose.LEFT_HIP, Pose.RIGHT_HIP, Pose.LEFT_SHOULDER, Pose.RIGHT_SHOULDER], 3] = 0.0
        return image, world

    poses = [cropped() for _ in range(24)]
    hands = [(S.hand("right", curl=1.0),)] * 24
    harness.run(poses, hands=hands)

    assert "right_fist_closed" in harness.held()


def test_hand_activities_release_when_the_hand_leaves(make_harness):
    config = EngineConfig(preset="fast", enable_hands=True)
    harness = make_harness(config)
    poses = standing(30)
    hands = [(S.hand("right", curl=1.0),)] * 15 + [()] * 15
    harness.run(poses, hands=hands)
    assert "right_fist_closed" not in harness.held()
    ended = [e.activity for e in harness.events if e.phase is Phase.END]
    assert "right_fist_closed" in ended
