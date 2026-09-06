"""Engine configuration and recognizer tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["EngineConfig", "Tuning", "PRESETS"]

Preset = Literal["fast", "balanced", "accurate"]
Dispatch = Literal["inline", "thread"]
Backend = Literal["auto", "solutions", "tasks"]


@dataclass(slots=True)
class Tuning:
    """Thresholds for the built-in recognizers.

    All distances are in *body units* (1.0 is about one shoulder-to-hip torso)
    and all angles are in degrees, so a threshold means the same thing whether
    the subject is two feet or fifteen feet from the camera, and whether they are
    tall or short. Velocities are body units per *second*, so they mean the same
    thing at 15 FPS and at 60 FPS.

    Level thresholds come in ``(enter, exit)`` pairs that form the hysteresis
    band. Widening a band trades reaction speed for stability.
    """

    # -- gating ---------------------------------------------------------------
    #: Landmarks below this visibility score are treated as missing rather than
    #: being used at a guessed position. MediaPipe happily extrapolates points
    #: that are out of frame, and trusting those is the single largest source of
    #: phantom activations.
    min_visibility: float = 0.55
    #: Minimum dwell before a level activity turns on / off (seconds).
    level_min_on: float = 0.06
    level_min_off: float = 0.10

    # -- arms ------------------------------------------------------------------
    hand_up_enter: float = 0.06        # wrist above nose by this much
    hand_up_exit: float = 0.01
    t_pose_arm_angle: float = 24.0     # max deviation from horizontal
    t_pose_elbow_min: float = 148.0    # elbow angle => arm is straight
    t_pose_spread: float = 0.35        # wrist distance from midline
    #: Multi-condition poses reduce their constraints to one dimensionless
    #: margin (1 = comfortably inside, 0 = exactly at the limit). These are the
    #: hysteresis band on that margin, shared by t_pose and arms_crossed.
    pose_margin_exit: float = -0.30
    #: How far the wrists must swap sides, measured as ``left.x - right.x``.
    #: Relative rather than per-wrist-past-the-midline, so it survives a subject
    #: standing off-centre and an asymmetric fold. Standing at rest reads about
    #: -0.93; a tight fold reads +0.21 and a wide one +0.77.
    arms_crossed_separation: float = 0.12
    #: Bent elbows corroborate the pose but never veto it -- see
    #: ``arms_crossed_min_visibility`` for why. Loose enough to tolerate the
    #: noisy depth estimates that self-occlusion produces.
    arms_crossed_elbow_max: float = 140.0
    #: Folded arms tuck the hands under the opposite arm, so this is precisely
    #: the pose that drives wrist visibility down -- applying the global
    #: ``min_visibility`` here would make the activity undetectable by
    #: construction. The crossing test is a strong enough constraint that a badly
    #: extrapolated wrist will not satisfy it anyway.
    arms_crossed_min_visibility: float = 0.30

    # -- posture ---------------------------------------------------------------
    lean_enter_deg: float = 17.0
    lean_exit_deg: float = 11.0
    head_down_enter: float = 0.15      # nose height above shoulder line
    head_down_exit: float = 0.21
    #: When a calibration exists these ratios of the subject's own neutral head
    #: height replace the absolutes above, which removes the dependence on body
    #: proportion and camera height.
    head_down_enter_ratio: float = 0.50
    head_down_exit_ratio: float = 0.70
    #: cos(thigh angle from vertical): 1.0 standing, 0.0 thigh horizontal.
    squat_enter: float = 0.80
    squat_exit: float = 0.89
    squat_knee_enter_deg: float = 132.0
    squat_knee_exit_deg: float = 152.0
    posture_min_on: float = 0.10
    posture_min_off: float = 0.14

    # -- jump --------------------------------------------------------------------
    #: Body units per second, upward. A modest hop measures 1.4-1.8 here while a
    #: still body's noise floor sits near 0.07, so there is over a decade of
    #: margin; the threshold is set low enough to catch a small jump at 20 FPS,
    #: where the ballistic peak is temporally under-resolved.
    jump_takeoff_speed: float = 0.95
    jump_landing_speed: float = -0.60
    jump_min_rise: float = 0.06        # peak displacement above the launch height
    jump_max_airtime: float = 0.95
    jump_cooldown: float = 0.45

    # -- wave ---------------------------------------------------------------------
    wave_window: float = 1.1
    wave_min_half_cycles: int = 3
    wave_min_amplitude: float = 0.09
    wave_freq_band: tuple[float, float] = (1.1, 6.5)
    wave_cooldown: float = 0.85

    # -- swipe ----------------------------------------------------------------------
    swipe_speed: float = 1.9           # body units / second
    swipe_min_travel: float = 0.42
    swipe_axis_ratio: float = 1.7      # dominant axis must beat the other by this
    swipe_cooldown: float = 0.6
    #: A wave is an oscillation and a swipe is a single sweep; without this a
    #: wave's outbound stroke also reads as a swipe.
    swipe_wave_suppression: float = 0.9
    #: Directness must also hold over this longer window, which has to exceed one
    #: wave period. Inside 0.22 s a wave's half-cycle is monotonic and
    #: indistinguishable from a sweep; over 0.45 s the wave has come back and the
    #: swipe has not.
    swipe_confirm_window: float = 0.45
    swipe_confirm_directness: float = 0.55
    swipe_directness: float = 0.7
    #: ``swipe_up`` / ``swipe_down`` are off by default. Raising a hand and
    #: swiping upward are the *same* physical motion, and a single camera has no
    #: information that separates them -- any threshold that catches a real
    #: upward swipe also fires when the user simply lifts their hand to wave.
    #: Horizontal swipes have no such twin, which is why they are on. Enable this
    #: if vertical scroll control is worth the overlap for your application.
    vertical_swipes: bool = False

    # -- hands ----------------------------------------------------------------------
    fist_closed_enter: float = 0.92    # mean fingertip/PIP radial ratio
    fist_closed_exit: float = 1.06
    pinch_enter: float = 0.34          # thumb-index gap / hand scale
    pinch_exit: float = 0.46
    #: A fist folds the thumb across the curled index, putting the two tips
    #: together -- geometrically a pinch, and the gap alone cannot tell them
    #: apart. What can is the rest of the hand: a pinch leaves the middle, ring
    #: and pinky out of it. This only has to clear the fist boundary above, not
    #: demand a splayed hand, or an ordinary pinch with relaxed fingers would
    #: stop registering.
    pinch_outer_curl_min: float = 0.95
    #: A thumbs-up curls the same four fingers a fist does, so it is separated by
    #: the thumb alone: extended rather than folded away, and pointing up.
    thumbs_up_extension_min: float = 1.05
    #: cos(angle from vertical), so 0.70 allows about 45 degrees of hand tilt.
    thumbs_up_direction_min: float = 0.70
    hand_min_score: float = 0.6


@dataclass(slots=True)
class EngineConfig:
    """How the engine captures, infers and dispatches.

    The defaults target a 30 FPS webcam on a laptop CPU. Start from a preset and
    override what you need::

        EngineConfig(preset="fast", enable_hands=False)
    """

    preset: Preset = "balanced"

    # -- model ------------------------------------------------------------------
    #: Which MediaPipe API to use. ``"auto"`` prefers the legacy
    #: ``mediapipe.solutions`` API when present (0.10.x, models bundled in the
    #: wheel, no setup) and otherwise uses the Tasks API (1.0+, which needs
    #: ``.task`` bundles -- see ``model_dir``).
    backend: Backend = "auto"
    #: Directory holding MediaPipe Tasks ``.task`` bundles. ``None`` searches
    #: ``$MOTIONSENSE_MODEL_DIR`` then the per-user cache. Ignored by the legacy
    #: backend, which has its models built in.
    model_dir: str | None = None
    #: Whether a missing Tasks model bundle may be fetched from Google's model
    #: storage into the cache directory. Set ``False`` in environments where a
    #: library must not reach the network; you then have to place the file
    #: yourself, and the error message says where.
    allow_model_download: bool = True
    #: 0 = lite, 1 = full, 2 = heavy. Set by the preset unless given explicitly.
    model_complexity: int | None = None
    #: Longest edge fed to the detector. Frames are downscaled to this first.
    #: The model resizes to its own input size internally anyway, so sending a
    #: 1080p frame only pays for a larger resize. Set to ``None`` to disable.
    inference_width: int | None = None
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    #: MediaPipe's own landmark smoothing. Off by default because the SDK applies
    #: a 1-Euro filter, which adapts its cutoff to movement speed and therefore
    #: costs less lag for the same steadiness. Stacking both over-smooths.
    model_smoothing: bool = False

    # -- inputs -----------------------------------------------------------------
    #: ``True`` runs the hand model, ``False`` never does, ``"auto"`` runs it only
    #: when something is subscribed to a hand activity. The hand model roughly
    #: doubles per-frame cost, so ``"auto"`` is the default.
    enable_hands: bool | Literal["auto"] = "auto"
    #: Run the hand model every Nth frame. Finger states change slowly relative
    #: to 30 FPS, so 2 is usually indistinguishable and buys back ~25% of frame time.
    hands_interval: int = 1
    max_hands: int = 2
    #: Set ``True`` if frames are already mirrored (selfie view). A raw webcam
    #: feed is not, which is the default. Getting this wrong swaps left and right.
    mirrored_input: bool = False

    # -- smoothing ----------------------------------------------------------------
    #: 1-Euro minimum cutoff (Hz). The cutoff used when the subject is still, so
    #: lower means steadier held poses and more lag on movement.
    filter_min_cutoff: float = 1.5
    #: 1-Euro speed coefficient, in Hz per unit of speed. Speed here is in
    #: normalised image units per second, where ordinary motion is order 1 --
    #: not pixels per second, where it is order 100. Published beta values are
    #: usually quoted for pixels; using one of those here (~0.01) leaves the
    #: cutoff essentially fixed and costs about 30% of the peak velocity of a
    #: ballistic gesture, without buying any measurable steadiness in return.
    filter_beta: float = 1.5
    #: Set ``False`` to receive raw landmarks (e.g. you filter downstream).
    filter_landmarks: bool = True

    # -- dispatch -------------------------------------------------------------------
    #: ``"inline"`` calls listeners on the inference thread: lowest possible
    #: latency, but a slow listener stalls detection. ``"thread"`` hands events to
    #: a worker queue: use it when listeners do I/O (HTTP, serial, MQTT).
    dispatch: Dispatch = "inline"
    #: Emit ``Phase.UPDATE`` every frame a level activity is held. Off by default
    #: since START/END is enough for most bindings.
    emit_updates: bool = False
    #: Attach the source image to ``FrameResult``. Only needed for preview.
    deliver_frames: bool = False

    # -- scope ----------------------------------------------------------------------
    #: Restrict recognition to these activity ids. ``None`` means everything.
    #: Narrowing this lets the engine skip work and skip models entirely.
    activities: frozenset[str] | None = None

    tuning: Tuning = field(default_factory=Tuning)

    def __post_init__(self):
        if isinstance(self.activities, (list, tuple, set)):
            self.activities = frozenset(self.activities)
        base = PRESETS.get(self.preset)
        if base is None:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}")
        if self.model_complexity is None:
            self.model_complexity = base["model_complexity"]
        if self.inference_width is None:
            self.inference_width = base["inference_width"]
        if self.hands_interval <= 0:
            raise ValueError("hands_interval must be >= 1")

    def wants(self, activity_id: str) -> bool:
        return self.activities is None or activity_id in self.activities


PRESETS: dict[str, dict] = {
    # Lite model, small input: ~2-3x the throughput of "balanced" with slightly
    # noisier landmarks. Good for always-on control where latency dominates.
    "fast": {"model_complexity": 0, "inference_width": 480},
    "balanced": {"model_complexity": 1, "inference_width": 640},
    # Heavy model: best landmark quality, roughly half the frame rate.
    "accurate": {"model_complexity": 2, "inference_width": 800},
}
