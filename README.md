# motionsense

[![test](https://github.com/ikbal-nayem/motionsense-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/ikbal-nayem/motionsense-sdk/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Real-time human activity events from any video source.

Feed it video, subscribe to activities, drive anything. The SDK detects *what
the body is doing* and reports it. What that means — a keystroke, a function
call, an MQTT publish, a servo angle — is your application's decision, and
nothing in the detection path knows or cares.

```python
from motionsense import MotionEngine

engine = MotionEngine()

@engine.on("swipe_right")
def next_slide(event):
    deck.advance()

@engine.on_start("left_hand_up")
def thrusters_on(event):
    ship.thrust = True

@engine.on_end("left_hand_up")
def thrusters_off(event):
    ship.thrust = False

engine.start()          # background threads, returns immediately
```

---

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
- [Activity catalog](#activity-catalog)
- [API](#api)
- [Extending](#extending)
- [Accuracy](#accuracy)
- [Performance](#performance)
- [Calibration](#calibration)
- [Testing](#testing)
- [How the MotionKey app uses it](#how-the-motionkey-app-uses-it)
- [License](#license)

---

## Install

```bash
pip install git+https://github.com/ikbal-nayem/motionsense-sdk.git          # core: numpy, opencv, mediapipe
pip install "motionsense[keys] @ git+https://github.com/ikbal-nayem/motionsense-sdk.git"  # + keyboard/mouse output
```

Working on the SDK itself, or pinning it into another project from a local
checkout:

```bash
pip install -e path/to/motionsense-sdk          # core: numpy, opencv, mediapipe
pip install -e "path/to/motionsense-sdk[keys]"  # + keyboard/mouse output
```

Python 3.10+. Works with both MediaPipe generations:

| MediaPipe | Backend used | Setup |
|---|---|---|
| 0.10.x | `mediapipe.solutions` | none — models ship in the wheel |
| 1.0+ | Tasks API | fetches `.task` bundles (~6 MB) into a user cache on first run |

Selection is automatic. To avoid the network entirely, download the bundles
yourself and point `MOTIONSENSE_MODEL_DIR` at them, or set
`EngineConfig(allow_model_download=False)` — the resulting error names the exact
files and URLs.

---

## Quickstart

```python
import time
from motionsense import MotionEngine

engine = MotionEngine()
engine.on_any(print)
engine.start()

try:
    while engine.running:
        time.sleep(0.5)
finally:
    engine.stop()
```

Runnable examples in [`examples/`](examples/):

| File | Shows |
|---|---|
| `00_check_camera.py` | verify a camera works, in isolation |
| `01_listen.py` | minimal listener |
| `02_keyboard.py` | game control via key bindings |
| `03_custom_activity.py` | your own activities, two ways |
| `04_offline_video.py` | deterministic analysis of a recording |
| `05_control_profiles.py` | one gesture set per context, swapped at runtime |
| `06_preview.py` | live preview window with overlay |
| `07_diagnose.py` | why a pose isn't firing |

---

## Core concepts

### Poses and gestures

Two kinds of activity, distinguished by `Trigger`:

**Poses** (`Trigger.LEVEL`) are true for as long as they hold. They produce
`START` when they begin and `END` when released, and `event.duration` on the
`END` says how long. Natural fit for a held control.

**Gestures** (`Trigger.EDGE`) fire once when recognised, producing `TRIGGER`.
Natural fit for a command.

`engine.on(activity, fn)` subscribes to whichever of those is natural for that
activity, so you rarely have to think about it. `on_start`, `on_end`,
`on_update` and `on_trigger` are explicit when you need them.

### Events

```python
Event(
    activity="jump",        # activity id
    phase=Phase.TRIGGER,    # START | UPDATE | END | TRIGGER
    timestamp=12.480,       # seconds since the engine started
    wall_time=1765...,      # time.time() at the same moment
    confidence=0.86,        # 0..1
    duration=0.0,           # how long held (END / UPDATE)
    data={"rise": 0.31, "airtime": 0.38, "peak_speed": 1.72},
    frame_index=374,
)
```

`data` carries recognizer-specific extras: a jump reports its height and
airtime, a wave its frequency and amplitude, a swipe its speed and which hand.

### Where events come from

```
VideoSource ──► LandmarkProvider ──► FeatureExtractor ──► Recognizers ──► Dispatcher ──► your code
 camera,          MediaPipe            normalise           poses and        events
 file, your       pose + hands         and filter          gestures
 own frames
```

Each stage is replaceable. Bring your own `VideoSource` for an RTSP stream or a
frame grabber; your own `LandmarkProvider` for a different pose model; your own
`Recognizer` for activities the SDK does not ship with.

---

## Activity catalog

25 built-in activities. Ids are stable.

**Arms**

| id | kind | needs | description |
|---|---|---|---|
| `both_hands_up` | pose | pose | Both wrists raised above head height |
| `left_hand_up` | pose | pose | Left wrist raised above head height |
| `right_hand_up` | pose | pose | Right wrist raised above head height |
| `hands_down` | pose | pose | Both hands resting below head height |
| `t_pose` | pose | pose | Both arms straight and horizontal, out to the sides |
| `arms_crossed` | pose | pose | Wrists crossed over the midline in front of the chest |

**Posture**

| id | kind | needs | description |
|---|---|---|---|
| `lean_left` | pose | pose | Torso tilted toward the subject's left |
| `lean_right` | pose | pose | Torso tilted toward the subject's right |
| `squat` | pose | pose | Knees bent, hips lowered |
| `head_down` | pose | pose | Head lowered or bowed toward the chest |

**Motion**

| id | kind | needs | description |
|---|---|---|---|
| `jump` | gesture | pose | A ballistic upward launch followed by a landing |
| `sit_down` | gesture | pose | Transition from standing into a lowered posture |
| `stand_up` | gesture | pose | Transition from a lowered posture back to standing |
| `wave_left_hand` | gesture | pose | Left hand raised and swung side to side |
| `wave_right_hand` | gesture | pose | Right hand raised and swung side to side |
| `swipe_left` | gesture | pose | A single fast hand sweep toward the subject's left |
| `swipe_right` | gesture | pose | A single fast hand sweep toward the subject's right |
| `swipe_up` † | gesture | pose | A single fast upward hand sweep |
| `swipe_down` † | gesture | pose | A single fast downward hand sweep |

† Off by default. Raising a hand and swiping upward are the *same* physical
motion, and one camera has no information that separates them — any threshold
that catches a real upward swipe also fires when the user lifts their hand to
wave. Horizontal swipes have no such twin. Enable with
`Tuning(vertical_swipes=True)` if scroll control is worth the overlap.

**Hands** — these need the hand model, which roughly doubles per-frame cost. It
is enabled automatically only when something subscribes to one of them.

| id | kind | needs | description |
|---|---|---|---|
| `left_fist_closed` | pose | hands | Left hand closed into a fist |
| `left_fist_open` | pose | hands | Left hand open with fingers extended |
| `right_fist_closed` | pose | hands | Right hand closed into a fist |
| `right_fist_open` | pose | hands | Right hand open with fingers extended |
| `left_pinch` | pose | hands | Left thumb and index fingertip touching |
| `right_pinch` | pose | hands | Right thumb and index fingertip touching |
| `left_thumbs_up` | pose | hands | Left hand closed with the thumb extended upward |
| `right_thumbs_up` | pose | hands | Right hand closed with the thumb extended upward |

Left and right are always the **subject's** own, corrected for mirroring.

The hand gestures are kept from contradicting each other, because a mapped key
should not be pressed by a gesture that merely resembles the one it is bound to.
`fist_open` and `fist_closed` share one hysteresis band, so a half-curled hand
reports neither rather than alternating between them. A thumbs-up curls the same
four fingers a fist does, so it suppresses `fist_closed` — the more specific
reading wins. And a pinch has to be corroborated by the fingers it does *not*
use: a fist folds the thumb across the curled index, which puts those two tips
as close together as a deliberate pinch does, so the gap alone cannot tell them
apart. `thumbs_up` is judged on the thumb being extended and pointing up
(`thumbs_up_direction_min` is a cosine, so the default 0.70 allows about 45° of
hand tilt).

```python
from motionsense import catalog
catalog.all_activities()   # every ActivityDef, including ones you registered
catalog.categories()
catalog.require("jmup")    # KeyError: unknown activity 'jmup'. Did you mean: jump?
```

---

## API

### MotionEngine

```python
MotionEngine(config=None, *, provider=None, recognizers=None)
```

**Subscribing** — every `on_*` returns a `Subscription`; call it or
`.cancel()` to unsubscribe. Unknown activity ids raise immediately rather than
silently never firing.

```python
engine.on(activity, callback=None, *, phase=None)   # decorator or direct call
engine.on_start(activity, cb)      engine.on_end(activity, cb)
engine.on_trigger(activity, cb)    engine.on_update(activity, cb)
engine.on_any(cb)                  # every event
engine.on_frame(cb)                # each processed FrameResult
engine.on_error(cb)                # capture/inference failures
```

**Running**

```python
engine.start(source=None)   # background threads; source may be a VideoSource,
                            # a camera index, a file path, or None for camera 0
engine.run(source=None)     # same, on the calling thread
engine.stop(timeout=3.0)    # releases every held activity first
engine.submit(image, timestamp=None) -> FrameResult   # synchronous, no threads
```

`stop()` matters: it emits `END` for everything still held, which is what lifts
a key a binding was holding. It is safe to call from inside a listener.

**State**

```python
engine.active      # frozenset of activity ids currently held
engine.tracking    # is a body visible
engine.snapshot()  # Snapshot(running, active, fps, latency, frames, dropped, tracking)
```

### EngineConfig

```python
EngineConfig(
    preset="balanced",          # "fast" | "balanced" | "accurate"
    enable_hands="auto",        # True | False | "auto"
    dispatch="inline",          # "inline" | "thread"
    activities=None,            # restrict to a set of ids
    mirrored_input=False,       # True if your frames are already selfie-flipped
    deliver_frames=False,       # attach the image to FrameResult
    preview_interval=1,         # attach it every Nth frame instead
    emit_updates=False,         # UPDATE events every frame a pose is held
    tuning=Tuning(),            # all recognizer thresholds
)
```

`preview_interval` exists because drawing a preview is not free and is not
free *somewhere harmless*: the skeleton overlay costs about 0.7 ms and is spent
on the thread that detects, so on hardware where inference alone already fills
the frame budget it comes out of reaction time. Raising it thins only the
preview — activity state, events and stats still update on every frame, so a
mapped key never waits for a frame that happens to be drawing:

```python
EngineConfig(deliver_frames=True, preview_interval=3)   # preview at a third the rate
```

Skipped frames arrive with `image=None`, exactly as they do when
`deliver_frames` is off, so `render()` returns `None` and there is nothing new
to branch on.

Presets set model complexity and inference resolution: `fast` (lite model,
480 px), `balanced` (full, 640), `accurate` (heavy, 800).

`dispatch="inline"` calls listeners on the recognition thread — lowest latency,
but a slow listener stalls detection. Use `"thread"` when listeners do I/O
(HTTP, serial, MQTT).

`mirrored_input` is the one setting that is silently wrong if you get it
backwards: it swaps left and right. Raw webcam frames are **not** mirrored,
which is the default.

### Tuning

Every threshold, in units that do not depend on framing: distances in **body
units** (1.0 ≈ one shoulder-to-hip torso), angles in **degrees**, velocities in
**body units per second**. A threshold means the same thing whether the subject
is two feet or fifteen from the camera, and at 15 FPS or 60.

```python
from motionsense import EngineConfig, Tuning

config = EngineConfig(tuning=Tuning(
    jump_takeoff_speed=1.3,     # harder to trigger a jump
    lean_enter_deg=22.0,        # require a deeper lean
    level_min_on=0.12,          # longer dwell before a pose registers
    vertical_swipes=True,
))
```

Level thresholds come in `(enter, exit)` pairs forming a hysteresis band.
Widening a band trades reaction speed for stability.

### Bindings

**Keyboard** (`pip install motionsense[keys]`) — poses are held, gestures are
tapped, chosen from each activity's own kind:

```python
from motionsense.bindings import KeyBindings

keys = KeyBindings(engine, {
    "left_hand_up": "a",
    "squat": "ctrl",
    "jump": "space",
})
```

Key holds are reference-counted, so overlapping poses mapped to the same key
cannot release it out from under each other. Taps schedule their release on a
timer rather than sleeping, so a tap never stalls the detection loop.

**Functions**, as a swappable mapping — for when the mapping is data rather
than code:

```python
from motionsense.bindings import ActionRouter

router = ActionRouter(engine, {
    "swipe_left":  library.previous_track,
    "swipe_right": library.next_track,
    "left_pinch":  {"start": grab, "end": release},
    "jump":        lambda e: log(f"jumped at {e.confidence:.2f}"),
})

router.load(DRIVING_PROFILE)   # atomically replaces the whole mapping
router.enabled = False         # pause without tearing anything down
```

Actions may take the `Event` or no arguments; both are detected.

### Sources

```python
from motionsense import CameraSource, VideoFileSource, IterableSource, CallableSource

CameraSource(0, width=1280, height=720, warmup=4.0, read_timeout=2.0)
VideoFileSource("clip.mp4", realtime=False, loop=False)
IterableSource(frames, fps=30.0)        # any iterable of BGR arrays
CallableSource(grab_frame)              # pull from a function
```

`CameraSource` treats a camera as unreliable, because it is. `warmup` is how
long to wait for the *first* frame: opening a device only reserves the handle,
and a webcam that reports itself open commonly fails its first reads for a
second or more while exposure settles. `read_timeout` is how long reads may keep
failing mid-stream before the feed is declared lost — USB cameras glitch
routinely, and treating one dropped read as a dead camera ends sessions that
would have recovered on the next frame.

If a camera misbehaves, check it in isolation before suspecting detection:

```bash
python examples/00_check_camera.py      # probes indices 0-3, reports what each does
```

Subclass `VideoSource` for anything else. Two flags shape how the engine treats
it: `drop_stale` (skip a backlog — right for live, wrong for recordings) and
`is_live` (a read returning nothing is a lull, not the end).

---

## Extending

### A function

```python
from motionsense import Pose

engine.define(
    "hand_over_heart",
    lambda f: -math.hypot(f.P[Pose.RIGHT_WRIST][0] + 0.18,
                          f.P[Pose.RIGHT_WRIST][1] - 0.62),
    trigger="level",
    enter=-0.20,   # within 0.20 body units
    exit=-0.30,
)
```

The function receives `BodyFeatures` — already normalised, filtered and
visibility-tagged — and returns a number (compared against `enter`/`exit` with
hysteresis) or a bool. Return `float('nan')` when the frame cannot answer; that
releases the activity rather than latching it.

### A recognizer

For anything needing state across frames, subclass `Recognizer`:

```python
class LeanOscillation(Recognizer):
    activities = (ActivityDef("rocking", "Rocking", "Custom", Trigger.EDGE, "..."),)

    def update(self, features, out):
        ...
        out.edge("rocking", confidence=0.9, amplitude=12.4)

    def reset(self):
        ...   # called when tracking is lost

engine.add_recognizer(LeanOscillation())
```

`out.level(id, active, confidence)` and `out.edge(id, confidence, **data)` are
the whole output surface. `out.is_active(id)` reads what earlier recognizers
decided this frame — recognizers run in a fixed order, which is how jump
detection knows to mute itself around a squat.

### What `BodyFeatures` gives you

| field | meaning |
|---|---|
| `P` | `(33, 2)` normalised: origin at hip centre, **+x subject's right, +y up**, 1 unit ≈ one torso |
| `W` | `(33, 3)` world landmarks in metres — use for joint angles, they are view-independent |
| `vis` | per-landmark visibility; `f.visible(*indices)` to gate |
| `scale` | the divisor that produced `P`, in frame-height units |
| `torso_lean` | degrees from vertical, positive to the subject's right |
| `nose_above_shoulders`, `shoulder_center_height`, `elevation` | heights in body units |
| `left_hand_rise` / `right_hand_rise` | wrist height above the nose |
| `left_elbow_angle`, `left_knee_angle`, … | joint angles in degrees, 3D where available |
| `left_thigh_vertical`, `thigh_length`, `arm_length` | self-measured limb geometry |
| `hands` | `HandFeatures(side, curl, pinch, score, center, points)` |

Unavailable measurements are `NaN`, never a guess, and `NaN` propagates through
the gates as "no evidence".

### A different pose model

Implement `LandmarkProvider`, produce the 33-point BlazePose layout (see
`motionsense.landmarks`), and every recognizer works unchanged:

```python
engine = MotionEngine(provider=MyProvider())
```

---

## Accuracy

The recognizers are built on a normalised body frame rather than raw landmark
coordinates. Three corrections get it there, and each fixes a class of error
that raw coordinates produce silently:

**Aspect correction.** Detector output is normalised to `[0, 1]` on each axis
*independently*, so on a 16:9 frame one x-unit is 1.78 times longer than one
y-unit. Any distance, angle or ratio mixing the two is wrong — a shoulder width
measured horizontally comes out 78% too large relative to a torso measured
vertically, and it changes again on a 4:3 camera.

**Scale normalisation.** Dividing by a body scale removes distance from the
camera. The scale is the RMS radius of the four torso corners about their
centroid — the scale term of a Procrustes fit. Shoulder width alone collapses
when the subject turns sideways and torso length alone collapses when they lean
toward the camera; pooling four points degrades gracefully instead of vanishing.
It is then smoothed hard, because a person's size does not change, which removes
scale jitter from every derived feature at once.

**Origin shift.** Translating to the hip centre removes where in the frame the
subject stands, so walking across the room changes nothing.

Rotation is deliberately *not* normalised: the image vertical is the gravity
reference that makes "up", "lean" and "head down" mean anything.

The result is verified, not asserted — `test_features.py` checks that a subject
at 2× distance, shifted across the frame, or seen through a different aspect
ratio produces the same feature values to within 1e-3.

### Specific improvements over a threshold-on-raw-coordinates approach

| Area | Naive approach | Here | Why it matters |
|---|---|---|---|
| Jump | per-frame `Δy` compared to a constant | least-squares velocity in body units/second | A per-frame delta scales with frame rate: the same jump reads twice as large at 60 FPS as at 30. Verified detected at 20, 30 and 60 FPS with one threshold. |
| Derivatives | two-point difference | least-squares fit over a window | Two-point error variance is `2σ²/Δt²` — larger than the signal for anything but violent motion. Measured >4× lower error. |
| Squat | `(knee_y − hip_y)/(ankle_y − hip_y)` | 3D knee angle, falling back to thigh verticality | A frontal squat *foreshortens* the thigh instead of rotating it — 2D sees a straight leg at 179° while the real joint is at 108°. Also survives the very common framing that cuts off the feet. |
| Fist | fingertip lower on screen than its knuckle | radial `tip/PIP` distance ratio from the wrist | The screen test only works for an upright hand and inverts entirely when the hand points down. |
| Waving | count per-frame direction changes | detrend, amplitude-gate, Schmitt-triggered zero crossings, frequency band | Raw sign-counting reads jitter on a still hand as a fast wave, and misses a real wave performed while the arm drifts. |
| Level activities | single threshold | hysteresis band + minimum dwell | One threshold flickers many times a second whenever the measurement sits near it — which is exactly where a held pose lands. Downstream that is a key pressed and released 30 times a second. |
| Occlusion | use whatever the detector reports | visibility gate, `NaN` for unknown | Detectors extrapolate landmarks that are out of frame. Trusting those is the largest single source of phantom activations. |
| Smoothing | fixed-α EMA, or none | 1-Euro filter | A fixed low-pass forces a choice between jitter and lag. 1-Euro adapts its cutoff to speed: measured ~3× less lag than a fixed cutoff at the same steadiness. |
| Tracking loss | drop everything, or latch forever | grace window, then release | Detection misses the odd frame with a person standing still. Ending on the first miss makes every hold stutter; never ending leaves a key down after the user walks away. |
| Arms crossed | wrists past the body midline, elbows bent | wrists swapped sides (`left.x − right.x`), elbows advisory | Folding your arms *hides your wrists*, so the strict form fails on people holding the pose correctly. A relative crossing test also survives an off-centre subject and an asymmetric fold. |
| Left/right | trust the detector's labels | swap handedness for non-mirrored input | MediaPipe assigns handedness *assuming a mirrored image*. On a raw camera feed every label is the wrong hand. |

### Where it is still limited

Honest boundaries, all inherent to one RGB camera:

- Vertical swipes overlap with arm raises (see the catalog note). Off by default.
- A squat with the feet out of frame **and** the subject facing straight on has
  no reliable cue — the knee angle needs ankles, and thigh verticality needs
  rotation the frontal view does not show.
- Depth is estimated, not measured. Features that would need real depth
  (reaching toward the camera) are not offered rather than offered badly.
- One subject at a time. The first detected body wins.
- Thresholds are population averages until you [calibrate](#calibration).

---

### Debugging a pose that isn't firing

```bash
python examples/07_diagnose.py arms_crossed
```

Every processed frame carries `FrameResult.levels` — every level activity that
was considered, **including the ones that did not fire**, with the score it was
judged on:

```python
@engine.on_frame
def why(result):
    active, confidence, data = result.levels["arms_crossed"]
    print(active, data["score"])
```

There are two distinct failures and they need opposite fixes:

| score | meaning | fix |
|---|---|---|
| negative | the geometry was measured and rejected | go further into the pose, or loosen the threshold. The value is in units of the hysteresis band, so −0.4 is 40% of a band short |
| `NaN` | it could not be measured — a required landmark is not visible | fix framing or lighting. Loosening a threshold cannot help; nothing was compared to it |

From outside they look identical — the pose just never fires — which is why the
score is surfaced rather than kept internal.

## Performance

Measured on this machine (Windows, CPU, 640×480, MediaPipe Tasks):

| | per frame |
|---|---|
| SDK: normalisation, filtering, all recognizers, dispatch | **0.09 ms** (0.13 ms with two hands) |
| MediaPipe inference, pose only | ~7.9 ms |
| MediaPipe inference, pose + hands | ~15.0 ms |

The shape of that is the important part: **inference dominates by ~90×**. So the
SDK's own maths is effectively free, and the useful optimisations are the ones
that avoid inference work rather than the ones that shave arithmetic. The
mathematical work in this package buys *accuracy*, not throughput — it is worth
being clear about which is which.

What actually moves the number:

1. **`enable_hands="auto"` (default)** — the hand model costs about as much as
   the pose model, and most control schemes never use a finger state. Measured
   1.9× fewer milliseconds per frame when nothing subscribes to a hand activity.
2. **Latest-frame capture** — a dedicated grab thread drains the driver queue
   into a one-slot mailbox. `VideoCapture.read()` returns the *oldest* queued
   frame, so a consumer slower than the camera falls further behind every frame
   and the lag compounds. This is a latency fix, not a throughput one, and no
   amount of detector tuning substitutes for it: the detector is looking at the
   past.
3. **`inference_width`** — the model resizes to its own input size anyway, so a
   1080p frame only pays for a bigger resize, twice when hands are on.
4. **Mirror coordinates, not pixels** — flipping a 720p frame touches ~2.7 MB;
   flipping the landmark x column touches 33 floats for identical geometry.
5. **Pipelined threads** — capture and inference run separately, so the two
   overlap instead of serialising.
6. **`preset="fast"`** — the lite model, when latency matters more than the last
   few percent of landmark precision. For interactive control it usually does.

Absolute inference numbers depend on hardware and on whether a subject is in
frame; measure on your target. The 0.09 ms SDK figure does not — it is the same
work regardless of image content.

---

## Calibration

Optional. Thresholds work out of the box on population averages; calibration
makes them personal.

```python
engine.calibrate(seconds=2.0)          # subject stands neutral, facing the camera
engine.calibration.to_dict()           # persist it
engine.calibration = Calibration.from_dict(saved)
```

It matters most where body proportion and camera angle both enter — how far the
nose normally sits above the shoulder line differs by a factor of two between a
tall subject with a high camera and a short one with a low camera, enough to make
a fixed "head down" threshold either fire constantly or never. Samples are
reduced with the **median**, because a few seconds of a human standing still
reliably contains frames where a landmark jumps.

---

## Testing

```bash
pytest                      # 140 tests, ~5 s, no camera needed
```

Recognizers are driven by a parametric 3D stick figure
([`tests/synthetic.py`](tests/synthetic.py)) orthographically projected to
landmarks. Being genuinely 3D matters: a frontal squat foreshortens the thigh
rather than rotating it, so a 2D-only fixture could not tell a correct squat
detector from a broken one.

That makes every parameter independently variable, which is what lets the suite
assert things like "the same jump is detected at 20, 30 and 60 FPS" and "the same
gesture is detected at 0.7×, 1.0× and 1.8× distance" — properties that are hard
to test with recorded video and easy to break without noticing.

---

## How the MotionKey app uses it

The MotionKey app in the parent directory runs on this SDK. It is a useful
worked example of embedding the engine in a GUI, and of the one thing that needs
care: **who calls your listeners.**

`core/camera_worker.py` is the whole integration — about 100 lines, and nothing
in it is about detection:

```python
class CameraWorker(QThread):
    def run(self):
        config = EngineConfig(
            preset="fast",
            deliver_frames=True,
            enable_hands=needs_hand_model(mapped),          # only if a finger key is bound
            tuning=Tuning(vertical_swipes=any_vertical_swipe_mapped),
        )
        engine = MotionEngine(config)
        KeyBindings(engine, self.mappings)                   # the dict straight from SQLite

        engine.on_frame(self._on_frame)                      # preview + active list
        engine.on_error(lambda exc: self.error.emit(str(exc)))

        engine.run(CameraSource(self.camera_index))          # blocks on this QThread
```

Running the engine with `run()` on the `QThread`, rather than `start()`, is
deliberate: the engine's listeners then fire on a thread Qt already knows about,
so the signals reaching the GUI are ordinary queued connections. Calling
`start()` here would work too, but the callbacks would arrive on a thread Qt
never created.

Two more details worth copying:

- `on_frame` supplies both the preview image *and* `result.active`, so the app
  needs no wildcard subscription. That matters, because a wildcard would defeat
  `enable_hands="auto"` and make the hand model run permanently.
- The overlay from `motionsense.draw.render` is BGR, and `QImage.Format_BGR888`
  reads it directly — no per-frame colour conversion.

The migration needed no database change: all 19 of the app's original activity
ids exist here unchanged, with the same level/edge triggers, so saved key
configurations kept working as-is.

What changed behaviourally: detection became scale- and frame-rate invariant,
level activities are debounced so mapped keys stop chattering, key taps no
longer block the detection thread for 50 ms, the hand model runs only when a
finger activity is mapped, and six activities are new (`swipe_*`, `*_pinch`).

---

## License

[MIT](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for release history.
