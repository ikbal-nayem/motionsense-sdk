"""Dynamic gestures: jump, wave, swipe.

These are the activities where the original per-frame-delta approach was both
frame-rate dependent and noise dominated. Everything here works from
least-squares velocities in body units per second, so a threshold means one
physical speed regardless of how fast the camera happens to be running.
"""

from __future__ import annotations

import math

from ..catalog import get
from ..config import Tuning
from ..features import BodyFeatures
from ..landmarks import Pose
from ..mathx import OscillationDetector, WindowedSlope
from .base import Recognizer, Sink

__all__ = ["MotionRecognizer"]

_IDS = (
    "jump",
    "wave_left_hand",
    "wave_right_hand",
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
)

_GROUND = 0
_RISING = 1

#: Standing up from a squat produces the same upward torso velocity as a jump.
#: Posture is decided first, so jumps are simply muted around a squat release.
_SQUAT_BLACKOUT = 0.55

_SIDES = ("left", "right")
_WRIST = {"left": Pose.LEFT_WRIST, "right": Pose.RIGHT_WRIST}


class MotionRecognizer(Recognizer):
    """Ballistic and oscillatory motion of the body and hands."""

    activities = tuple(d for d in (get(i) for i in _IDS) if d is not None)

    def __init__(self):
        self._tuning = Tuning()
        self._elevation = WindowedSlope(window=0.15, min_samples=3)
        self._state = _GROUND
        self._launch_elevation = 0.0
        self._launch_t = 0.0
        self._peak_elevation = 0.0
        self._peak_speed = 0.0
        self._last_jump = float("-inf")
        self._squat_until = float("-inf")

        self._wave: dict[str, OscillationDetector] = {}
        self._last_wave: dict[str, float] = {s: float("-inf") for s in _SIDES}
        self._swipe_x: dict[str, WindowedSlope] = {}
        self._swipe_y: dict[str, WindowedSlope] = {}
        self._last_swipe = float("-inf")

    def configure(self, tuning: Tuning) -> None:
        self._tuning = tuning
        self._wave = {
            side: OscillationDetector(
                window=tuning.wave_window,
                min_half_cycles=tuning.wave_min_half_cycles,
                min_amplitude=tuning.wave_min_amplitude,
                freq_band=tuning.wave_freq_band,
                cooldown=tuning.wave_cooldown,
            )
            for side in _SIDES
        }
        self._swipe_x = {s: WindowedSlope(window=0.22, min_samples=3) for s in _SIDES}
        self._swipe_y = {s: WindowedSlope(window=0.22, min_samples=3) for s in _SIDES}

    def reset(self) -> None:
        self._elevation.clear()
        self._state = _GROUND
        self._last_jump = float("-inf")
        self._squat_until = float("-inf")
        self._last_swipe = float("-inf")
        for side in _SIDES:
            self._last_wave[side] = float("-inf")
            if side in self._wave:
                self._wave[side].clear()
                self._swipe_x[side].clear()
                self._swipe_y[side].clear()

    def update(self, f: BodyFeatures, out: Sink) -> None:
        if not self._wave:
            self.configure(self._tuning)
        if out.is_active("squat"):
            self._squat_until = f.t + _SQUAT_BLACKOUT

        self._update_jump(f, out)
        for side in _SIDES:
            self._update_hand_motion(f, out, side)

    # -- jump --------------------------------------------------------------------
    def _update_jump(self, f: BodyFeatures, out: Sink) -> None:
        tn = self._tuning
        elevation = f.elevation
        if math.isnan(elevation):
            return
        self._elevation.push(f.t, elevation)

        velocity = self._elevation.slope(f.t)
        if velocity is None:
            return

        if self._state == _GROUND:
            if velocity >= tn.jump_takeoff_speed and f.t >= self._squat_until:
                self._state = _RISING
                self._launch_elevation = elevation
                self._launch_t = f.t
                self._peak_elevation = elevation
                self._peak_speed = velocity
            return

        # Rising: wait for the descent that proves this was a launch and not a
        # slow stand-up, then check the flight was tall enough and short enough.
        self._peak_elevation = max(self._peak_elevation, elevation)
        self._peak_speed = max(self._peak_speed, velocity)
        airtime = f.t - self._launch_t

        if airtime > tn.jump_max_airtime:
            self._state = _GROUND
            return

        if velocity <= tn.jump_landing_speed:
            rise = self._peak_elevation - self._launch_elevation
            self._state = _GROUND
            if (
                rise >= tn.jump_min_rise
                and f.t - self._last_jump >= tn.jump_cooldown
                and f.t >= self._squat_until
            ):
                self._last_jump = f.t
                confidence = min(1.0, 0.5 * rise / tn.jump_min_rise + 0.5 * self._peak_speed / tn.jump_takeoff_speed)
                out.edge(
                    "jump",
                    confidence,
                    rise=round(rise, 4),
                    airtime=round(airtime, 3),
                    peak_speed=round(self._peak_speed, 3),
                )

    # -- wave and swipe ----------------------------------------------------------
    def _update_hand_motion(self, f: BodyFeatures, out: Sink, side: str) -> None:
        tn = self._tuning
        index = _WRIST[side]
        wrist = f.P[index]
        visible = f.visible(index)

        # Waving is only meaningful with the hand raised; requiring that also
        # keeps ordinary gesticulation while talking from registering.
        raised = visible and float(wrist[1]) > float(f.shoulder_center[1]) - 0.05
        if raised:
            result = self._wave[side].update(f.t, float(wrist[0]))
            if result is not None:
                self._last_wave[side] = f.t
                out.edge(
                    f"wave_{side}_hand",
                    result.confidence,
                    frequency=round(result.frequency, 2),
                    amplitude=round(result.amplitude, 3),
                )
        else:
            self._wave[side].clear()

        if not visible or float(wrist[1]) < 0.1:
            self._swipe_x[side].clear()
            self._swipe_y[side].clear()
            return
        self._swipe_x[side].push(f.t, float(wrist[0]))
        self._swipe_y[side].push(f.t, float(wrist[1]))

        if f.t - self._last_swipe < tn.swipe_cooldown:
            return
        if f.t - self._last_wave[side] < tn.swipe_wave_suppression:
            return

        vx = self._swipe_x[side].slope(f.t)
        vy = self._swipe_y[side].slope(f.t)
        if vx is None or vy is None:
            return

        horizontal = abs(vx) >= abs(vy)
        if not horizontal and not tn.vertical_swipes:
            return
        series = self._swipe_x[side] if horizontal else self._swipe_y[side]
        speed = abs(vx) if horizontal else abs(vy)
        other = abs(vy) if horizontal else abs(vx)

        if speed < tn.swipe_speed:
            return
        if speed < tn.swipe_axis_ratio * other:
            return
        if series.span(f.t) < tn.swipe_min_travel:
            return
        # A wave's outbound stroke is as fast and as wide as a swipe; what it is
        # not is one-way. Checked twice: over the swipe window, and over a longer
        # one that outlasts a wave period so an oscillation cancels itself out.
        if series.directness(f.t) < tn.swipe_directness:
            return
        if series.directness(f.t, window=tn.swipe_confirm_window) < tn.swipe_confirm_directness:
            return

        if horizontal:
            activity = "swipe_right" if vx > 0 else "swipe_left"
        else:
            activity = "swipe_up" if vy > 0 else "swipe_down"

        self._last_swipe = f.t
        self._swipe_x[side].clear()
        self._swipe_y[side].clear()
        out.edge(
            activity,
            min(1.0, speed / (tn.swipe_speed * 1.6)),
            hand=side,
            speed=round(speed, 3),
            travel=round(series.span(f.t), 3),
        )
