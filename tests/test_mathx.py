"""Unit tests for the signal-processing layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motionsense.mathx import (
    ExpSmoother,
    OneEuroFilter,
    OscillationDetector,
    SchmittGate,
    WindowedSlope,
    angle_at,
    angle_at3,
    procrustes_scale,
    signed_angle_from_up,
)


# -- geometry ----------------------------------------------------------------
def test_angle_at_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angle_at(a, b, c) == pytest.approx(math.pi / 2)


def test_angle_at_is_stable_when_straight():
    """acos(dot/|u||v|) can round outside [-1, 1] and return NaN for collinear
    points; the atan2 form must not."""
    a = np.array([0.0, 2.0])
    b = np.array([0.0, 1.0])
    c = np.array([0.0, 0.0])
    angle = angle_at(a, b, c)
    assert not math.isnan(angle)
    assert angle == pytest.approx(math.pi, abs=1e-9)


def test_angle_at3_matches_2d_when_planar():
    a, b, c = np.array([0, 1.0, 0]), np.array([0, 0, 0]), np.array([1.0, 0, 0])
    assert angle_at3(a, b, c) == pytest.approx(math.pi / 2)


def test_angle_at3_sees_bend_that_2d_projection_hides():
    """A limb bending directly toward the camera keeps a near-straight 2D
    silhouette. This is the whole reason joint angles use world landmarks."""
    hip = np.array([0.0, 1.0, 0.0])
    knee = np.array([0.0, 0.0, 0.6])
    ankle = np.array([0.0, -1.0, 0.0])

    in_3d = math.degrees(angle_at3(hip, knee, ankle))
    in_2d = math.degrees(angle_at(hip[:2], knee[:2], ankle[:2]))

    assert in_2d == pytest.approx(180.0, abs=1.0)  # projection says "straight"
    assert in_3d < 120.0  # reality says "bent"


def test_signed_angle_from_up():
    assert math.degrees(signed_angle_from_up(np.array([0.0, 1.0]))) == pytest.approx(0.0)
    assert math.degrees(signed_angle_from_up(np.array([1.0, 1.0]))) == pytest.approx(45.0)
    assert math.degrees(signed_angle_from_up(np.array([-1.0, 1.0]))) == pytest.approx(-45.0)


def test_procrustes_scale_is_rotation_invariant():
    points = np.array([[-1.0, -2.0], [1.0, -2.0], [-1.0, 2.0], [1.0, 2.0]])
    base = procrustes_scale(points)
    theta = 0.7
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    assert procrustes_scale(points @ rotation.T) == pytest.approx(base)


def test_procrustes_scale_is_translation_invariant():
    points = np.array([[-1.0, -2.0], [1.0, -2.0], [-1.0, 2.0], [1.0, 2.0]])
    assert procrustes_scale(points + 17.0) == pytest.approx(procrustes_scale(points))


# -- filters -----------------------------------------------------------------
def test_one_euro_suppresses_jitter_on_a_still_signal():
    rng = np.random.default_rng(0)
    truth = np.array([0.5, 0.5])
    filt = OneEuroFilter(min_cutoff=0.8, beta=0.01)

    residuals = []
    for i in range(120):
        noisy = truth + rng.normal(0, 0.01, 2)
        out = filt(noisy, i / 30.0)
        if i > 30:
            residuals.append(np.abs(out - truth).mean())

    assert np.mean(residuals) < 0.004  # >2.5x quieter than the input noise


def test_one_euro_costs_less_lag_than_a_plain_low_pass():
    """The whole reason for the speed term: at the same minimum cutoff, tracking
    a moving target must lag far less than a fixed-cutoff filter would."""

    def steady_state_lag(beta: float) -> float:
        filt = OneEuroFilter(min_cutoff=0.8, beta=beta)
        lag = 0.0
        for i in range(90):
            t = i / 30.0
            truth = np.array([t * 2.0, 0.0])
            out = filt(truth, t)
            lag = abs(out[0] - truth[0])
        return lag

    fixed = steady_state_lag(0.0)
    adaptive = steady_state_lag(0.4)
    assert adaptive < 0.35 * fixed


def test_one_euro_reset_adopts_the_next_sample():
    filt = OneEuroFilter()
    filt(np.array([0.0, 0.0]), 0.0)
    filt(np.array([0.0, 0.0]), 0.1)
    filt.reset()
    out = filt(np.array([9.0, 9.0]), 0.2)
    assert out == pytest.approx(np.array([9.0, 9.0]))


def test_exp_smoother_is_framerate_independent():
    """Same time constant, same settling, regardless of sample rate."""
    results = []
    for fps in (15.0, 60.0):
        smoother = ExpSmoother(tau=0.5, initial=0.0)
        steps = int(2.0 * fps)
        for i in range(steps):
            value = smoother.update(1.0, i / fps)
        results.append(value)
    assert results[0] == pytest.approx(results[1], abs=0.02)


# -- slope --------------------------------------------------------------------
def test_windowed_slope_recovers_a_known_rate():
    slope = WindowedSlope(window=0.3, min_samples=3)
    for i in range(30):
        t = i / 30.0
        slope.push(t, 2.5 * t + 1.0)
    assert slope.slope() == pytest.approx(2.5, rel=1e-6)


def test_windowed_slope_is_framerate_independent():
    """A per-frame difference would report half the value at double the rate.
    This is what makes one jump threshold work across frame rates."""
    values = []
    for fps in (15.0, 30.0, 60.0):
        slope = WindowedSlope(window=0.25, min_samples=3)
        for i in range(int(fps)):
            t = i / fps
            slope.push(t, 3.0 * t)
        values.append(slope.slope())
    assert all(v == pytest.approx(3.0, rel=1e-6) for v in values)


def test_windowed_slope_beats_two_point_difference_under_noise():
    rng = np.random.default_rng(7)
    slope = WindowedSlope(window=0.33, min_samples=3)
    truth = 2.0

    fitted, two_point = [], []
    previous = None
    for i in range(200):
        t = i / 30.0
        y = truth * t + rng.normal(0, 0.01)
        slope.push(t, y)
        if previous is not None:
            two_point.append(abs((y - previous[1]) / (t - previous[0]) - truth))
        previous = (t, y)
        estimate = slope.slope()
        if i > 15 and estimate is not None:
            fitted.append(abs(estimate - truth))

    assert np.mean(fitted) < 0.25 * np.mean(two_point)


def test_windowed_slope_is_precise_at_large_timestamps():
    """Rebasing the window guards the n*Stt - St^2 cancellation. Without it a
    long-running session slowly loses derivative precision."""
    slope = WindowedSlope(window=0.2, min_samples=3)
    base = 86_400.0
    for i in range(30):
        t = base + i / 60.0
        slope.push(t, 1.75 * t)
    assert slope.slope() == pytest.approx(1.75, rel=1e-6)


def test_directness_separates_sweep_from_oscillation():
    sweep = WindowedSlope(window=1.0)
    wobble = WindowedSlope(window=1.0)
    for i in range(30):
        t = i / 30.0
        sweep.push(t, t)
        wobble.push(t, math.sin(2 * math.pi * 3 * t))
    assert sweep.directness() > 0.95
    assert wobble.directness() < 0.4


# -- gates ---------------------------------------------------------------------
def test_schmitt_gate_rejects_threshold_chatter():
    """A single threshold would toggle on every sample here."""
    gate = SchmittGate(0.6, 0.4)
    toggles = 0
    state = False
    for i in range(200):
        value = 0.5 + (0.02 if i % 2 else -0.02)
        new = gate.update(value, i / 30.0)
        toggles += new != state
        state = new
    assert toggles == 0


def test_schmitt_gate_falling_polarity_is_inferred():
    gate = SchmittGate(0.3, 0.5)  # on when the value drops below 0.3
    assert gate.update(0.9, 0.0) is False
    assert gate.update(0.2, 0.1) is True
    assert gate.update(0.4, 0.2) is True  # inside the band: holds
    assert gate.update(0.6, 0.3) is False


def test_schmitt_gate_debounces_brief_spikes():
    gate = SchmittGate(0.6, 0.4, min_on=0.1)
    assert gate.update(0.9, 0.00) is False  # candidate, not yet held long enough
    assert gate.update(0.9, 0.05) is False
    assert gate.update(0.9, 0.20) is True


def test_schmitt_gate_releases_on_sustained_missing_data():
    gate = SchmittGate(0.6, 0.4, min_off=0.05)
    gate.update(0.9, 0.0)
    assert gate.state is True
    gate.update(float("nan"), 0.10)  # starts the release timer
    gate.update(float("nan"), 0.20)  # min_off elapsed
    assert gate.state is False


def test_schmitt_gate_tolerates_a_single_dropout():
    gate = SchmittGate(0.6, 0.4, min_off=0.15)
    gate.update(0.9, 0.0)
    gate.update(float("nan"), 1 / 30)
    assert gate.state is True  # one missed frame must not drop a held pose
    gate.update(0.9, 2 / 30)
    assert gate.state is True


# -- oscillation ------------------------------------------------------------------
def test_oscillation_detects_a_wave():
    detector = OscillationDetector(min_amplitude=0.05, cooldown=0.0)
    result = None
    for i in range(60):
        t = i / 30.0
        result = detector.update(t, 0.3 * math.sin(2 * math.pi * 3.0 * t)) or result
    assert result is not None
    assert result.frequency == pytest.approx(3.0, abs=0.9)


def test_oscillation_ignores_noise_on_a_still_hand():
    rng = np.random.default_rng(3)
    detector = OscillationDetector(min_amplitude=0.05)
    for i in range(120):
        assert detector.update(i / 30.0, rng.normal(0, 0.004)) is None


def test_oscillation_survives_drift():
    """Detrending is what lets a wave register while the whole arm travels."""
    detector = OscillationDetector(min_amplitude=0.05, cooldown=0.0)
    fired = False
    for i in range(60):
        t = i / 30.0
        fired = bool(detector.update(t, 0.9 * t + 0.25 * math.sin(2 * math.pi * 3.0 * t))) or fired
    assert fired


def test_oscillation_ignores_a_single_sweep():
    detector = OscillationDetector(min_amplitude=0.05)
    for i in range(40):
        t = i / 30.0
        assert detector.update(t, min(t, 0.6)) is None


