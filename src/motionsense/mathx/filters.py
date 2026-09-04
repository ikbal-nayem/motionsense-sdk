"""Time-aware smoothing filters.

Every filter here is parameterised in *continuous time* (cutoff frequencies and
time constants), not in frames. A filter tuned at 30 FPS therefore behaves
identically at 15 or 60 FPS, which matters because webcam frame rates drift with
lighting and CPU load. Filters expressed as a fixed per-frame ``alpha`` silently
change their cutoff whenever the frame rate moves.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["ExpSmoother", "OneEuroFilter", "alpha_for_cutoff", "alpha_for_tau"]

# A stall (window drag, GC pause, camera hiccup) can produce a huge dt. Clamping
# keeps one bad interval from snapping the filter state to the raw sample.
_MAX_DT = 0.25


def alpha_for_cutoff(cutoff, dt: float):
    """Smoothing factor of a first-order low-pass with the given -3 dB cutoff (Hz).

    Derived from the RC equivalent: ``tau = 1 / (2*pi*fc)`` and
    ``alpha = dt / (dt + tau)``. Accepts scalars or arrays for ``cutoff``.
    """
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return dt / (dt + tau)


def alpha_for_tau(tau: float, dt: float) -> float:
    """Smoothing factor for an exponential decay with time constant ``tau`` seconds.

    Uses the exact discretisation ``1 - exp(-dt/tau)`` rather than the ``dt/tau``
    first-order approximation, so it stays correct (and stable) even when ``dt``
    approaches or exceeds ``tau`` after a dropped frame.
    """
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


class ExpSmoother:
    """Scalar exponential smoother with a time constant in seconds."""

    __slots__ = ("tau", "_value", "_t")

    def __init__(self, tau: float, initial: float | None = None):
        self.tau = tau
        self._value = initial
        self._t: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value
        self._t = None

    def update(self, x: float, t: float) -> float:
        if self._value is None or self._t is None:
            self._value = x
            self._t = t
            return x
        dt = min(max(t - self._t, 1e-6), _MAX_DT)
        self._t = t
        a = alpha_for_tau(self.tau, dt)
        self._value += a * (x - self._value)
        return self._value


class OneEuroFilter:
    """Vectorised 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

    A plain low-pass forces a choice between jitter (high cutoff) and lag (low
    cutoff). The 1-Euro filter removes that trade-off by making the cutoff a
    function of the signal's own speed::

        fc = min_cutoff + beta * |x_dot_hat|

    Slow movement -> low cutoff -> jitter is suppressed while the hand is held
    still. Fast movement -> high cutoff -> the filter tracks the motion with
    almost no lag. For gesture control this is the difference between a stable
    "hold" and a responsive "swipe", and it is why smoothing here does not cost
    reaction time.

    Operates on whole landmark arrays at once (e.g. ``(33, 2)``); the adaptive
    cutoff is computed per element, so a still torso keeps being smoothed hard
    while a moving wrist is not.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x", "_dx", "_t")

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.035, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        """Drop filter state. Call when tracking is lost, so the next detection
        is adopted immediately instead of being smoothed in from a stale pose."""
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x is None or self._t is None:
            self._x = x.astype(np.float64, copy=True)
            self._dx = np.zeros_like(self._x)
            self._t = t
            return self._x.copy()

        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        dt = min(dt, _MAX_DT)
        self._t = t

        # Derivative estimate, itself low-passed at a fixed cutoff so that noise
        # in the raw difference does not drive the adaptive cutoff.
        dx = (x - self._x) / dt
        a_d = alpha_for_cutoff(self.d_cutoff, dt)
        self._dx += a_d * (dx - self._dx)

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = alpha_for_cutoff(cutoff, dt)
        self._x += a * (x - self._x)
        return self._x.copy()
