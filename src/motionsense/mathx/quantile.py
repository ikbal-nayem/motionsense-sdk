"""Constant-memory online quantile estimation."""

from __future__ import annotations

import math

from .filters import alpha_for_tau

__all__ = ["RunningQuantile"]


class RunningQuantile:
    """Tracks a quantile of a stream by stochastic descent on the pinball loss.

    Used for self-calibrating references, such as "how tall does this person
    stand when they are not crouching". A running *mean* is wrong for that job:
    the signal is bimodal (standing vs. crouching) and the mean lands between the
    modes, so the threshold drifts toward whichever state the user held longest.
    A high quantile stays pinned to the standing mode.

    The estimator minimises the pinball loss
    ``rho_tau(u) = u * (tau - 1{u < 0})`` whose subgradient with respect to the
    estimate ``q`` is ``P(x < q) - tau``. One sample gives the unbiased estimate
    ``1{x < q} - tau``, so the update is::

        q += step * (tau - 1{x < q})

    Step size is the interesting part. A fixed step is wrong in general because
    it has the units of ``x``: too small and the estimate never reaches the
    distribution, too large and it rattles. Here the step is
    ``alpha(tau_seconds, dt) * scale``, where ``scale`` is a running mean
    absolute deviation. That makes it dimensionless in the signal's own units and
    frame-rate independent -- the estimator converges over roughly
    ``tau_seconds`` regardless of FPS -- and it needs O(1) memory, unlike a
    sliding-window quantile that has to retain every sample.
    """

    __slots__ = ("tau", "time_constant", "min_scale", "_q", "_scale", "_t")

    def __init__(
        self,
        tau: float = 0.5,
        time_constant: float = 12.0,
        min_scale: float = 1e-3,
        initial: float | None = None,
    ):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.tau = float(tau)
        self.time_constant = float(time_constant)
        self.min_scale = float(min_scale)
        self._q = initial
        self._scale = min_scale
        self._t: float | None = None

    @property
    def value(self) -> float | None:
        return self._q

    def reset(self, value: float | None = None) -> None:
        self._q = value
        self._scale = self.min_scale
        self._t = None

    def update(self, x: float, t: float) -> float:
        if math.isnan(x):
            return self._q if self._q is not None else float("nan")

        if self._q is None or self._t is None:
            self._q = x
            self._t = t
            return x

        dt = min(max(t - self._t, 1e-6), 0.25)
        self._t = t
        a = alpha_for_tau(self.time_constant, dt)

        deviation = abs(x - self._q)
        self._scale += a * (deviation - self._scale)
        step = a * max(self._scale, self.min_scale)

        gradient = self.tau - (1.0 if x < self._q else 0.0)
        self._q += step * gradient
        return self._q
