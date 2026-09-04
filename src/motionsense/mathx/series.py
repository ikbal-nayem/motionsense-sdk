"""Fixed-capacity time series with least-squares differentiation."""

from __future__ import annotations

import numpy as np

__all__ = ["RingSeries", "WindowedSlope"]


class RingSeries:
    """Circular buffer of ``(t, value)`` samples with no per-frame allocation.

    A ``deque`` plus ``np.fromiter`` would be simpler, but this sits in the
    per-frame path for several signals at once and a preallocated ring keeps the
    steady state allocation-free apart from the small window copy handed back by
    :meth:`window`.
    """

    __slots__ = ("_t", "_v", "_cap", "_n", "_head")

    def __init__(self, capacity: int = 96):
        self._cap = int(capacity)
        self._t = np.zeros(self._cap, dtype=np.float64)
        self._v = np.zeros(self._cap, dtype=np.float64)
        self._n = 0
        self._head = 0  # index of the next slot to write

    def __len__(self) -> int:
        return self._n

    def clear(self) -> None:
        self._n = 0
        self._head = 0

    def push(self, t: float, value: float) -> None:
        self._t[self._head] = t
        self._v[self._head] = value
        self._head = (self._head + 1) % self._cap
        if self._n < self._cap:
            self._n += 1

    @property
    def last(self) -> tuple[float, float] | None:
        if self._n == 0:
            return None
        i = (self._head - 1) % self._cap
        return float(self._t[i]), float(self._v[i])

    def recent(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """The last ``count`` samples, oldest first, times rebased to zero.

        Used as the low-frame-rate fallback for :class:`WindowedSlope`: a fixed
        time window can hold fewer samples than a fit needs when the camera is
        slow, and a derivative computed over a slightly longer span is far better
        than no derivative at all.
        """
        n = min(count, self._n)
        if n == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        indices = (self._head - n + np.arange(n)) % self._cap
        t = self._t[indices]
        return t - t[0], self._v[indices]

    def window(self, seconds: float, now: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Samples from the last ``seconds``, oldest first, as contiguous arrays.

        Times are rebased so the oldest sample in the window sits at ``t = 0``.
        That rebasing is not cosmetic: the regression below forms
        ``n*Stt - St^2``, and with raw epoch- or session-scale timestamps that
        subtraction cancels ~10 significant digits before the answer appears.
        Rebasing keeps every term the same order of magnitude as the result.
        """
        if self._n == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        start = (self._head - self._n) % self._cap
        if start + self._n <= self._cap:
            t = self._t[start : start + self._n]
            v = self._v[start : start + self._n]
        else:
            tail = self._cap - start
            t = np.concatenate((self._t[start:], self._t[: self._n - tail]))
            v = np.concatenate((self._v[start:], self._v[: self._n - tail]))

        ref = t[-1] if now is None else now
        keep = t >= ref - seconds
        t = t[keep]
        v = v[keep]
        if t.size == 0:
            return t, v
        return t - t[0], v.copy()


class WindowedSlope:
    """Least-squares time derivative over a sliding window.

    A two-point difference ``(y1 - y0) / dt`` is an unbiased derivative estimate
    but it passes noise straight through: its error variance is
    ``2*sigma^2 / dt^2``, which for a 30 FPS webcam and landmark noise of a few
    millimetres is larger than the signal for anything but a violent movement.
    Fitting a line to the last ``window`` seconds instead gives the
    minimum-variance unbiased slope under white noise, cutting the error
    variance by roughly ``12 / (n * (n^2 - 1)) * (n-1)^2`` relative to the
    two-point estimate -- about an 8x reduction in standard deviation for a
    10-sample window -- while still responding within the window length.

    This is what makes jump detection frame-rate independent: the result is in
    units per *second*, so the same threshold holds at 15 or 60 FPS. Comparing
    raw per-frame deltas (as a naive implementation does) makes the effective
    threshold scale with the frame rate.
    """

    __slots__ = ("window", "min_samples", "_series")

    def __init__(self, window: float = 0.16, min_samples: int = 4, capacity: int = 96):
        self.window = float(window)
        self.min_samples = int(min_samples)
        self._series = RingSeries(capacity)

    def clear(self) -> None:
        self._series.clear()

    def push(self, t: float, value: float) -> None:
        self._series.push(t, value)

    @property
    def latest(self) -> float | None:
        last = self._series.last
        return None if last is None else last[1]

    def slope(self, now: float | None = None) -> float | None:
        """Rate of change in value-units per second, or ``None`` if underdetermined.

        At low frame rates the time window can hold fewer samples than the fit
        needs -- a 0.15 s window sees three frames at 20 FPS. Rather than going
        silent exactly where temporal resolution is already scarce, the estimator
        falls back to the last ``min_samples`` over whatever span they cover.
        """
        t, v = self._series.window(self.window, now)
        if t.size < self.min_samples:
            t, v = self._series.recent(self.min_samples)
            if t.size < self.min_samples:
                return None
        t_mean = t.mean()
        dt = t - t_mean
        denom = float(dt @ dt)
        if denom < 1e-12:
            return None
        return float(dt @ (v - v.mean()) / denom)

    def span(self, now: float | None = None, window: float | None = None) -> float:
        """Extent (max - min) of the values inside the window."""
        _, v = self._series.window(self.window if window is None else window, now)
        if v.size == 0:
            return 0.0
        return float(v.max() - v.min())

    def net(self, now: float | None = None, window: float | None = None) -> float:
        """Signed displacement from the oldest to the newest sample in the window."""
        _, v = self._series.window(self.window if window is None else window, now)
        if v.size < 2:
            return 0.0
        return float(v[-1] - v[0])

    def directness(self, now: float | None = None, window: float | None = None) -> float:
        """``|net| / span`` in ``[0, 1]``: 1 for a straight sweep, near 0 for a
        back-and-forth wobble.

        Separates a swipe from a wave using motion the two share -- both are fast
        and wide, so speed and travel cannot tell them apart. Only one of them
        ends up somewhere new.

        The window length is what makes it work. Over a short window a wave's
        half-cycle *is* monotonic and looks exactly like a sweep, so the same
        signal must also be checked over a window longer than one wave period,
        where the oscillation returns to where it began and a real swipe does
        not.
        """
        _, v = self._series.window(self.window if window is None else window, now)
        if v.size < 2:
            return 0.0
        span = float(v.max() - v.min())
        if span < 1e-9:
            return 0.0
        return abs(float(v[-1] - v[0])) / span
