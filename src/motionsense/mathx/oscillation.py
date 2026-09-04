"""Detection of repetitive back-and-forth motion (waving)."""

from __future__ import annotations

import numpy as np

from .series import RingSeries

__all__ = ["OscillationDetector", "OscillationResult"]


class OscillationResult:
    __slots__ = ("amplitude", "frequency", "half_cycles", "confidence")

    def __init__(self, amplitude: float, frequency: float, half_cycles: int, confidence: float):
        self.amplitude = amplitude
        self.frequency = frequency
        self.half_cycles = half_cycles
        self.confidence = confidence

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"OscillationResult(amplitude={self.amplitude:.3f}, "
            f"frequency={self.frequency:.2f}Hz, half_cycles={self.half_cycles}, "
            f"confidence={self.confidence:.2f})"
        )


class OscillationDetector:
    """Recognises a 1-D signal swinging back and forth, e.g. a waving wrist.

    The naive approach -- count how often the per-frame difference changes sign
    -- fails in two directions at once. Landmark jitter flips the sign of a small
    difference constantly, so a perfectly still hand reads as a fast wave; and a
    real wave that also drifts across the frame (because the whole arm is
    moving) spends long stretches with a single sign and reads as no wave at all.

    Three corrections, in order:

    1. **Detrend.** Fit and subtract a least-squares line over the window. This
       removes bulk translation of the arm, leaving only the oscillation. A wave
       performed while walking sideways still registers.
    2. **Amplitude gate.** Measure peak-to-peak of the residual in body units and
       require a real excursion. Scale-invariant, so it behaves the same close to
       and far from the camera.
    3. **Schmitt-triggered zero crossings.** Count sign changes only after the
       residual crosses +/- ``hysteresis * amplitude``, rather than at zero. Noise
       around the centre line no longer counts as a crossing, and the count
       becomes the number of genuine half-cycles.

    Frequency then follows from ``half_cycles / (2 * span)`` and is band-limited,
    which separates a wave (2-5 Hz) from a slow deliberate reach and from
    high-frequency tracking noise.
    """

    __slots__ = (
        "window",
        "min_half_cycles",
        "min_amplitude",
        "freq_band",
        "cooldown",
        "hysteresis",
        "min_samples",
        "_series",
        "_last_fire",
        "_last_result",
    )

    def __init__(
        self,
        window: float = 1.1,
        min_half_cycles: int = 3,
        min_amplitude: float = 0.09,
        freq_band: tuple[float, float] = (1.1, 6.5),
        cooldown: float = 0.85,
        hysteresis: float = 0.32,
        min_samples: int = 7,
    ):
        self.window = window
        self.min_half_cycles = min_half_cycles
        self.min_amplitude = min_amplitude
        self.freq_band = freq_band
        self.cooldown = cooldown
        self.hysteresis = hysteresis
        self.min_samples = min_samples
        self._series = RingSeries(capacity=128)
        self._last_fire = float("-inf")
        self._last_result: OscillationResult | None = None

    @property
    def last_result(self) -> OscillationResult | None:
        return self._last_result

    @property
    def last_fire(self) -> float:
        return self._last_fire

    def clear(self) -> None:
        self._series.clear()
        self._last_result = None

    def update(self, t: float, value: float) -> OscillationResult | None:
        """Push a sample; returns a result on the frame the oscillation is confirmed."""
        self._series.push(t, value)
        self._last_result = None

        if t - self._last_fire < self.cooldown:
            return None

        ts, vs = self._series.window(self.window, now=t)
        n = ts.size
        if n < self.min_samples:
            return None
        span = float(ts[-1] - ts[0])
        if span < 0.3 * self.window:
            return None

        # 1. Detrend: subtract the least-squares line.
        t_c = ts - ts.mean()
        denom = float(t_c @ t_c)
        if denom < 1e-12:
            return None
        slope = float(t_c @ (vs - vs.mean()) / denom)
        residual = vs - (vs.mean() + slope * t_c)

        # 2. Amplitude gate (half peak-to-peak, in body units).
        amplitude = 0.5 * float(residual.max() - residual.min())
        if amplitude < self.min_amplitude:
            return None

        # 3. Schmitt-triggered crossings of +/- h.
        h = self.hysteresis * amplitude
        sign = np.zeros(n, dtype=np.int8)
        sign[residual > h] = 1
        sign[residual < -h] = -1
        nz = sign[sign != 0]
        if nz.size < 2:
            return None
        half_cycles = int(np.count_nonzero(np.diff(nz) != 0))
        if half_cycles < self.min_half_cycles:
            return None

        frequency = half_cycles / (2.0 * span)
        lo, hi = self.freq_band
        if not (lo <= frequency <= hi):
            return None

        confidence = min(
            1.0,
            0.45 * min(1.0, amplitude / (2.0 * self.min_amplitude))
            + 0.55 * min(1.0, half_cycles / (self.min_half_cycles + 2.0)),
        )
        result = OscillationResult(amplitude, frequency, half_cycles, confidence)
        self._last_result = result
        self._last_fire = t
        self._series.clear()
        return result
