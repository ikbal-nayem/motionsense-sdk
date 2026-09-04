"""Optional per-subject calibration.

The built-in thresholds are population averages expressed in body units, and
they work without any setup. Calibration makes them personal: it records what
*this* subject's neutral standing pose measures, and recognizers then express
their thresholds as fractions of that neutral instead of as absolutes.

It matters most for features where body proportion and camera angle both enter,
such as how far the nose normally sits above the shoulder line -- that value
differs by a factor of two between a tall subject with a high camera and a short
one with a low camera, which is enough to make a fixed "head down" threshold
either fire constantly or never.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Calibration", "CalibrationCollector"]

#: Features worth calibrating, with the population default used when uncalibrated.
NEUTRAL_DEFAULTS: dict[str, float] = {
    "nose_above_shoulders": 0.30,
    "shoulder_center_height": 0.86,
    "elevation": 2.4,
    "thigh_length": 0.72,
    "arm_length": 0.94,
    "shoulder_width": 0.52,
}


@dataclass(frozen=True, slots=True)
class Calibration:
    """Neutral reference measurements for one subject."""

    neutral: dict[str, float] = field(default_factory=dict)
    captured_at: float | None = None
    samples: int = 0

    @property
    def is_set(self) -> bool:
        return bool(self.neutral)

    def get(self, key: str, default: float | None = None) -> float:
        """Calibrated value for ``key``, falling back to the population default."""
        if key in self.neutral:
            return self.neutral[key]
        if default is not None:
            return default
        return NEUTRAL_DEFAULTS.get(key, float("nan"))

    def scaled(self, key: str, ratio: float, absolute: float) -> float:
        """A threshold as ``ratio`` of the calibrated neutral for ``key``.

        Falls back to ``absolute`` when this key was never calibrated, so call
        sites read the same whether or not the user ran a calibration.
        """
        if key in self.neutral:
            return self.neutral[key] * ratio
        return absolute

    def to_dict(self) -> dict:
        return {"neutral": dict(self.neutral), "captured_at": self.captured_at, "samples": self.samples}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(
            neutral=dict(data.get("neutral", {})),
            captured_at=data.get("captured_at"),
            samples=int(data.get("samples", 0)),
        )


EMPTY = Calibration()


class CalibrationCollector:
    """Accumulates neutral-pose samples and reduces them to a :class:`Calibration`.

    Reduces with the **median**, not the mean. A calibration window is a few
    seconds of a human standing still, and it reliably contains a handful of
    frames where a landmark jumps -- someone blinks, an arm is briefly
    mis-assigned, the detector re-acquires. The median ignores those outright;
    a mean lets a single bad frame skew every threshold derived from it.
    """

    __slots__ = ("duration", "_started", "_values")

    def __init__(self, duration: float = 2.0):
        self.duration = duration
        self._started: float | None = None
        self._values: dict[str, list[float]] = {}

    @property
    def started(self) -> bool:
        return self._started is not None

    def begin(self, t: float) -> None:
        self._started = t
        self._values = {}

    def progress(self, t: float) -> float:
        if self._started is None:
            return 0.0
        return min(1.0, (t - self._started) / max(self.duration, 1e-6))

    def add(self, features, t: float) -> None:
        """Sample the calibratable features of one frame."""
        if self._started is None:
            self.begin(t)
        for key in NEUTRAL_DEFAULTS:
            value = getattr(features, key, None)
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            self._values.setdefault(key, []).append(value)

    def done(self, t: float) -> bool:
        return self._started is not None and (t - self._started) >= self.duration

    def finish(self) -> Calibration:
        neutral = {
            key: float(np.median(values))
            for key, values in self._values.items()
            if len(values) >= 5
        }
        count = max((len(v) for v in self._values.values()), default=0)
        self._started = None
        self._values = {}
        return Calibration(neutral=neutral, captured_at=time.time(), samples=count)
