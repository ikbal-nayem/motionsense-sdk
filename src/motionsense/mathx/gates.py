"""Hysteresis gate with temporal debouncing.

Every continuous ("level") activity in this SDK goes through a
:class:`SchmittGate` instead of a bare ``value > threshold`` test. A single
threshold makes a state that flickers on and off many times per second whenever
the measurement sits near it -- which is exactly where a user holding a pose
tends to land. Downstream that flicker becomes a key being pressed and released
30 times a second.

Two independent mechanisms are combined:

* **Hysteresis** (two thresholds): the value must travel a full band to change
  state. This rejects noise whose amplitude is smaller than the band.
* **Debounce** (minimum dwell time): a candidate change must persist. This
  rejects noise of *any* amplitude that is shorter than the dwell.

They cover different failure modes -- small-and-slow versus large-and-brief --
so both are worth having.
"""

from __future__ import annotations

import math

__all__ = ["SchmittGate"]


class SchmittGate:
    """Two-threshold gate with minimum on/off dwell times.

    Polarity is inferred from the thresholds: if ``on_threshold >
    off_threshold`` the gate opens on rising values, otherwise it opens on
    falling values. That keeps call sites readable -- ``SchmittGate(0.75, 0.85)``
    obviously means "on when the value drops below 0.75".
    """

    __slots__ = (
        "on_threshold",
        "off_threshold",
        "min_on",
        "min_off",
        "_rising",
        "_state",
        "_cand",
        "_cand_since",
        "_changed_at",
        "_conf",
    )

    def __init__(
        self,
        on_threshold: float,
        off_threshold: float,
        min_on: float = 0.0,
        min_off: float = 0.0,
        initial: bool = False,
    ):
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.min_on = float(min_on)
        self.min_off = float(min_off)
        self._rising = on_threshold > off_threshold
        self._state = bool(initial)
        self._cand: bool | None = None
        self._cand_since = 0.0
        self._changed_at = float("-inf")
        self._conf = 0.0

    # -- state ---------------------------------------------------------------
    @property
    def state(self) -> bool:
        return self._state

    @property
    def confidence(self) -> float:
        """How far past the opening threshold the last value sat, in ``[0, 1]``.

        Measured in units of the hysteresis band, so it is comparable across
        gates with different scales.
        """
        return self._conf

    @property
    def changed_at(self) -> float:
        return self._changed_at

    def reset(self, state: bool = False) -> None:
        self._state = state
        self._cand = None
        self._conf = 0.0
        self._changed_at = float("-inf")

    # -- update --------------------------------------------------------------
    def update(self, value: float | None, t: float) -> bool:
        """Feed one sample and return the debounced state.

        ``None`` or NaN means "no measurement" (landmark not visible, subject out
        of frame). That is treated as evidence for *off* rather than being
        ignored, so a gate cannot latch on forever after tracking is lost -- but
        it still goes through ``min_off``, so a one-frame dropout does not
        interrupt a held pose.
        """
        if value is None or math.isnan(value):
            raw = False
            self._conf = 0.0
        else:
            raw = self._evaluate(value)
            self._conf = self._confidence_for(value)

        if raw == self._state:
            self._cand = None
            return self._state

        if self._cand != raw:
            self._cand = raw
            self._cand_since = t

        dwell = self.min_on if raw else self.min_off
        if t - self._cand_since >= dwell:
            self._state = raw
            self._cand = None
            self._changed_at = t
        return self._state

    def _evaluate(self, value: float) -> bool:
        if self._rising:
            if value >= self.on_threshold:
                return True
            if value <= self.off_threshold:
                return False
        else:
            if value <= self.on_threshold:
                return True
            if value >= self.off_threshold:
                return False
        return self._state  # inside the dead band: hold

    def _confidence_for(self, value: float) -> float:
        band = self.on_threshold - self.off_threshold
        if abs(band) < 1e-12:
            return 1.0 if self._evaluate(value) else 0.0
        x = (value - self.off_threshold) / band
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
