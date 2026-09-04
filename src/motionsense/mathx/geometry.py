"""Planar geometry helpers used by the feature extractor and recognizers.

Conventions inside this package (after normalisation):

* ``+x`` points to the **subject's right**, ``+y`` points **up**.
* Lengths are in *body units*: one unit is roughly a shoulder-to-hip torso.
* Angles are radians.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "angle_at",
    "angle_at3",
    "direction",
    "procrustes_scale",
    "signed_angle_from_up",
]

_EPS = 1e-9


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC in radians, in ``[0, pi]``.

    Uses ``atan2(|cross|, dot)`` instead of ``acos(dot / (|u||v|))``. The acos
    form has two defects that matter here: its derivative blows up near 0 and pi
    (exactly where a straightened elbow or knee is judged, so noise gets
    amplified precisely where the threshold sits), and rounding can push the
    quotient a hair outside ``[-1, 1]`` and yield NaN. The atan2 form is
    well-conditioned over the whole range and cannot produce NaN for finite
    input.
    """
    ux, uy = float(a[0]) - float(b[0]), float(a[1]) - float(b[1])
    vx, vy = float(c[0]) - float(b[0]), float(c[1]) - float(b[1])
    cross = abs(ux * vy - uy * vx)
    dot = ux * vx + uy * vy
    if cross < _EPS and abs(dot) < _EPS:
        return 0.0
    return math.atan2(cross, dot)


def angle_at3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC for 3-vectors, in ``[0, pi]``.

    Joint angles belong in 3D. In a frontal camera view a knee bending directly
    toward the lens barely changes its 2D angle -- the limb foreshortens instead
    of rotating -- so a 2D squat test misses exactly the squat people actually
    perform. Fed MediaPipe's world landmarks this reads the true joint angle
    regardless of which way the subject faces.

    Same ``atan2(|cross|, dot)`` formulation as :func:`angle_at`, with the 3D
    cross product.
    """
    ux, uy, uz = float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])
    vx, vy, vz = float(c[0]) - float(b[0]), float(c[1]) - float(b[1]), float(c[2]) - float(b[2])
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    cross = math.sqrt(cx * cx + cy * cy + cz * cz)
    dot = ux * vx + uy * vy + uz * vz
    if cross < _EPS and abs(dot) < _EPS:
        return 0.0
    return math.atan2(cross, dot)


def direction(v: np.ndarray) -> float:
    """Angle of ``v`` measured from the ``+x`` axis, in ``(-pi, pi]``."""
    return math.atan2(float(v[1]), float(v[0]))


def signed_angle_from_up(v: np.ndarray) -> float:
    """Angle of ``v`` away from straight up, signed toward ``+x``.

    Positive means tilted toward the subject's right. Returns radians in
    ``(-pi, pi]``.
    """
    return math.atan2(float(v[0]), float(v[1]))


def procrustes_scale(points: np.ndarray) -> float:
    """Root-mean-square radius of a point set about its own centroid.

    This is the scale term of a Procrustes fit, and it is the right way to
    measure "how big is this person in the frame". The obvious alternatives are
    both fragile: shoulder width collapses when the subject turns sideways, and
    torso length collapses when they lean toward or away from the camera. The
    RMS radius over the four torso corners degrades gracefully instead of
    vanishing, because it pools every available distance rather than betting on
    one segment, and a single mis-placed landmark moves it by ``O(1/n)``.
    """
    centroid = points.mean(axis=0)
    d = points - centroid
    return float(np.sqrt(np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)))
