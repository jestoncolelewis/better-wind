"""Wind vector conventions and conversions.

Internally we store meteorological u (eastward) and v (northward) components,
in knots. METAR reports `drct` as the direction the wind is blowing **from**
(0=N, 90=E) — converted to u/v on ingest.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


def dir_speed_to_uv(direction_deg_from: FloatArr, speed: FloatArr) -> tuple[FloatArr, FloatArr]:
    """Convert METAR (direction-from, speed) to (u, v).

    With θ in the "direction wind comes from" convention:
        u = -speed * sin(θ)   # eastward component of the *to* vector
        v = -speed * cos(θ)   # northward component of the *to* vector
    """
    theta = np.deg2rad(np.asarray(direction_deg_from, dtype=np.float64))
    s = np.asarray(speed, dtype=np.float64)
    u = -s * np.sin(theta)
    v = -s * np.cos(theta)
    return u, v


def uv_to_dir_speed(u: FloatArr, v: FloatArr) -> tuple[FloatArr, FloatArr]:
    """Convert (u, v) back to METAR-style (direction-from, speed)."""
    u_a = np.asarray(u, dtype=np.float64)
    v_a = np.asarray(v, dtype=np.float64)
    speed = np.hypot(u_a, v_a)
    direction = (np.rad2deg(np.arctan2(-u_a, -v_a)) + 360.0) % 360.0
    return direction, speed
