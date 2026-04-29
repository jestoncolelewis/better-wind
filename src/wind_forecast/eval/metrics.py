"""Pointwise + circular metrics for deterministic baselines.

CLAUDE.md calls for RMSE/MAE on u, v, scalar speed, gust, plus a circular
direction error masked at 3 kt observed wind, plus CRPS from the quantile
output. Phase 2 baselines are deterministic, so CRPS reduces to MAE on scalar
speed (CRPS of a delta distribution at y against truth x is `|y − x|`).

All inputs are 1-D numpy arrays of equal length. Functions tolerate NaNs via
`nanmean` — pairs where either side is missing are skipped.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]

DIRECTION_MIN_SPEED_KT = 3.0  # below this, observed direction is meaningless


def _as_arr(x: object) -> FloatArr:
    return np.asarray(x, dtype=np.float64)


def rmse(pred: object, obs: object) -> float:
    """Root mean squared error, ignoring NaNs in either side."""
    p = _as_arr(pred)
    o = _as_arr(obs)
    diff = p - o
    if not np.any(np.isfinite(diff)):
        return float("nan")
    return float(np.sqrt(np.nanmean(diff * diff)))


def mae(pred: object, obs: object) -> float:
    """Mean absolute error, ignoring NaNs in either side."""
    p = _as_arr(pred)
    o = _as_arr(obs)
    diff = np.abs(p - o)
    if not np.any(np.isfinite(diff)):
        return float("nan")
    return float(np.nanmean(diff))


def circular_diff_deg(pred_deg: object, obs_deg: object) -> FloatArr:
    """Smallest angular difference in degrees, in [0, 180]."""
    p = _as_arr(pred_deg)
    o = _as_arr(obs_deg)
    diff = np.abs(p - o) % 360.0
    return np.minimum(diff, 360.0 - diff)


def mae_direction_deg(
    pred_deg: object,
    obs_deg: object,
    obs_speed: object,
    *,
    min_speed_kt: float = DIRECTION_MIN_SPEED_KT,
) -> float:
    """Circular MAE in degrees, masked where observed wind < `min_speed_kt`."""
    diff = circular_diff_deg(pred_deg, obs_deg)
    speed = _as_arr(obs_speed)
    mask = np.isfinite(diff) & np.isfinite(speed) & (speed >= min_speed_kt)
    if not mask.any():
        return float("nan")
    return float(np.nanmean(diff[mask]))


def crps_deterministic(pred: object, obs: object) -> float:
    """CRPS of a deterministic forecast — equivalent to MAE.

    Provided as a named symbol so downstream metric tables can stay consistent
    when the LightGBM quantile model lands and CRPS becomes proper.
    """
    return mae(pred, obs)


def speed_from_uv(u: object, v: object) -> FloatArr:
    return np.hypot(_as_arr(u), _as_arr(v))


def direction_from_uv(u: object, v: object) -> FloatArr:
    """METAR-style direction-from in [0, 360)."""
    ua = _as_arr(u)
    va = _as_arr(v)
    return (np.rad2deg(np.arctan2(-ua, -va)) + 360.0) % 360.0


class MetricRow(TypedDict, total=False):
    baseline: str
    lead_hour: int | str
    n: int
    rmse_u: float
    rmse_v: float
    rmse_speed: float
    mae_u: float
    mae_v: float
    mae_speed: float
    mae_dir_deg: float
    rmse_gust: float
    mae_gust: float
    crps_speed: float


def metric_row(
    *,
    pred_u: object,
    pred_v: object,
    pred_gust: object,
    obs_u: object,
    obs_v: object,
    obs_gust: object,
) -> MetricRow:
    """Compute every per-target metric over one set of paired arrays."""
    pu = _as_arr(pred_u)
    pv = _as_arr(pred_v)
    ou = _as_arr(obs_u)
    ov = _as_arr(obs_v)

    pred_speed = speed_from_uv(pu, pv)
    obs_speed = speed_from_uv(ou, ov)
    pred_dir = direction_from_uv(pu, pv)
    obs_dir = direction_from_uv(ou, ov)

    return MetricRow(
        n=int(np.sum(np.isfinite(pu - ou))),
        rmse_u=rmse(pu, ou),
        rmse_v=rmse(pv, ov),
        rmse_speed=rmse(pred_speed, obs_speed),
        mae_u=mae(pu, ou),
        mae_v=mae(pv, ov),
        mae_speed=mae(pred_speed, obs_speed),
        mae_dir_deg=mae_direction_deg(pred_dir, obs_dir, obs_speed),
        rmse_gust=rmse(pred_gust, obs_gust),
        mae_gust=mae(pred_gust, obs_gust),
        crps_speed=crps_deterministic(pred_speed, obs_speed),
    )
