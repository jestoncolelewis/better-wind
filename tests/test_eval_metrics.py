"""Tests for `wind_forecast.eval.metrics`."""

from __future__ import annotations

import math

import numpy as np

from wind_forecast.eval.metrics import (
    circular_diff_deg,
    crps_deterministic,
    direction_from_uv,
    mae,
    mae_direction_deg,
    metric_row,
    rmse,
    speed_from_uv,
)


def test_rmse_perfect_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_known_value() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.5, 2.0, 4.0])
    expected = math.sqrt(np.mean([0.25, 0.0, 1.0]))
    assert math.isclose(rmse(pred, obs), expected, rel_tol=1e-12)


def test_mae_ignores_nans_in_pred_and_obs() -> None:
    pred = np.array([1.0, np.nan, 3.0, 4.0])
    obs = np.array([1.0, 2.0, np.nan, 5.0])
    # Only finite pairs: (1,1) -> 0, (4,5) -> 1; mean = 0.5
    assert math.isclose(mae(pred, obs), 0.5, rel_tol=1e-12)


def test_circular_diff_handles_wraparound() -> None:
    diff = circular_diff_deg(np.array([350.0, 10.0, 180.0]), np.array([10.0, 350.0, 0.0]))
    assert np.allclose(diff, [20.0, 20.0, 180.0])


def test_mae_direction_masks_low_speed() -> None:
    pred = np.array([10.0, 90.0, 180.0])
    obs = np.array([20.0, 100.0, 200.0])
    # Both first two are below 3 kt — should be ignored. Only the 180 vs 200 (Δ=20) at 5 kt counts.
    speed = np.array([2.0, 2.5, 5.0])
    assert math.isclose(mae_direction_deg(pred, obs, speed), 20.0, rel_tol=1e-12)


def test_mae_direction_returns_nan_when_all_below_threshold() -> None:
    pred = np.array([10.0])
    obs = np.array([20.0])
    speed = np.array([0.5])
    assert math.isnan(mae_direction_deg(pred, obs, speed))


def test_speed_and_direction_round_trip_against_winds() -> None:
    # 10 kt from 360 -> u=0, v=-10. direction_from_uv should return ~360 (i.e. 0).
    speed = speed_from_uv(np.array([0.0]), np.array([-10.0]))
    direction = direction_from_uv(np.array([0.0]), np.array([-10.0]))
    assert math.isclose(speed[0], 10.0)
    assert math.isclose(direction[0] % 360.0, 0.0, abs_tol=1e-9)


def test_crps_deterministic_equals_mae() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    obs = np.array([2.0, 2.0, 5.0])
    assert math.isclose(crps_deterministic(pred, obs), mae(pred, obs), rel_tol=1e-12)


def test_metric_row_perfect_forecast() -> None:
    u = np.array([1.0, -2.0, 3.0])
    v = np.array([0.5, 4.0, -1.5])
    gust = np.array([5.0, 10.0, 15.0])
    row = metric_row(
        pred_u=u, pred_v=v, pred_gust=gust,
        obs_u=u, obs_v=v, obs_gust=gust,
    )
    assert row["n"] == 3
    assert row["rmse_u"] == 0.0
    assert row["rmse_v"] == 0.0
    assert row["rmse_speed"] == 0.0
    assert row["mae_speed"] == 0.0
    # Direction MAE: all winds are well above 3 kt and identical -> 0.
    assert row["mae_dir_deg"] == 0.0
    assert row["rmse_gust"] == 0.0
