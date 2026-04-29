"""Tests for `wind_forecast.eval.baselines`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wind_forecast.eval.baselines import (
    ALL_BASELINES,
    climatology,
    fit_climatology_bias,
    persistence,
    predict,
    raw_hrrr,
)


def _paired_frame(n: int = 24, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic paired frame: one cycle per hour for `n` hours."""
    rng = np.random.default_rng(seed)
    cycles = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    leads = [1, 6, 12]
    rows = []
    for cyc in cycles:
        for lead in leads:
            valid = cyc + pd.Timedelta(hours=lead)
            obs_u = rng.normal(0.0, 5.0)
            obs_v = rng.normal(0.0, 5.0)
            obs_gust = max(0.0, rng.normal(15.0, 3.0))
            # HRRR has a constant +1.0 bias on u and v; gust unbiased.
            rows.append({
                "cycle_utc": cyc,
                "lead_hour": lead,
                "valid_utc": valid,
                "u10": obs_u + 1.0,
                "v10": obs_v + 1.0,
                "gust": obs_gust,
                "obs_u": obs_u, "obs_v": obs_v, "obs_gust": obs_gust,
                "obs_drct": 0.0, "obs_sknt": math.hypot(obs_u, obs_v),
                "init_u": obs_u + 0.5, "init_v": obs_v + 0.5,
                "init_gust": obs_gust,
                "init_drct": 0.0, "init_sknt": math.hypot(obs_u, obs_v),
            })
    return pd.DataFrame(rows)


def test_raw_hrrr_returns_hrrr_columns() -> None:
    df = _paired_frame(n=4)
    pred = raw_hrrr(df)
    assert np.array_equal(pred["pred_u"].to_numpy(), df["u10"].to_numpy())
    assert np.array_equal(pred["pred_v"].to_numpy(), df["v10"].to_numpy())
    assert np.array_equal(pred["pred_gust"].to_numpy(), df["gust"].to_numpy())


def test_persistence_uses_init_columns() -> None:
    df = _paired_frame(n=4)
    pred = persistence(df)
    assert np.array_equal(pred["pred_u"].to_numpy(), df["init_u"].to_numpy())


def test_climatology_removes_constant_bias() -> None:
    df = _paired_frame(n=48)
    train = df.iloc[: len(df) // 2]
    test = df.iloc[len(df) // 2 :]
    bias_before = float(np.mean(test["u10"] - test["obs_u"]))
    pred = climatology(train, test)
    bias_after = float(np.mean(pred["pred_u"].to_numpy() - test["obs_u"].to_numpy()))
    # Started near +1.0, should now be much closer to zero.
    assert abs(bias_after) < abs(bias_before)
    assert abs(bias_after) < 0.5


def test_climatology_falls_back_to_train_mean_for_unseen_buckets() -> None:
    # All training cycles share month=1; test has a row in month=2 -> unseen bucket.
    df = _paired_frame(n=24)
    train = df.copy()
    test_cycle = pd.Timestamp("2024-02-15T00:00", tz="UTC")
    test_row = pd.DataFrame([{
        "cycle_utc": test_cycle,
        "lead_hour": 1,
        "valid_utc": test_cycle + pd.Timedelta(hours=1),
        "u10": 5.0, "v10": -3.0, "gust": 12.0,
        "obs_u": 0.0, "obs_v": 0.0, "obs_gust": 10.0,
        "obs_drct": 0.0, "obs_sknt": 0.0,
        "init_u": 0.0, "init_v": 0.0, "init_gust": 0.0,
        "init_drct": 0.0, "init_sknt": 0.0,
    }])
    pred = climatology(train, test_row)
    assert np.isfinite(pred["pred_u"]).all()
    # Bias for unseen bucket falls back to ~+1.0 (mean training bias).
    assert abs(float(pred["pred_u"].iloc[0]) - (5.0 - 1.0)) < 0.5


def test_fit_climatology_bias_returns_one_row_per_bucket() -> None:
    df = _paired_frame(n=24)
    bias = fit_climatology_bias(df, keys=["lead_hour"])
    # Three lead hours -> three bias rows.
    assert len(bias) == 3
    assert set(bias.columns) >= {"lead_hour", "bias_u", "bias_v", "bias_gust"}


def test_predict_dispatches_by_name() -> None:
    df = _paired_frame(n=8)
    train = df.iloc[:4]
    test = df.iloc[4:]
    for name in ALL_BASELINES:
        pred = predict(name, train, test)
        assert list(pred.columns) == ["pred_u", "pred_v", "pred_gust"]
        assert len(pred) == len(test)


def test_predict_unknown_baseline_raises() -> None:
    df = _paired_frame(n=4)
    try:
        predict("nonsense", df, df)
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("expected ValueError")
