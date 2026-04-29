"""Tests for the end-to-end eval harness."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wind_forecast.config import Airport
from wind_forecast.eval.harness import (
    OVERALL_LEAD,
    chronological_split,
    evaluate_airport,
    format_table,
)


def _airport() -> Airport:
    return Airport(
        icao="KTST", name="Test", latitude=40.0, longitude=-100.0,
        elevation_ft=1000, timezone="UTC",
        runways=[{"id": "10/28", "heading_deg_true": 100}],
    )


def _paired_frame(n_cycles: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycles = pd.date_range("2024-01-01", periods=n_cycles, freq="h", tz="UTC")
    leads = [1, 6, 12]
    rows = []
    for cyc in cycles:
        for lead in leads:
            obs_u = rng.normal(0.0, 5.0)
            obs_v = rng.normal(0.0, 5.0)
            obs_gust = max(0.0, rng.normal(15.0, 3.0))
            rows.append({
                "cycle_utc": cyc,
                "lead_hour": lead,
                "valid_utc": cyc + pd.Timedelta(hours=lead),
                "u10": obs_u + 1.0, "v10": obs_v + 1.0, "gust": obs_gust,
                "obs_u": obs_u, "obs_v": obs_v, "obs_gust": obs_gust,
                "obs_drct": 0.0, "obs_sknt": math.hypot(obs_u, obs_v),
                "init_u": obs_u, "init_v": obs_v, "init_gust": obs_gust,
                "init_drct": 0.0, "init_sknt": 0.0,
            })
    return pd.DataFrame(rows)


def test_chronological_split_keeps_cycles_together() -> None:
    df = _paired_frame(n_cycles=20)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    # Splits should be ordered in time.
    assert train["cycle_utc"].max() < val["cycle_utc"].min()
    assert val["cycle_utc"].max() < test["cycle_utc"].min()
    # Every cycle's leads stay grouped.
    for part in (train, val, test):
        for _, sub in part.groupby("cycle_utc"):
            assert len(sub) == 3  # all three lead hours present


def test_chronological_split_rejects_bad_fractions() -> None:
    df = _paired_frame(n_cycles=10)
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.0, val_frac=0.1)
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.9, val_frac=0.2)


def test_evaluate_airport_runs_all_baselines() -> None:
    df = _paired_frame(n_cycles=40)
    metrics = evaluate_airport(_airport(), baselines=("hrrr", "persistence", "climatology"), paired=df)
    # Three baselines × (3 leads + 1 overall) = 12 rows.
    assert len(metrics) == 12
    assert set(metrics["baseline"].unique()) == {"hrrr", "persistence", "climatology"}
    assert set(metrics["lead_hour"].unique()) >= {1, 6, 12, OVERALL_LEAD}


def test_evaluate_airport_climatology_beats_raw_on_speed() -> None:
    """With a constant +1 kt bias on u and v, climatology should improve on raw HRRR."""
    df = _paired_frame(n_cycles=80)
    metrics = evaluate_airport(_airport(), baselines=("hrrr", "climatology"), paired=df)
    overall = metrics[metrics["lead_hour"] == OVERALL_LEAD].set_index("baseline")
    assert overall.loc["climatology", "rmse_speed"] < overall.loc["hrrr", "rmse_speed"]


def test_evaluate_airport_empty_paired_raises() -> None:
    empty = _paired_frame(n_cycles=0)
    with pytest.raises(RuntimeError, match="no paired"):
        evaluate_airport(_airport(), paired=empty)


def test_format_table_default_shows_only_overall() -> None:
    df = _paired_frame(n_cycles=40)
    metrics = evaluate_airport(_airport(), baselines=("hrrr",), paired=df)
    out = format_table(metrics, by_lead=False)
    # Overall should be there, individual leads should not.
    assert "all" in out
    assert " 1 " not in out  # lead-1 row is filtered


def test_format_table_by_lead_includes_per_lead_rows() -> None:
    df = _paired_frame(n_cycles=40)
    metrics = evaluate_airport(_airport(), baselines=("hrrr",), paired=df)
    out = format_table(metrics, by_lead=True)
    # Both per-lead and overall present.
    assert "all" in out
    assert "12" in out
