"""Tests for HRRR/METAR loading and pairing in `wind_forecast.eval.io`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_forecast.eval.io import (
    load_hrrr_forecasts,
    load_metar_obs,
    nearest_grid_point,
    pair_obs_to_forecasts,
)


def _hrrr_row(cycle: pd.Timestamp, lead: int, iy: int, ix: int, lat: float, lon: float) -> dict:
    return {
        "cycle_utc": cycle,
        "lead_hour": lead,
        "valid_utc": cycle + pd.Timedelta(hours=lead),
        "iy": iy, "ix": ix,
        "latitude": lat, "longitude": lon,
        "u10": 0.0, "v10": -5.0, "gust": 10.0,
    }


def test_nearest_grid_point_picks_closest_per_slot() -> None:
    cycle = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    rows = [
        _hrrr_row(cycle, 1, 0, 0, 43.5, -116.5),  # closest
        _hrrr_row(cycle, 1, 0, 1, 43.6, -116.4),
        _hrrr_row(cycle, 1, 1, 0, 43.7, -116.3),
        _hrrr_row(cycle, 2, 0, 0, 43.5, -116.5),  # closest
        _hrrr_row(cycle, 2, 0, 1, 43.8, -116.0),
    ]
    df = pd.DataFrame(rows)
    out = nearest_grid_point(df, lat=43.5817, lon=-116.5225)
    # One row per (cycle, lead). Both leads should keep iy=0, ix=0.
    assert len(out) == 2
    assert (out["iy"] == 0).all()
    assert (out["ix"] == 0).all()


def test_nearest_grid_point_handles_longitude_wrap() -> None:
    """Ensure normalization works when HRRR uses 0..360 and config uses -180..180."""
    cycle = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    rows = [
        _hrrr_row(cycle, 1, 0, 0, 43.5, 243.5),     # 0..360 form of -116.5
        _hrrr_row(cycle, 1, 0, 1, 43.5, 244.5),
    ]
    df = pd.DataFrame(rows)
    out = nearest_grid_point(df, lat=43.5, lon=-116.5)
    assert len(out) == 1
    assert out.iloc[0]["ix"] == 0


def test_pair_obs_to_forecasts_drops_unmatched_rows() -> None:
    cycle = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    fcst = pd.DataFrame([
        _hrrr_row(cycle, 1, 0, 0, 43.5, -116.5),
        _hrrr_row(cycle, 5, 0, 0, 43.5, -116.5),  # no obs near valid_utc -> dropped
    ])
    obs = pd.DataFrame([
        # Obs at 23:55 -> matches cycle (within 90 min)
        {"valid_utc": pd.Timestamp("2023-12-31T23:55", tz="UTC"),
         "u": 1.0, "v": 2.0, "gust": 8.0, "drct": 200.0, "sknt": 5.0},
        # Obs at 00:55 -> matches lead=1 (valid 01:00) within 30 min tolerance
        {"valid_utc": pd.Timestamp("2024-01-01T00:55", tz="UTC"),
         "u": 1.5, "v": 2.5, "gust": 9.0, "drct": 210.0, "sknt": 6.0},
        # Obs at 02:00 -> two hours away from lead=5 valid 05:00, outside tolerance
        {"valid_utc": pd.Timestamp("2024-01-01T02:00", tz="UTC"),
         "u": 2.0, "v": 3.0, "gust": 10.0, "drct": 220.0, "sknt": 7.0},
    ])
    paired = pair_obs_to_forecasts(fcst, obs)
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["lead_hour"] == 1
    assert row["obs_u"] == 1.5
    assert row["init_u"] == 1.0  # last obs at-or-before cycle


def test_pair_obs_init_is_nan_when_no_obs_before_cycle() -> None:
    cycle = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    fcst = pd.DataFrame([_hrrr_row(cycle, 1, 0, 0, 43.5, -116.5)])
    # All obs are AFTER cycle -> no init obs found.
    obs = pd.DataFrame([
        {"valid_utc": pd.Timestamp("2024-01-01T00:55", tz="UTC"),
         "u": 1.0, "v": 2.0, "gust": 5.0, "drct": 200.0, "sknt": 5.0},
    ])
    paired = pair_obs_to_forecasts(fcst, obs)
    assert len(paired) == 1
    assert pd.isna(paired.iloc[0]["init_u"])


def test_load_metar_obs_round_trip(tmp_path) -> None:
    from wind_forecast.config import Airport

    airport = Airport(
        icao="KTST", name="Test", latitude=40.0, longitude=-100.0,
        elevation_ft=1000, timezone="UTC",
        runways=[{"id": "10/28", "heading_deg_true": 100}],
    )
    out_dir = airport.raw_metar_dir(tmp_path)
    out_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "station": pd.array(["KTST", "KTST"], dtype="string"),
        "valid_utc": pd.to_datetime(
            ["2024-01-01T00:00", "2024-01-01T01:00"], utc=True
        ),
        "drct": [0.0, 90.0], "sknt": [10.0, 5.0], "gust": [np.nan, 15.0],
        "u": [0.0, -5.0], "v": [-10.0, 0.0],
        "tmpf": [30.0, 31.0], "dwpf": [20.0, 21.0],
        "alti": [30.0, 30.0], "mslp": [1020.0, 1020.0], "vsby": [10.0, 10.0],
        "metar": pd.array(["", ""], dtype="string"),
    })
    df.to_parquet(out_dir / "KTST.parquet", index=False)
    loaded = load_metar_obs(airport, data_root=tmp_path)
    assert len(loaded) == 2
    assert isinstance(loaded["valid_utc"].dtype, pd.DatetimeTZDtype)
    assert str(loaded["valid_utc"].dtype.tz) == "UTC"


def test_load_hrrr_forecasts_concats_year_dirs(tmp_path) -> None:
    from wind_forecast.config import Airport

    airport = Airport(
        icao="KTST", name="Test", latitude=40.0, longitude=-100.0,
        elevation_ft=1000, timezone="UTC",
        runways=[{"id": "10/28", "heading_deg_true": 100}],
    )
    root = airport.raw_hrrr_dir(tmp_path)
    (root / "2024").mkdir(parents=True)
    (root / "2025").mkdir(parents=True)
    cycle1 = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    cycle2 = pd.Timestamp("2025-06-01T00:00", tz="UTC")
    pd.DataFrame([_hrrr_row(cycle1, 1, 0, 0, 40.0, -100.0)]).to_parquet(
        root / "2024" / "20240601_00Z.parquet", index=False,
    )
    pd.DataFrame([_hrrr_row(cycle2, 1, 0, 0, 40.0, -100.0)]).to_parquet(
        root / "2025" / "20250601_00Z.parquet", index=False,
    )
    df = load_hrrr_forecasts(airport, data_root=tmp_path)
    assert len(df) == 2
    assert {pd.Timestamp(t).year for t in df["cycle_utc"]} == {2024, 2025}


def test_load_hrrr_missing_dir_raises(tmp_path) -> None:
    from wind_forecast.config import Airport

    airport = Airport(
        icao="KTST", name="Test", latitude=40.0, longitude=-100.0,
        elevation_ft=1000, timezone="UTC",
        runways=[{"id": "10/28", "heading_deg_true": 100}],
    )
    with pytest.raises(FileNotFoundError):
        load_hrrr_forecasts(airport, data_root=tmp_path)
