"""Read raw HRRR + METAR Parquets and pair them on `valid_utc` / `cycle_utc`.

Phase 2 keeps this deliberately simple: we only need the nearest HRRR grid
point to the airport, the obs at the forecast valid time (ground truth), and
the obs at the cycle init time (persistence baseline).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wind_forecast.config import DEFAULT_DATA_ROOT, Airport

logger = logging.getLogger(__name__)

DEFAULT_VALID_TOLERANCE = pd.Timedelta(minutes=30)
DEFAULT_CYCLE_TOLERANCE = pd.Timedelta(minutes=90)


def load_metar_obs(
    airport: Airport,
    *,
    station: str | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load the METAR Parquet for one station (default: target airport ICAO)."""
    station = (station or airport.icao).upper()
    path = airport.raw_metar_dir(data_root) / f"{station}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"METAR Parquet not found at {path}. Run `wind-forecast ingest-metar` first."
        )
    df = pd.read_parquet(path)
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    return df.sort_values("valid_utc").reset_index(drop=True)


def load_hrrr_forecasts(
    airport: Airport, *, data_root: Path = DEFAULT_DATA_ROOT
) -> pd.DataFrame:
    """Concatenate every per-cycle HRRR Parquet under the airport's raw dir."""
    root = airport.raw_hrrr_dir(data_root)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No HRRR Parquets under {root}. Run `wind-forecast ingest-hrrr` first."
        )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["cycle_utc"] = pd.to_datetime(df["cycle_utc"], utc=True)
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    return df


def nearest_grid_point(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Reduce a multi-grid-point HRRR frame to one row per (cycle_utc, lead_hour).

    Picks the grid cell closest to `(lat, lon)` for each forecast slot. We
    compute the squared error in (Δlat, Δlon·cos(lat)) — sufficient to
    discriminate among ~3 km HRRR cells for an airport-sized footprint.
    """
    if df.empty:
        return df
    lon_norm = ((df["longitude"].to_numpy() - lon) + 180.0) % 360.0 - 180.0
    dlat = df["latitude"].to_numpy() - lat
    dist2 = dlat * dlat + (lon_norm * np.cos(np.deg2rad(lat))) ** 2
    df = df.assign(_dist2=dist2)
    keepers = df.groupby(["cycle_utc", "lead_hour"], sort=False)["_dist2"].idxmin()
    return (
        df.loc[keepers]
        .drop(columns="_dist2")
        .sort_values(["cycle_utc", "lead_hour"])
        .reset_index(drop=True)
    )


def pair_obs_to_forecasts(
    fcst: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    valid_tolerance: pd.Timedelta = DEFAULT_VALID_TOLERANCE,
    cycle_tolerance: pd.Timedelta = DEFAULT_CYCLE_TOLERANCE,
) -> pd.DataFrame:
    """Attach observed truth (at `valid_utc`) and persistence init (at `cycle_utc`).

    Adds columns `obs_u/obs_v/obs_gust/obs_drct/obs_sknt` (truth) and
    `init_u/init_v/init_gust/init_drct/init_sknt` (latest obs at or before the
    cycle time, for the persistence baseline). Rows without a truth observation
    inside `valid_tolerance` are dropped. `init_*` is filled when an obs exists
    at-or-before the cycle within `cycle_tolerance`, otherwise NaN — metrics use
    `nanmean` so missing init rows are skipped.
    """
    if fcst.empty:
        return fcst.iloc[0:0].copy()

    obs_cols = ["valid_utc", "u", "v", "gust", "drct", "sknt"]
    obs_clean = (
        obs[obs_cols]
        .dropna(subset=["valid_utc"])
        .sort_values("valid_utc")
        .reset_index(drop=True)
    )

    truth = obs_clean.rename(
        columns={
            "u": "obs_u",
            "v": "obs_v",
            "gust": "obs_gust",
            "drct": "obs_drct",
            "sknt": "obs_sknt",
        }
    )
    fcst_sorted = fcst.sort_values("valid_utc").reset_index(drop=True)
    paired = pd.merge_asof(
        fcst_sorted,
        truth,
        on="valid_utc",
        direction="nearest",
        tolerance=valid_tolerance,
    )

    init = obs_clean.rename(
        columns={
            "valid_utc": "_obs_time",
            "u": "init_u",
            "v": "init_v",
            "gust": "init_gust",
            "drct": "init_drct",
            "sknt": "init_sknt",
        }
    )
    paired = paired.sort_values("cycle_utc").reset_index(drop=True)
    paired = pd.merge_asof(
        paired,
        init,
        left_on="cycle_utc",
        right_on="_obs_time",
        direction="backward",
        tolerance=cycle_tolerance,
    ).drop(columns=["_obs_time"])

    paired = paired.dropna(subset=["obs_u", "obs_v"]).reset_index(drop=True)
    paired = paired.sort_values(["cycle_utc", "lead_hour"]).reset_index(drop=True)
    logger.info(
        "paired %d (cycle, lead) rows with observed truth (valid_tol=%s)",
        len(paired), valid_tolerance,
    )
    return paired


def load_and_pair(
    airport: Airport,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    valid_tolerance: pd.Timedelta = DEFAULT_VALID_TOLERANCE,
    cycle_tolerance: pd.Timedelta = DEFAULT_CYCLE_TOLERANCE,
) -> pd.DataFrame:
    """End-to-end: load METAR + HRRR for one airport, return the paired frame."""
    obs = load_metar_obs(airport, data_root=data_root)
    fcst = load_hrrr_forecasts(airport, data_root=data_root)
    fcst = nearest_grid_point(fcst, airport.latitude, airport.longitude)
    return pair_obs_to_forecasts(
        fcst, obs,
        valid_tolerance=valid_tolerance,
        cycle_tolerance=cycle_tolerance,
    )
