"""Iowa State Mesonet ASOS bulk downloader.

Fetches METAR observations for the target airport plus its configured
`neighbor_stations` and writes one tidy Parquet per station to
`data/raw/metar/{airport_icao}/{station}.parquet`.

Endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wind_forecast.config import Airport, DEFAULT_DATA_ROOT
from wind_forecast.winds import dir_speed_to_uv

logger = logging.getLogger(__name__)

ASOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Fields pulled from the Mesonet CSV. Order doesn't matter; we select columns
# explicitly after parsing so extra fields are harmless.
REQUEST_FIELDS: tuple[str, ...] = (
    "drct",   # wind direction, degrees from
    "sknt",   # wind speed, knots
    "gust",   # gust, knots
    "tmpf",   # air temperature, F
    "dwpf",   # dewpoint, F
    "alti",   # altimeter, inHg
    "mslp",   # sea level pressure, mb
    "vsby",   # visibility, miles
    "metar", # raw METAR
)

# Final schema written to Parquet. Kept identical across stations and airports
# so downstream code can concat without surprises. Dtype is pinned at write
# time so that an all-null column at one station still matches the numeric
# column at another.
OUTPUT_DTYPES: dict[str, str] = {
    "station": "string",
    "valid_utc": "datetime64[ns, UTC]",
    "drct": "float64",
    "sknt": "float64",
    "gust": "float64",
    "u": "float64",
    "v": "float64",
    "tmpf": "float64",
    "dwpf": "float64",
    "alti": "float64",
    "mslp": "float64",
    "vsby": "float64",
    "metar": "string",
}
OUTPUT_COLUMNS: tuple[str, ...] = tuple(OUTPUT_DTYPES.keys())

DEFAULT_HISTORY_START = date(2015, 1, 1)


def _build_query(
    station: str,
    start: date,
    end: date,
    fields: Iterable[str] = REQUEST_FIELDS,
) -> dict[str, str]:
    params: dict[str, str] = {
        "station": station,
        "data": ",".join(fields),
        "year1": str(start.year),
        "month1": str(start.month),
        "day1": str(start.day),
        "year2": str(end.year),
        "month2": str(end.month),
        "day2": str(end.day),
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",
        "trace": "empty",
        "direct": "no",
        "report_type": "3,4",  # routine + special
    }
    return params


def _fetch_station_csv(
    station: str,
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
    max_retries: int = 4,
) -> str:
    """GET one station's CSV from the Mesonet endpoint with simple backoff."""
    s = session or requests.Session()
    params = _build_query(station, start, end)
    backoff = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = s.get(ASOS_ENDPOINT, params=params, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last_exc = e
            logger.warning(
                "Mesonet fetch failed for %s (attempt %d/%d): %s",
                station, attempt + 1, max_retries, e,
            )
            time.sleep(backoff)
            backoff *= 2
    assert last_exc is not None
    raise last_exc


def _empty_frame() -> pd.DataFrame:
    """Return a zero-row frame with the canonical schema and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in OUTPUT_DTYPES.items()})


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Project onto OUTPUT_COLUMNS and pin dtypes."""
    for col, dtype in OUTPUT_DTYPES.items():
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype(dtype)
    return df[list(OUTPUT_COLUMNS)]


def parse_csv(text: str, station_override: str | None = None) -> pd.DataFrame:
    """Parse the Mesonet CSV into the canonical OUTPUT_COLUMNS schema.

    `station_override` forces the `station` column to a fixed value before
    dedup. Iowa Mesonet returns 3-letter FAA codes for US stations
    (e.g. `MAN` for KMAN) even when you query with the 4-letter ICAO, so the
    caller always passes the ICAO it requested to keep the partition key
    consistent with `Airport.icao`.
    """
    df = pd.read_csv(
        io.StringIO(text),
        na_values=["M", "", " "],
        dtype={"station": "string", "metar": "string"},
        low_memory=False,
    )
    if df.empty:
        return _empty_frame()

    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid_utc"]).copy()

    numeric_cols = ["drct", "sknt", "gust", "tmpf", "dwpf", "alti", "mslp", "vsby"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    mask = df["drct"].notna() & df["sknt"].notna()
    u = np.full(len(df), np.nan, dtype="float64")
    v = np.full(len(df), np.nan, dtype="float64")
    if mask.any():
        mask_arr = mask.to_numpy()
        u_vals, v_vals = dir_speed_to_uv(
            df.loc[mask, "drct"].to_numpy(dtype="float64"),
            df.loc[mask, "sknt"].to_numpy(dtype="float64"),
        )
        u[mask_arr] = u_vals
        v[mask_arr] = v_vals
    df["u"] = u
    df["v"] = v

    if "metar" not in df.columns:
        df["metar"] = pd.NA

    if station_override is not None:
        df["station"] = station_override

    out = _coerce_schema(df).sort_values("valid_utc").reset_index(drop=True)
    out = out.drop_duplicates(subset=["station", "valid_utc"], keep="last").reset_index(drop=True)
    return out


def ingest_station(
    station: str,
    out_path: Path,
    *,
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one station, write Parquet, return the frame."""
    logger.info("Fetching METAR for %s  [%s..%s]", station, start, end)
    csv_text = _fetch_station_csv(station, start, end, session=session)
    df = parse_csv(csv_text, station_override=station)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return df


def ingest_airport(
    airport: Airport,
    *,
    start: date | None = None,
    end: date | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Ingest the target airport and every neighbor station.

    Returns a mapping of station ICAO to the Parquet path that was written.
    """
    resolved_start = start or airport.history_start or DEFAULT_HISTORY_START
    resolved_end = end or datetime.now(tz=timezone.utc).date()
    out_dir = airport.raw_metar_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = session or requests.Session()
    written: dict[str, Path] = {}
    for station in airport.all_stations():
        path = out_dir / f"{station}.parquet"
        ingest_station(
            station, path, start=resolved_start, end=resolved_end, session=sess
        )
        written[station] = path
    return written
