"""Iowa State Mesonet ASOS bulk downloader.

Fetches METAR observations for the target airport plus its configured
`neighbor_stations` and writes one tidy Parquet per station to
`data/raw/metar/{airport_icao}/{station}.parquet`.

Endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
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
DEFAULT_CHUNK_DAYS = 366
DEFAULT_WORKERS = 4

_thread_local = threading.local()


def _thread_session() -> requests.Session:
    """One requests.Session per worker thread.

    Sharing a single Session across many concurrent threads has connection-pool
    contention; per-thread sessions keep keep-alive and pool ownership clean.
    """
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


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


def date_chunks(start: date, end: date, chunk_days: int = DEFAULT_CHUNK_DAYS) -> list[tuple[date, date]]:
    """Tile `[start, end]` into inclusive chunks of at most `chunk_days` days.

    Mesonet handles many small requests far better than one huge one — its
    backend appears to scale poorly with the year range. Each chunk hits a
    fast path on their server.
    """
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if end < start:
        return []
    chunks: list[tuple[date, date]] = []
    cur = start
    span = timedelta(days=chunk_days - 1)
    while cur <= end:
        nxt = min(cur + span, end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


def _fetch_chunk(station: str, start: date, end: date) -> pd.DataFrame:
    """Worker task: fetch one (station, chunk) pair and parse it."""
    text = _fetch_station_csv(station, start, end, session=_thread_session())
    return parse_csv(text, station_override=station)


def _finalize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_frame()
    df = pd.concat(frames, ignore_index=True)
    return (
        df.drop_duplicates(subset=["station", "valid_utc"], keep="last")
          .sort_values("valid_utc")
          .reset_index(drop=True)
    )


def ingest_airport(
    airport: Airport,
    *,
    start: date | None = None,
    end: date | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_workers: int = DEFAULT_WORKERS,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Ingest the target airport and every neighbor station.

    Each station's date range is split into ~yearly chunks, and all
    (station, chunk) pairs run in a shared `ThreadPoolExecutor`. Existing
    Parquet files are skipped unless `skip_existing=False`.
    """
    resolved_start = start or airport.history_start or DEFAULT_HISTORY_START
    resolved_end = end or datetime.now(tz=timezone.utc).date()
    out_dir = airport.raw_metar_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = airport.all_stations()
    paths: dict[str, Path] = {s: out_dir / f"{s}.parquet" for s in stations}

    tasks: list[tuple[str, date, date]] = []
    skipped: list[str] = []
    for station in stations:
        path = paths[station]
        if skip_existing and path.exists() and path.stat().st_size > 0:
            skipped.append(station)
            continue
        for s, e in date_chunks(resolved_start, resolved_end, chunk_days):
            tasks.append((station, s, e))

    if skipped:
        logger.info("METAR skip-existing: %s", ", ".join(skipped))

    if not tasks:
        return {s: paths[s] for s in stations if paths[s].exists()}

    workers = max(1, min(max_workers, len(tasks)))
    logger.info(
        "METAR ingest: airport=%s stations=%d tasks=%d workers=%d chunk_days=%d",
        airport.icao, len(stations) - len(skipped), len(tasks), workers, chunk_days,
    )

    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    chunks_by_station: dict[str, list[pd.DataFrame]] = {s: [] for s in stations}
    bar = tqdm(total=len(tasks), desc=f"METAR {airport.icao}", unit="chunk", dynamic_ncols=True)
    with logging_redirect_tqdm(), bar, ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_fetch_chunk, station, s, e): (station, s, e)
            for station, s, e in tasks
        }
        for fut in as_completed(future_map):
            station, s, e = future_map[fut]
            try:
                df = fut.result()
            except Exception as exc:
                logger.warning(
                    "chunk failed station=%s [%s..%s]: %s", station, s, e, exc,
                )
                bar.update(1)
                continue
            chunks_by_station[station].append(df)
            logger.info("%s %s..%s -> %d rows", station, s, e, len(df))
            bar.update(1)

    written: dict[str, Path] = {}
    for station, frames in chunks_by_station.items():
        path = paths[station]
        if not frames:
            if path.exists():
                written[station] = path
            continue
        df = _finalize(frames)
        df.to_parquet(path, index=False)
        logger.info("wrote %d rows to %s", len(df), path)
        written[station] = path
    return written


def ingest_station(
    station: str,
    out_path: Path,
    *,
    start: date,
    end: date,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Fetch one station (chunked + parallel) and write a Parquet."""
    chunks = date_chunks(start, end, chunk_days)
    workers = max(1, min(max_workers, len(chunks)))
    frames: list[pd.DataFrame] = []
    if workers == 1:
        for s, e in chunks:
            frames.append(_fetch_chunk(station, s, e))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_chunk, station, s, e) for s, e in chunks]
            for fut in as_completed(futures):
                frames.append(fut.result())
    df = _finalize(frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return df
