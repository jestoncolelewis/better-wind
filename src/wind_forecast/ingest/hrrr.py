"""HRRR retrieval via `herbie`.

For each requested init cycle, pull the forecast hours we care about, extract
a 5x5 grid box around the airport's lat/lon for every required variable, and
save one Parquet per cycle at `data/raw/hrrr/{icao}/{YYYY}/{YYYYMMDD_HHZ}.parquet`.

Heavy imports (`herbie`, `xarray`, `cfgrib`) are deferred to call-time so that
unit tests can import this module without pulling in the entire GRIB stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from wind_forecast.config import DEFAULT_DATA_ROOT, Airport

ProgressCallback = Callable[[int, int, dict[str, Any]], None]

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

CycleStatus = Literal["written", "skipped", "empty"]


@dataclass(frozen=True)
class CycleResult:
    """Outcome of fetching one HRRR init cycle."""

    cycle: datetime
    path: Path | None
    status: CycleStatus
    rows: int


@dataclass
class IngestSummary:
    """Aggregate counts across an ingest run."""

    total: int = 0
    written: int = 0
    skipped: int = 0
    empty: int = 0
    failed: int = 0
    rows: int = 0

    @property
    def done(self) -> int:
        return self.written + self.skipped + self.empty + self.failed

    def postfix(self) -> dict[str, str]:
        return {
            "new": str(self.written),
            "skip": str(self.skipped),
            "fail": str(self.failed + self.empty),
            "rows": _humanize(self.rows),
        }


def _humanize(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}G"


DEFAULT_LEAD_HOURS: tuple[int, ...] = tuple(range(1, 19))  # +1h..+18h
DEFAULT_GRID_HALF: int = 2  # -> 5x5 box
DEFAULT_WORKERS: int = 8
DEFAULT_CYCLE_WORKERS: int = 4


@dataclass(frozen=True)
class HRRRVariableSpec:
    """One GRIB field to extract.

    `search` is the regex passed to `Herbie.xarray(searchString=...)`.
    `out_prefix` is used to name the resulting columns in the flattened frame.
    """

    name: str
    search: str
    out_prefix: str


# Variable catalog. Regexes target HRRR's wgrib2-style index lines.
HRRR_VARIABLES: tuple[HRRRVariableSpec, ...] = (
    HRRRVariableSpec("u10", r":UGRD:10 m above ground:", "u10"),
    HRRRVariableSpec("v10", r":VGRD:10 m above ground:", "v10"),
    HRRRVariableSpec("gust", r":GUST:surface:", "gust"),
    HRRRVariableSpec("u925", r":UGRD:925 mb:", "u925"),
    HRRRVariableSpec("v925", r":VGRD:925 mb:", "v925"),
    HRRRVariableSpec("t925", r":TMP:925 mb:", "t925"),
    HRRRVariableSpec("u850", r":UGRD:850 mb:", "u850"),
    HRRRVariableSpec("v850", r":VGRD:850 mb:", "v850"),
    HRRRVariableSpec("t850", r":TMP:850 mb:", "t850"),
    HRRRVariableSpec("t2m", r":TMP:2 m above ground:", "t2m"),
    HRRRVariableSpec("d2m", r":DPT:2 m above ground:", "d2m"),
    HRRRVariableSpec("psfc", r":PRES:surface:", "psfc"),
    HRRRVariableSpec("mslp", r":MSLMA:mean sea level:", "mslp"),
    HRRRVariableSpec("pblh", r":HPBL:surface:", "pblh"),
    HRRRVariableSpec("cape", r":CAPE:surface:", "cape"),
    HRRRVariableSpec("cin", r":CIN:surface:", "cin"),
)


def iter_cycles(start: datetime, end: datetime, step_hours: int = 1) -> Iterator[datetime]:
    """Yield tz-aware UTC cycle datetimes in `[start, end)` at `step_hours`."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cur = start
    step = timedelta(hours=step_hours)
    while cur < end:
        yield cur
        cur += step


def _nearest_grid_box(
    ds: xr.Dataset, lat: float, lon: float, half: int = DEFAULT_GRID_HALF
) -> xr.Dataset:
    """Return a (2*half+1)x(2*half+1) slice around the nearest grid point.

    HRRR is on a Lambert-conformal grid with 2-D `latitude`/`longitude`
    coordinates over `y`/`x` dims. We find the nearest (y, x) by minimum
    haversine-ish distance and then slice a box around it.
    """
    # Normalize longitude to match HRRR (which uses 0..360)
    lon_q = lon % 360.0
    lat_grid = ds["latitude"].values
    lon_grid = ds["longitude"].values % 360.0

    dlat = lat_grid - lat
    dlon = (lon_grid - lon_q + 180.0) % 360.0 - 180.0
    dist2 = dlat * dlat + (dlon * np.cos(np.deg2rad(lat))) ** 2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    ny, nx = lat_grid.shape
    y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
    x0, x1 = max(0, ix - half), min(nx, ix + half + 1)
    return ds.isel(y=slice(y0, y1), x=slice(x0, x1))


def _box_to_rows(
    box: xr.Dataset,
    var_name: str,
    out_prefix: str,
    cycle: datetime,
    lead_hour: int,
) -> list[dict[str, Any]]:
    """Flatten a variable's 2-D slice into one row per grid point."""
    arr = box[var_name].values
    lat = box["latitude"].values
    lon = box["longitude"].values
    ny, nx = arr.shape
    rows: list[dict[str, Any]] = []
    for iy in range(ny):
        for ix in range(nx):
            rows.append(
                {
                    "cycle_utc": cycle,
                    "lead_hour": lead_hour,
                    "valid_utc": cycle + timedelta(hours=lead_hour),
                    "iy": iy,
                    "ix": ix,
                    "latitude": float(lat[iy, ix]),
                    "longitude": float(lon[iy, ix]),
                    out_prefix: float(arr[iy, ix]),
                }
            )
    return rows


def _fetch_lead(
    cycle_naive_utc: datetime,
    cycle_aware: datetime,
    lead: int,
    *,
    airport: Airport,
    variables: Iterable[HRRRVariableSpec],
    grid_half: int,
) -> pd.DataFrame | None:
    """Fetch every variable for one (cycle, lead) into one frame.

    Variables are pulled with one Herbie instance, so the .idx file is shared
    across calls. Each variable still costs a byte-range fetch — the big
    speedup comes from running this function in parallel across leads.
    """
    from herbie import Herbie  # deferred import

    # verbose=False suppresses Herbie's "Downloading inventory file from..."
    # prints. We can't safely redirect sys.stdout here because this runs in
    # many threads concurrently and they would race on the global handle.
    H = Herbie(cycle_naive_utc, model="hrrr", product="sfc", fxx=lead, verbose=False)
    per_var_rows: dict[tuple[int, int, int], dict[str, Any]] = {}

    for spec in variables:
        try:
            ds = H.xarray(spec.search)
        except Exception as exc:
            # Per-variable failures are expected during backfills (missing
            # GRIB on AWS, herbie subset edge cases on early cycles). Log to
            # file at INFO; the ribbon's `failed` counter is the user-facing
            # signal. Bump to WARNING if it ever blocks operational use.
            logger.info(
                "HRRR fetch failed cycle=%s lead=%d var=%s: %s",
                cycle_aware.isoformat(), lead, spec.name, exc,
            )
            continue
        box = _nearest_grid_box(ds, airport.latitude, airport.longitude, grid_half)
        data_vars = list(box.data_vars)
        if not data_vars:
            continue
        rows = _box_to_rows(box, data_vars[0], spec.out_prefix, cycle_aware, lead)
        for row in rows:
            key = (lead, row["iy"], row["ix"])
            merged = per_var_rows.setdefault(
                key,
                {
                    "cycle_utc": row["cycle_utc"],
                    "lead_hour": row["lead_hour"],
                    "valid_utc": row["valid_utc"],
                    "iy": row["iy"],
                    "ix": row["ix"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                },
            )
            merged[spec.out_prefix] = row[spec.out_prefix]

    if not per_var_rows:
        return None
    return pd.DataFrame(list(per_var_rows.values()))


def fetch_cycle(
    cycle: datetime,
    *,
    airport: Airport,
    lead_hours: Iterable[int] = DEFAULT_LEAD_HOURS,
    variables: Iterable[HRRRVariableSpec] = HRRR_VARIABLES,
    grid_half: int = DEFAULT_GRID_HALF,
    max_workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Fetch one HRRR init cycle and return a long/wide frame keyed by
    `(cycle_utc, lead_hour, iy, ix)` with one column per variable.

    Lead hours are fetched in parallel with a `ThreadPoolExecutor` because
    each (cycle, lead) is independent, network I/O bound, and Herbie/cfgrib
    release the GIL during downloads and GRIB decoding.
    """
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    # Herbie compares its internal date against a tz-naive pd.Timestamp.utcnow()
    # in core.py::_validate, so it rejects tz-aware inputs. Pass a naive UTC
    # datetime; we keep the tz-aware `cycle` for our own output rows.
    cycle_naive_utc = cycle.astimezone(timezone.utc).replace(tzinfo=None)

    leads = list(lead_hours)
    var_list = list(variables)
    workers = max(1, min(max_workers, len(leads)))

    per_lead: list[pd.DataFrame] = []
    if workers == 1:
        for lead in leads:
            frame = _fetch_lead(
                cycle_naive_utc, cycle, lead,
                airport=airport, variables=var_list, grid_half=grid_half,
            )
            if frame is not None:
                per_lead.append(frame)
                logger.debug("cycle=%s lead=%d done", cycle.isoformat(), lead)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fetch_lead, cycle_naive_utc, cycle, lead,
                    airport=airport, variables=var_list, grid_half=grid_half,
                ): lead
                for lead in leads
            }
            for fut in as_completed(futures):
                lead = futures[fut]
                try:
                    frame = fut.result()
                except Exception as exc:
                    logger.info(
                        "HRRR lead failed cycle=%s lead=%d: %s",
                        cycle.isoformat(), lead, exc,
                    )
                    continue
                if frame is not None:
                    per_lead.append(frame)
                    logger.debug("cycle=%s lead=%d done", cycle.isoformat(), lead)

    if not per_lead:
        return pd.DataFrame()
    return (
        pd.concat(per_lead, ignore_index=True)
        .sort_values(["lead_hour", "iy", "ix"])
        .reset_index(drop=True)
    )


def cycle_path(cycle: datetime, airport: Airport, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    cycle_utc = cycle.astimezone(timezone.utc) if cycle.tzinfo else cycle.replace(tzinfo=timezone.utc)
    return (
        airport.raw_hrrr_dir(data_root)
        / f"{cycle_utc:%Y}"
        / f"{cycle_utc:%Y%m%d_%HZ}.parquet"
    )


def _process_cycle(
    cycle: datetime,
    *,
    airport: Airport,
    lead_hours: Iterable[int],
    variables: Iterable[HRRRVariableSpec],
    grid_half: int,
    skip_existing: bool,
    data_root: Path,
    max_workers: int,
) -> CycleResult:
    """Fetch one cycle and write its Parquet. Reports status + row count."""
    path = cycle_path(cycle, airport, data_root)
    if skip_existing and path.exists():
        logger.debug("skip existing %s", path.name)
        return CycleResult(cycle, path, "skipped", 0)
    t0 = datetime.now(tz=timezone.utc)
    logger.debug("cycle %s", cycle.isoformat())
    df = fetch_cycle(
        cycle,
        airport=airport,
        lead_hours=lead_hours,
        variables=variables,
        grid_half=grid_half,
        max_workers=max_workers,
    )
    elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    if df.empty:
        logger.info("no data for cycle %s", cycle.isoformat())
        return CycleResult(cycle, None, "empty", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.debug("wrote %d rows to %s in %.1fs", len(df), path.name, elapsed)
    return CycleResult(cycle, path, "written", len(df))


def ingest_airport(
    airport: Airport,
    *,
    start: date | datetime,
    end: date | datetime,
    lead_hours: Iterable[int] = DEFAULT_LEAD_HOURS,
    variables: Iterable[HRRRVariableSpec] = HRRR_VARIABLES,
    grid_half: int = DEFAULT_GRID_HALF,
    cycle_step_hours: int = 1,
    skip_existing: bool = True,
    data_root: Path = DEFAULT_DATA_ROOT,
    max_workers: int = DEFAULT_WORKERS,
    cycle_workers: int = DEFAULT_CYCLE_WORKERS,
    on_progress: ProgressCallback | None = None,
) -> IngestSummary:
    """Ingest HRRR for every init cycle in `[start, end)`.

    Writes one Parquet per cycle under `data/raw/hrrr/{icao}/{YYYY}/` and
    returns an `IngestSummary` with counts (written / skipped / failed /
    empty) and the total row count.

    Concurrency is two-tiered: `cycle_workers` cycles are fetched in parallel,
    and within each cycle `max_workers` lead hours are fetched in parallel.
    Peak in-flight lead fetches is ~`cycle_workers * max_workers`.

    `on_progress(done, total, counters)` is called once before any cycle
    runs and again after each cycle finishes. `counters` mirrors the running
    `IngestSummary` (written / skipped / failed / empty / rows) so the CLI
    can show a live summary.
    """
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    if isinstance(end, date) and not isinstance(end, datetime):
        end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)

    cycles = list(iter_cycles(start, end, cycle_step_hours))
    total = len(cycles)
    cycle_pool_size = max(1, min(cycle_workers, total)) if total else 1
    logger.info(
        "HRRR ingest: airport=%s cycles=%d leads=%s cycle_workers=%d lead_workers=%d",
        airport.icao, total, list(lead_hours), cycle_pool_size, max_workers,
    )

    def _run(cycle: datetime) -> CycleResult:
        return _process_cycle(
            cycle,
            airport=airport,
            lead_hours=lead_hours,
            variables=variables,
            grid_half=grid_half,
            skip_existing=skip_existing,
            data_root=data_root,
            max_workers=max_workers,
        )

    summary = IngestSummary(total=total)
    done = 0

    def _counters() -> dict[str, Any]:
        return {
            "written": summary.written,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "empty": summary.empty,
            "rows": summary.rows,
        }

    if on_progress is not None:
        on_progress(0, total, _counters())

    def _record(cycle: datetime, result: CycleResult | None, exc: Exception | None) -> None:
        nonlocal done
        if exc is not None:
            summary.failed += 1
            logger.info("cycle %s failed: %s", cycle.isoformat(), exc)
        else:
            assert result is not None
            if result.status == "written":
                summary.written += 1
                summary.rows += result.rows
            elif result.status == "skipped":
                summary.skipped += 1
            else:
                summary.empty += 1
        done += 1
        if on_progress is not None:
            on_progress(done, total, _counters())

    if cycle_pool_size == 1:
        for cycle in cycles:
            try:
                result = _run(cycle)
            except Exception as exc:
                _record(cycle, None, exc)
                continue
            _record(cycle, result, None)
    else:
        with ThreadPoolExecutor(max_workers=cycle_pool_size) as pool:
            futures = {pool.submit(_run, c): c for c in cycles}
            for fut in as_completed(futures):
                cycle = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    _record(cycle, None, exc)
                    continue
                _record(cycle, result, None)

    logger.info(
        "HRRR ingest done: airport=%s new=%d skip=%d fail=%d empty=%d rows=%d",
        airport.icao,
        summary.written,
        summary.skipped,
        summary.failed,
        summary.empty,
        summary.rows,
    )
    return summary
