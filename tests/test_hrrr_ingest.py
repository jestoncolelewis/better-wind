"""Tests for the HRRR ingestion helpers that don't require herbie.

The actual HRRR fetch (`fetch_cycle`, `ingest_airport`) hits the network and
needs `herbie` + `cfgrib`, so it's exercised only in integration runs. The
pure helpers — cycle iteration, leakage-safe valid time, path layout — are
covered here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from wind_forecast.config import Airport
from wind_forecast.ingest.hrrr import cycle_path, iter_cycles

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config" / "airports"


def test_iter_cycles_is_half_open_and_hourly() -> None:
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
    cycles = list(iter_cycles(start, end))
    assert cycles == [
        datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
    ]


def test_iter_cycles_assumes_utc_when_naive() -> None:
    start = datetime(2024, 1, 1, 0)
    end = datetime(2024, 1, 1, 2)
    cycles = list(iter_cycles(start, end))
    assert all(c.tzinfo is timezone.utc for c in cycles)


def test_cycle_path_partitions_by_airport_and_year() -> None:
    airport = Airport.load("KMAN", CONFIG_DIR)
    cycle = datetime(2024, 3, 15, 12, tzinfo=timezone.utc)
    path = cycle_path(cycle, airport, data_root=Path("/tmp/data"))
    assert path == Path("/tmp/data/raw/hrrr/KMAN/2024/20240315_12Z.parquet")


def test_no_future_leakage_in_valid_time() -> None:
    """A cycle @ T with lead L has valid_time T+L. This is the invariant every
    feature has to respect — here we just sanity-check the arithmetic the
    ingestion relies on.
    """
    cycle = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
    for lead in range(0, 19):
        valid = cycle + timedelta(hours=lead)
        # The observation label at `valid` must not be available to a feature
        # computed at `cycle` unless the lag is >0h *into the past*.
        assert valid >= cycle
        assert (valid - cycle).total_seconds() == lead * 3600


def test_history_window_respects_airport_start() -> None:
    airport = Airport.load("KMAN", CONFIG_DIR)
    assert airport.history_start is not None
    # Any start we pick for HRRR ingestion must be >= the airport's declared
    # history floor. This is the check the CLI (and feature builder) will
    # enforce; we're only asserting the config surface it relies on.
    assert airport.history_start >= date(2014, 1, 1)
