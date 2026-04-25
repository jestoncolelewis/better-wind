"""Tests for METAR CSV parsing and output schema.

Network-free: exercises the parser against fixed CSV text. Schema stability
across stations/airports is part of the Phase 1 acceptance criterion.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wind_forecast.ingest.metar import OUTPUT_COLUMNS, date_chunks, parse_csv

CSV_HEADER = "station,valid,drct,sknt,gust,tmpf,dwpf,alti,mslp,vsby,metar"


def _csv(*rows: str) -> str:
    return "\n".join([CSV_HEADER, *rows]) + "\n"


def test_parse_basic_row() -> None:
    text = _csv("KMAN,2024-01-01 00:00,360,10,,30.0,20.0,30.10,1020.0,10.0,METAR KMAN 010000Z 36010KT")
    df = parse_csv(text)

    assert list(df.columns) == list(OUTPUT_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station"] == "KMAN"
    # North wind 10kt -> u=0, v=-10
    assert np.isclose(row["u"], 0.0, atol=1e-9)
    assert np.isclose(row["v"], -10.0)


def test_missing_wind_yields_null_uv() -> None:
    text = _csv("KMAN,2024-01-01 00:00,M,M,,30.0,20.0,30.10,1020.0,10.0,")
    df = parse_csv(text)
    assert len(df) == 1
    assert df["u"].isna().all()
    assert df["v"].isna().all()


def test_duplicate_timestamps_deduped() -> None:
    text = _csv(
        "KMAN,2024-01-01 00:00,360,10,,30.0,20.0,30.10,1020.0,10.0,first",
        "KMAN,2024-01-01 00:00,360,12,,30.0,20.0,30.10,1020.0,10.0,second",
    )
    df = parse_csv(text)
    assert len(df) == 1
    # keep=last -> the 12 kt observation wins
    assert df.iloc[0]["sknt"] == 12


def test_sorted_by_valid_utc() -> None:
    text = _csv(
        "KMAN,2024-01-01 02:00,180,5,,50.0,40.0,30.00,1015.0,10.0,",
        "KMAN,2024-01-01 00:00,180,5,,50.0,40.0,30.00,1015.0,10.0,",
        "KMAN,2024-01-01 01:00,180,5,,50.0,40.0,30.00,1015.0,10.0,",
    )
    df = parse_csv(text)
    assert df["valid_utc"].is_monotonic_increasing


def test_schema_stable_with_empty_input() -> None:
    df = parse_csv(CSV_HEADER + "\n")
    assert list(df.columns) == list(OUTPUT_COLUMNS)
    assert len(df) == 0


def test_station_override_normalizes_faa_to_icao() -> None:
    # Mesonet returns 3-letter FAA codes (e.g. "MAN") even when we query KMAN.
    # The override forces the partition key back to the requested ICAO.
    text = _csv("MAN,2024-01-01 00:00,360,10,,30.0,20.0,30.10,1020.0,10.0,KMAN 010000Z 36010KT")
    df = parse_csv(text, station_override="KMAN")
    assert (df["station"] == "KMAN").all()


def test_schema_identical_across_stations() -> None:
    kman = parse_csv(_csv("KMAN,2024-01-01 00:00,360,10,,30.0,20.0,30.10,1020.0,10.0,"))
    kboi = parse_csv(_csv("KBOI,2024-01-01 00:00,180,15,20,45.0,30.0,30.10,1020.0,10.0,"))
    assert list(kman.columns) == list(kboi.columns)
    assert kman.dtypes.equals(kboi.dtypes)


def test_date_chunks_tiles_full_range_without_overlap() -> None:
    chunks = date_chunks(date(2020, 1, 1), date(2022, 12, 31), chunk_days=366)
    # Every chunk must be within bounds, contiguous (next start = prev end + 1),
    # and at most chunk_days long.
    assert chunks[0][0] == date(2020, 1, 1)
    assert chunks[-1][1] == date(2022, 12, 31)
    for (s, e) in chunks:
        assert (e - s).days < 366
        assert s <= e
    for prev, nxt in zip(chunks[:-1], chunks[1:], strict=True):
        assert (nxt[0] - prev[1]).days == 1


def test_date_chunks_single_chunk_when_range_fits() -> None:
    assert date_chunks(date(2024, 1, 1), date(2024, 1, 31), chunk_days=366) == [
        (date(2024, 1, 1), date(2024, 1, 31)),
    ]


def test_date_chunks_empty_when_end_before_start() -> None:
    assert date_chunks(date(2024, 6, 1), date(2024, 5, 1)) == []
