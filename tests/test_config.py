"""Config loader tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from wind_forecast.config import Airport

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config" / "airports"


def test_kman_loads() -> None:
    a = Airport.load("KMAN", CONFIG_DIR)
    assert a.icao == "KMAN"
    assert a.name.startswith("Nampa")
    assert -90 <= a.latitude <= 90
    assert -180 <= a.longitude <= 180
    assert a.runways and all(0 <= r.heading_deg_true < 360 for r in a.runways)
    assert "KBOI" in a.neighbor_stations


def test_kboi_loads() -> None:
    a = Airport.load("KBOI", CONFIG_DIR)
    assert a.icao == "KBOI"
    assert "KMAN" in a.neighbor_stations


def test_icao_normalizes_case() -> None:
    a = Airport.load("kman", CONFIG_DIR)
    assert a.icao == "KMAN"


def test_missing_airport_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Airport.load("XXXX", CONFIG_DIR)


def test_list_all_includes_both() -> None:
    airports = Airport.list_all(CONFIG_DIR)
    icaos = {a.icao for a in airports}
    assert {"KMAN", "KBOI"}.issubset(icaos)


def test_all_stations_dedupes_target() -> None:
    a = Airport.load("KMAN", CONFIG_DIR)
    stations = a.all_stations()
    assert stations[0] == "KMAN"
    assert len(stations) == len(set(stations))


def test_data_dirs_use_icao_partition(tmp_path: Path) -> None:
    a = Airport.load("KMAN", CONFIG_DIR)
    assert a.raw_metar_dir(tmp_path) == tmp_path / "raw" / "metar" / "KMAN"
    assert a.raw_hrrr_dir(tmp_path) == tmp_path / "raw" / "hrrr" / "KMAN"
    assert a.features_dir(tmp_path) == tmp_path / "features" / "KMAN"
    assert a.models_dir(tmp_path) == tmp_path / "models" / "KMAN"


def test_rejects_bad_heading(tmp_path: Path) -> None:
    bad = {
        "icao": "KTST",
        "name": "Test",
        "latitude": 40.0,
        "longitude": -100.0,
        "elevation_ft": 1000,
        "timezone": "UTC",
        "runways": [{"id": "10/28", "heading_deg_true": 999}],
    }
    p = tmp_path / "KTST.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="heading_deg_true"):
        Airport.load("KTST", tmp_path)


def test_history_start_parses_as_date() -> None:
    a = Airport.load("KMAN", CONFIG_DIR)
    assert a.history_start == date(2015, 1, 1)


def test_rejects_extra_fields(tmp_path: Path) -> None:
    bad = {
        "icao": "KTST",
        "name": "Test",
        "latitude": 40.0,
        "longitude": -100.0,
        "elevation_ft": 1000,
        "timezone": "UTC",
        "runways": [{"id": "10/28", "heading_deg_true": 100}],
        "unexpected_field": True,
    }
    p = tmp_path / "KTST.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        Airport.load("KTST", tmp_path)
