"""Airport configuration: one YAML per airport, loaded into a pydantic model.

Every CLI command takes `--airport ICAO` and loads the matching
`config/airports/<ICAO>.yaml`. No lat/lon, runway heading, or station list
should ever be hardcoded inside `src/`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_DIR = Path("config/airports")
DEFAULT_DATA_ROOT = Path("data")


class Runway(BaseModel):
    """A single runway. `heading_deg_true` is the true heading of the low-numbered end."""

    model_config = ConfigDict(extra="forbid")

    id: str
    heading_deg_true: int

    @field_validator("heading_deg_true")
    @classmethod
    def _heading_in_range(cls, v: int) -> int:
        if not 0 <= v < 360:
            raise ValueError(f"heading_deg_true must be in [0, 360), got {v}")
        return v


class Airport(BaseModel):
    """Typed representation of a `config/airports/<ICAO>.yaml` file."""

    model_config = ConfigDict(extra="forbid")

    icao: str
    name: str
    latitude: float
    longitude: float
    elevation_ft: int
    timezone: str
    runways: list[Runway]
    neighbor_stations: list[str] = Field(default_factory=list)
    history_start: date | None = None

    @field_validator("icao")
    @classmethod
    def _icao_upper(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 4 or not v.isalnum():
            raise ValueError(f"icao must be a 4-character alphanumeric code, got {v!r}")
        return v

    @field_validator("latitude")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"latitude out of range: {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def _lon_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"longitude out of range: {v}")
        return v

    @field_validator("neighbor_stations")
    @classmethod
    def _stations_upper(cls, v: list[str]) -> list[str]:
        return [s.strip().upper() for s in v]

    @classmethod
    def load(cls, icao: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> Airport:
        """Load the YAML for `icao` from `config_dir`."""
        path = config_dir / f"{icao.upper()}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No airport config at {path}")
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def list_all(cls, config_dir: Path = DEFAULT_CONFIG_DIR) -> list[Airport]:
        """Load every airport YAML in `config_dir`, sorted by ICAO."""
        return sorted(
            (cls.load(p.stem, config_dir) for p in config_dir.glob("*.yaml")),
            key=lambda a: a.icao,
        )

    def raw_metar_dir(self, root: Path = DEFAULT_DATA_ROOT) -> Path:
        return root / "raw" / "metar" / self.icao

    def raw_hrrr_dir(self, root: Path = DEFAULT_DATA_ROOT) -> Path:
        return root / "raw" / "hrrr" / self.icao

    def features_dir(self, root: Path = DEFAULT_DATA_ROOT) -> Path:
        return root / "features" / self.icao

    def models_dir(self, root: Path = DEFAULT_DATA_ROOT) -> Path:
        return root / "models" / self.icao

    def all_stations(self) -> list[str]:
        """Target ICAO followed by neighbor stations, preserving order, deduped."""
        seen: set[str] = set()
        out: list[str] = []
        for s in [self.icao, *self.neighbor_stations]:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
